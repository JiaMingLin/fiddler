import copy
import math
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers


class FiddlerMiniCPM:
    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        self.causal_lm = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            trust_remote_code=True, # 這是重點！沒有加的話 HF 會無法載入模型代碼
            use_cache=True,
        )
        self.lm_head = self.causal_lm.lm_head
        self.model = self.causal_lm.model
        self.moe_attr_name = self._detect_moe_attr_name()
        self.expert_placeholder = copy.deepcopy(
            self._get_moe(self.model.layers[0]).experts[0]
        ).to(self.dev)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0
        self.cpu_offload = args.cpu_offload
        self.beam_width = args.beam_width
        self.n_layer = len(self.model.layers)
        self.n_expert = len(self._get_moe(self.model.layers[0]).experts)
       
        # TODO: find this value based on device config
        self.latency_cpu = 7
        self.latency_gpu = 70

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        self.bring_non_expert_to_gpu()

        # 0: CPU, 1: GPU
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        # n_expert_on_gpu = self.calc_n_expert_on_gpu()
        n_expert_on_gpu = 80  # 你可以改成 0 或其他數字來測
        print(
            f"Number of experts on GPU: {n_expert_on_gpu}/{self.n_layer * self.n_expert}"
        )

        self.set_expert_loc(n_expert_on_gpu)
        # print(self.expert_loc)

        self.bring_expert_to_gpu()

        print("Model is ready.")

    def bring_non_expert_to_gpu(self):
        """Bring non-expert layers to GPU"""
        self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        for i in range(len(self.model.layers)):
            self.model.layers[i].self_attn.to(self.dev)
            self.model.layers[i].input_layernorm.to(self.dev)
            self._get_moe(self.model.layers[i]).gate.to(self.dev)
            self.model.layers[i].post_attention_layernorm.to(self.dev)
            # only moe.experts is on CPU

    def _detect_moe_attr_name(self):
        first_layer = self.model.layers[0]
        # MiniCPM / Mixtral implementations use different names for MoE module.
        candidates = ["block_sparse_moe", "moe", "mlp"]
        for attr in candidates:
            if not hasattr(first_layer, attr):
                continue
            moe_module = getattr(first_layer, attr)
            if hasattr(moe_module, "experts") and hasattr(moe_module, "gate"):
                return attr
        raise AttributeError(
            "Cannot locate MoE module on decoder layer. "
            "Expected one of block_sparse_moe/moe/mlp with experts+gate."
        )

    def _get_moe(self, layer):
        return getattr(layer, self.moe_attr_name)

    def _run_expert_module(self, expert_module, hidden_states, routing_weights):
        """
        Support both expert signatures:
        - Mixtral style: expert(hidden_states, routing_weights)
        - MiniCPM style: expert(hidden_states), then apply routing weights outside
        """
        try:
            return expert_module(hidden_states, routing_weights)
        except TypeError as exc:
            if "positional arguments" not in str(exc):
                raise
            return expert_module(hidden_states) * routing_weights

    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        """Set the location of experts"""
        if popular_experts is None:
            # list of (i_layer, i_expert) in the order of popularity
            # determined based on profile
            popular_experts = [
                (l, e) for l in range(self.n_layer) for e in range(self.n_expert)
            ]

        for i in range(n_expert_on_gpu):
            i_layer, i_expert = popular_experts[i]
            self.expert_loc[i_layer, i_expert] = 1

    def bring_expert_to_gpu(self):
        """Bring part of expert layers to GPU"""
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if self.is_expert_in_gpu(i, j):
                    self._get_moe(self.model.layers[i]).experts[j].to(self.dev)

    def is_expert_in_gpu(self, i_layer, i_expert):
        """Determine if the expert is in GPU"""
        return self.expert_loc[i_layer, i_expert] == 1

    def calc_n_expert_on_gpu(self):
        """Get the number of experts that we can put on GPU"""
        # get the number of parameters of one expert
        n_param = sum(
            p.numel()
            for p in self._get_moe(self.model.layers[0]).experts[0].parameters()
        )
        # get the amount of free memory on GPU
        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        free_mem = total_mem * 0.95 - torch.cuda.memory_allocated(self.dev) # TODO: magic number
        return int((free_mem) // (n_param * 2))

    def initial_beam_tensor(self, input_tensor):
        # transpose tensor of shape (beam_width, seq_len, beam_width) to (beam_width, 1) properly
        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        row_idx = torch.tensor(
            [i * self.beam_width for i in range(input_tensor.shape[0] // self.beam_width)]
        )
        output_tensor = input_tensor[row_idx].view(-1, 1)
        return output_tensor

    def generate(self, text=None, output_token=20, input_token=None):
        torch.set_num_threads(16) # TODO: set appropriately
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        
        input_ids, position_ids = self.tokenize(text)

        if input_token is not None:
            input_ids = input_ids[:, :input_token]
            position_ids = position_ids[:, :input_token]

        tick = time.time()
        is_decode = False
        prefill_time, decode_time = 0, 0
        generated_token_ids = [[] for _ in range(input_ids.shape[0])]
        search_start = False
        probs = torch.full((input_ids.shape[0], 1), 1.0)

        for i_token in range(output_token):
            logits = self.mixtral_forward(input_ids, position_ids, is_decode)

            logits = logits.to("cpu")
            # logits.shape: (batch_size, seq_len, vocab_size)

            # normalize logits
            logits = F.softmax(logits, dim=-1)

            # greedy search:
            # output = torch.argmax(logits, dim=-1)

            # beam_search:
            self.past_key_values_length += logits.shape[1]
            if search_start:
                new_probs, output = torch.topk(logits, 1, dim=-1)
                new_probs = new_probs[:, -1].flatten().view(-1, 1)
            else:
                new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                new_probs = self.initial_beam_tensor(new_probs)
                output = self.initial_beam_tensor(output)
                search_start = True
            # new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
            probs = probs * new_probs

            input_ids = output[:, -1].flatten().view(-1, 1).to(self.dev)
            # input_ids.shape: (batch_size, seq_len=1)
            for i in range(input_ids.shape[0]):
                generated_token_ids[i].append(int(input_ids[i, 0].item()))

            position_ids = (
                torch.arange(
                    self.past_key_values_length,
                    self.past_key_values_length + 1,
                    dtype=torch.long,
                    device=self.dev,
                )
                .unsqueeze(0)
                .view(-1, 1)
            )
            # position_ids.shape: (1, 1)
            if not is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            is_decode = True
        decode_time = time.time() - tick
        probs = probs.view(-1, self.beam_width)
        max_ids = torch.argmax(probs, dim=-1)
        best_idx = int(max_ids[0].item())
        output_text = self.tokenizer.decode(
            generated_token_ids[best_idx],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        print("--------------------")
        print(f"Input: {text}")
        print(f"Output: {output_text}")

        return (
            prefill_time,
            decode_time,
            self.cnt_expert_hit / self.cnt_expert_all,
        )

    def tokenize(self, text):
        input_ids = []
        encodings = self.tokenizer(text, return_tensors="pt")
        input_id = encodings.input_ids.to(self.dev)
        for i in range(self.beam_width):
            input_ids.append(input_id[0])
        
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.dev)

        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.dev
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])

        return input_ids, position_ids

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, is_decode):
        hidden_dim = self.model.config.hidden_size
        inps = input_ids.to(self.dev)
        inps = self.model.embed_tokens(inps) * self.model.config.scale_emb

        for i_layer, layer in enumerate(self.model.layers):
            original_inps_shape = inps.shape

            inps_residual = inps
            inps = layer.input_layernorm(inps)
            inps, self_attn_weights, present_key_value = layer.self_attn(
                inps,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
            )
            # inps.shape: (batch_size, seq_len/token_num, embed_dim)
            scale = layer.scale_depth / math.sqrt(layer.num_hidden_layers)
            inps = inps_residual + inps * scale
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)
            inps = inps.view(-1, hidden_dim)
            # inps.shape: (batch_size*seq_len*embed_dim/hidden_dim, hidden_dim)
            moe_layer = self._get_moe(layer)
            router_logits = moe_layer.gate(inps)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            # routing_weights.shape: (batch_size*seq_len, num_experts)
            n_selected_experts = getattr(moe_layer, "num_experts_per_tok", 2)
            routing_weights, selected_experts = torch.topk(
                routing_weights, n_selected_experts, dim=-1
            )
            # routing_weights.shape: (batch_size*seq_len, 2)
            # selected_experts.shape: (batch_size*seq_len, 2)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(inps.dtype)

            # intermediate variable to store the output of experts
            inps_after_experts = torch.zeros_like(inps, device=self.dev)
            experts = moe_layer.experts

            if self.cpu_offload == 0:
                # baseline: do everything at GPU
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=self.n_expert
                ).permute(2, 1, 0)

                for i_expert in range(len(experts)):
                    is_cuda = self.is_expert_in_gpu(i_layer, i_expert)
                    idx, top_2 = torch.where(expert_mask[i_expert])

                    if top_2.shape[0] == 0:
                        # print(f"Expert {i_expert}: has no tokens")
                        continue

                    # torch.cuda.synchronize()
                    top_2_list = top_2.tolist()
                    idx_list = idx.tolist()

                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    if not is_cuda:
                        self.expert_placeholder.load_state_dict(
                            experts[i_expert].state_dict()
                        )
                        current_state = self._run_expert_module(
                            self.expert_placeholder,
                            current_state,
                            routing_weights[top_2_list, idx_list, None],
                        )
                    else:
                        current_state = self._run_expert_module(
                            experts[i_expert],
                            current_state,
                            routing_weights[top_2_list, idx_list, None],
                        )
                    inps_after_experts.index_add_(
                        0, top_2, current_state.to(inps.dtype)
                    )

                    if not is_cuda:
                        experts[i_expert] = experts[i_expert].to("cpu")

                    # end of one expert

            else:
                # prefill stage with offloading
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=self.n_expert
                ).permute(2, 1, 0)

                # first, calculate the number of tokens for each expert
                idxs, top_2s = [], []
                cost_per_expert = np.zeros(
                    (len(experts), 2), dtype=float
                )  # 0: CPU, 1: GPU
                for i_expert in range(len(experts)):
                    idx, top_2 = torch.where(expert_mask[i_expert])
                    idxs.append(idx)
                    top_2s.append(top_2)
                    # expected latency at CPU: number of token * cost_at_cpu
                    # expected latency at GPU: cost_at_gpu (constant)
                    cost_per_expert[i_expert, 0] = top_2.shape[0] * self.latency_cpu
                    cost_per_expert[i_expert, 1] = self.latency_gpu
                    if self.is_expert_in_gpu(i_layer, i_expert):
                        # if the expert is in GPU, the latency at GPU is
                        # approximately 0
                        cost_per_expert[i_expert, 1] = 0
                        self.cnt_expert_hit += top_2.shape[0]
                    self.cnt_expert_all += top_2.shape[0]
                
                # second, partition experts processing between CPU and GPU so that we can minimize:
                # max(sum of cost at CPU, sum of cost at GPU)
                # greedy algorithm is just as there are only 8 experts for Mixtral
                best_config = -1
                best_cost = float("inf")
                for config in range(1 << len(experts)):
                    sum_cost = 0
                    for i_expert in range(len(experts)):
                        if (config >> i_expert) & 1:
                            sum_cost += cost_per_expert[i_expert, 0]
                        else:
                            sum_cost += cost_per_expert[i_expert, 1]
                    if sum_cost < best_cost:
                        best_cost = sum_cost
                        best_config = config

                # then, we can offload the experts according to the best
                # configuration
                cpu_experts = []
                gpu_experts = []
                for i_expert in range(self.n_expert):
                    if (best_config >> i_expert) & 1:
                        cpu_experts.append(i_expert)
                    else:
                        gpu_experts.append(i_expert)

                for i_expert in gpu_experts:
                    top_2_list = top_2s[i_expert].tolist()
                    idx_list = idxs[i_expert].tolist()
                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    if self.is_expert_in_gpu(i_layer, i_expert):
                        current_state = self._run_expert_module(
                            experts[i_expert],
                            current_state,
                            routing_weights[top_2_list, idx_list, None],
                        )
                    else:
                        self.expert_placeholder.load_state_dict(
                            experts[i_expert].state_dict()
                        )
                        current_state = self._run_expert_module(
                            self.expert_placeholder,
                            current_state,
                            routing_weights[top_2_list, idx_list, None],
                        )
                    inps_after_experts.index_add_(
                        0,
                        top_2s[i_expert].to(self.dev, non_blocking=True),
                        current_state.to(self.dev, non_blocking=True),
                    )

                for i_expert in cpu_experts:
                    top_2_list = top_2s[i_expert].tolist()
                    idx_list = idxs[i_expert].tolist()
                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    current_state = self.run_expert_at_cpu(
                        i_layer,
                        i_expert,
                        current_state.to("cpu"),
                        routing_weights[top_2_list, idx_list, None].to("cpu"),
                    )
                    inps_after_experts.index_add_(
                        0,
                        top_2s[i_expert].to(self.dev, non_blocking=True),
                        current_state.to(self.dev, non_blocking=True),
                    )

            # addition because there's residual connection over moe layer
            inps = inps_residual + inps_after_experts.reshape(original_inps_shape) * scale

            # end of one layer

        inps = self.model.norm(inps)
        if hasattr(self.model.config, "dim_model_base"):
            inps = inps / (
                self.model.config.hidden_size / self.model.config.dim_model_base
            )
        lm_logis = self.lm_head(inps)

        self.present_key_value = present_key_value
        return lm_logis

    def run_expert_at_cpu(self, i_layer, i_expert, inps, routing_weights):
        """Run the expert at CPU"""
        return self._run_expert_module(
            self._get_moe(self.model.layers[i_layer]).experts[i_expert],
            inps,
            routing_weights,
        )
