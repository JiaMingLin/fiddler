# MiniCPM Output 異常排查 Execution Summary

日期：2026-04-14  
範圍：`src/fiddler/minicpm.py`、`src/fiddler/infer_minicpm.py`

## 1) 問題現象

- 初始執行 `infer_minicpm.py` 時，先遇到模型結構錯誤：
  - `MiniCPMDecoderLayer` 沒有 `block_sparse_moe` 欄位。
- 修正可執行後，生成結果異常，輸出大量重複片段（例如 `，2，2，...`），不符合預期自然語句。

## 2) 主要排查與修改紀錄

### A. 模型結構相容性（必要修正）

- 問題：程式沿用 Mixtral 命名，直接存取 `block_sparse_moe`。
- 修改：
  - 新增 MoE 欄位偵測機制（支援 `block_sparse_moe` / `moe` / `mlp`）。
  - 以 `_get_moe(layer)` 統一存取 gate/experts。
  - 將 expert 數量與 top-k 相關硬編碼改為動態值。
- 結果：可正常初始化 MiniCPM，不再因欄位名崩潰。

### B. Expert 呼叫簽名相容（必要修正）

- 問題：`MiniCPMMLP.forward()` 不接受第二個 `routing_weights` 參數。
- 修改：
  - 新增 `_run_expert_module(...)`：
    - 先嘗試 `expert(x, w)`（Mixtral 風格）。
    - 若簽名不符，改為 `expert(x) * w`（MiniCPM 風格）。
- 結果：消除 `TypeError`，可繼續前向。

### C. 輸出解碼方式（顯示層修正）

- 問題：原本逐 token 即時 decode 並串接，造成輸出碎片感嚴重。
- 修改：改為累積 `generated_token_ids`，最後一次 `tokenizer.decode(...)` 輸出。
- 結果：顯示行為合理化，但仍有內容品質問題（非純 decode 問題）。

### D. CPU bfloat16 假設檢查（已依需求還原）

- 曾做過：
  - CPU dtype 偵測與 fallback（`bfloat16 -> float32`）。
  - 對 CPU resident experts 先做 dtype 準備。
- 使用者提供 CPU 規格（Ryzen 9 9950X，含 `avx512_bf16`）後，已確認你要保留原策略。
- 現況：上述 CPU dtype 相關保護改動已還原，不再強制覆寫 CPU 計算 dtype。

### E. 對齊 MiniCPM 原生實作（關鍵品質修正）

比對 Hugging Face `modeling_minicpm.py` 後，補上兩個高影響差異：

1. Decoder residual scaling  
   - `residual + branch * (scale_depth / sqrt(num_hidden_layers))`  
   - 於 attention 與 MLP 殘差都對齊。

2. Embedding / LM head scaling  
   - `embed_tokens(...) * config.scale_emb`  
   - `lm_head` 前 `hidden_states / (hidden_size / dim_model_base)`。

## 3) 驗證結果（本次最後狀態）

- 程式可穩定執行，不再出現最早的結構錯誤與 expert 簽名錯誤。
- 輸出異常（`，2，2...`）已改善為不再該型態亂碼；目前偏向很快停住、輸出極短（接近空字串）。
- 這代表 forward 邏輯已比先前更接近正確，但生成策略仍可能過於保守（例如 greedy 易提早 EOS）。

## 4) 尚未變動的核心設計

- CPU/GPU collaborative execution 核心流程保留：
  - token -> router -> expert 分配 -> CPU/GPU 執行 -> 聚合 -> 殘差。
- 未移除你的分配策略與 offload 主要邏輯。

## 5) 建議下一步（不改動核心 offload 邏輯）

1. 先加生成參數避免過早 EOS：`min_new_tokens`。  
2. 支援可選 sampling：`do_sample / top_p / temperature`。  
3. 做「逐層 hidden state 對照」：
   - 同 prompt 比對官方 forward 與 custom forward 每層輸出差異（L2/最大誤差），快速定位首個發散層。  
4. 若要維持公平 benchmark，可分兩種模式：
   - `quality mode`（較穩定生成參數）
   - `latency mode`（維持現行設定，專注 offload 時延）

## 6) 本次結論

- 問題不只在 CPU 或 decode；主要是 MiniCPM 與 Mixtral 在 layer/scale/forward 細節不同，導致 custom path 數值偏移。
- 目前已完成「結構相容 + 關鍵縮放對齊 + 可執行」階段。
- 下一階段建議進入「逐層對照」與「生成策略調整」，以在保留 collaborative execution 的前提下恢復語句品質。
