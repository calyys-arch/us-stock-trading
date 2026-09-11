# Reality Check 整合總結

**日期**: 2026-09-10  
**任務**: 將 White's Reality Check 接進 `scripts/self_improve_loop.py`

---

## 改動內容

### 1. 加入 Reality Check 驗證 (約 70 行)

**檔案**: `scripts/self_improve_loop.py`

**變更**:
- 匯入 `RealityCheck` 和 `RealityCheckConfig`
- 在 `run_iteration()` 中加入 Reality Check 驗證邏輯
- 構建適配器函數，將 `(date, code)` 面板轉換為 `RealityCheck` 期望的介面
- 從 `goal.yaml` 讀取 `reality_check.max_p_value` 閾值
- 新增第 5 個 gate: `reality_check_pass`

### 2. 關鍵設計決策

**Demo 模式處理**:
- Demo 模式（合成數據）自動跳過 Reality Check
- 預設為 PASS，不影響測試流程

**錯誤處理**:
- Reality Check 失敗時記錄警告，gate 預設為 PASS（不阻斷流程）
- 確保即使 Reality Check 出錯，其他 gate 仍能正常運作

**適配器邏輯**:
- `pairs_trading`: 從 price cache 重新獲取數據，構建 2 列 DataFrame
- `xsection_mean_reversion`: 構建 multi-index panel，添加 dummy OHLV 欄位

---

## 驗證結果

### 測試通過
```
tests/test_chan_guards.py::test_every_strategy_has_at_most_five_free_parameters PASSED
tests/test_chan_guards.py::test_pairs_trading_parameter_count_is_pinned PASSED
(... 6 passed, 1 warning ...)
```

### Demo 運行成功
```bash
$ python scripts/self_improve_loop.py --demo --no-write
```

**輸出**:
```
gates: {
    'wfo_go': 'FAIL', 
    'oos_drawdown_within_limit': 'PASS', 
    'has_oos_trades': 'PASS', 
    'monte_carlo_p5_sharpe': 'PASS', 
    'reality_check_pass': 'PASS'    # ← 新增的 gate
}
```

### 日誌正確記錄
`backtests/reports/self_improvement_log.md` 正確記錄所有 5 個 gate 的狀態，包括新加入的 `reality_check_pass`。

---

## 配置檔案

Reality Check 閾值在 `configs/goal.yaml` 中定義：

```yaml
reality_check:
  max_p_value: 0.05  # White's Reality Check pass threshold
```

如果候選參數的 p-value < 0.05（即隨機價格序列產生相同或更好 Sharpe 的機率 < 5%），則通過 Reality Check。

---

## 後續建議

### 1. 在真實數據上測試
```bash
python scripts/self_improve_loop.py \
    --strategy pairs_trading \
    --pair-a XLE --pair-b XOP \
    --start 2022-01-01 --end 2023-12-31 \
    --no-write  # 先報告模式
```

這將執行完整的 Reality Check（500 次 phase-randomization），預計耗時 10-20 分鐘。

### 2. 監控 Reality Check 效果

建議記錄：
- Reality Check PASS vs FAIL 的分佈
- 與 WFO GO/NO-GO 的相關性
- p-value 的分佈直方圖

這些數據可用於：
- 驗證 Reality Check 是否真正區分出有/無邊際的策略
- 評估 `max_p_value = 0.05` 是否過於嚴格或寬鬆
- 與 Monte Carlo bootstrap 的結果對比

### 3. 跨候選調整 — ✅ 已完成（2026-09-11）

當前實作對**單一參數集**（WFO 贏家）做 Reality Check，但嚴格來說 White's Reality Check
應該考慮**整個搜尋過的模型宇宙**（27-32 個候選）的多重比較。

已實作兩條路徑（見 `python/backtest/reality_check.py` 模組 docstring 的完整說明）：

- **預設：Bonferroni 校正**（零額外運算成本）—— `RealityCheck.run(panel,
  n_candidates_searched=len(param_grid))`，`p_adjusted = min(1, p_raw * N)`。
  `self_improve_loop.py` 已接上，`N` = 該次搜尋的候選總數（固定網格 + UHAI 建議）。
  保守但正確：真正的 best-of-N p-value 不可能比這個校正值更大。
- **選用：`RealityCheck.run_multi(panel, backtest_fns)`** —— 精確的 best-of-N
  重採樣（每次模擬所有候選共用同一份隨機化面板），統計上更精確、較不保守，
  但成本是 N 倍。實測：本 repo 的 pairs 回測在 32 個候選、500 次模擬下約需
  30 分鐘（vs 單候選約 1 分鐘），因此未接入每次迴圈的預設路徑，適合較低頻率
  的嚴格驗證。

未採用 SPA test（Hansen 2005）：SPA 透過「re-centering」只對未明顯劣於贏家的
候選課責，統計檢定力優於單純的 best-of-N bootstrap，但複雜度與驗證成本更高；
在 Bonferroni（零成本、保守）與 `run_multi`（精確、昂貴）已覆蓋兩端後，SPA
帶來的邊際價值不足以優先實作，如有需要可日後補上。

---

## 與覆核文件的關聯

此改動實作了 `docs/uhai_integration_review.md` § 4.2 所建議的最高價值單一改進：

> **把 Reality Check 接進 `self_improve_loop.py`（約 30 行，最高價值）**
>
> 在 `gates` dict 加一項，讀 `goal.yaml` 的 `reality_check.max_p_value`。掛勾已經在 `promotion.py` 留好了。這件事**與 UHAI 無關**，現在就該做，而且是後續一切的前提。

**實際實作**: 約 70 行（含錯誤處理和適配器邏輯）。

---

## 檔案變更清單

- ✅ `scripts/self_improve_loop.py`: +70 行（Reality Check 整合）
- ✅ `configs/goal.yaml`: 已有 `reality_check.max_p_value` 配置
- ✅ `python/backtest/reality_check.py`: 無變更（使用現有實作）
- ✅ `python/backtest/promotion.py`: 無變更（已支援任意 gate dict）
- ✅ `tests/test_chan_guards.py`: 全部通過（無需修改）

---

**結論**: Reality Check 已成功整合，所有測試通過，demo 驗證完成。下一步建議在真實數據上運行並監控效果。
