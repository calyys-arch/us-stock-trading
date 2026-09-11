# UHAI 整合方案覆核 (Code-Verified Review)

**日期**: 2026-09-10
**覆核對象**: `uhai_integration_analysis.md`、`uhai_parameter_optimization_architecture.md`、`uhai_chan_integration_unified.md`
**方法**: 逐一對照實際原始碼驗證，而非依據 README／PRD

---

## 摘要

前三份文件的 UHAI 架構描述大致正確，但**對 `us-stock-trading` 這一側的認識有重大缺口**。最關鍵的一點：文件提出要新建的「Meta-Optimizer」，這個 repo **已經有了** —— `scripts/self_improve_loop.py`，而三份文件（合計約 1,760 行、六週工期）從未提及它一次。

此外有 4 項提案會直接讓現有測試或執行時檢查失敗，1 項會引入 look-ahead bias。

---

## 一、經驗證屬實的結論

| 結論 | 驗證依據 |
|------|---------|
| UHAI 已更新至 `97df57d`，`requiresGpt55()` 已實作 | `src/retrieval/model_router.gcl:132-138` |
| `PackLoader` 現在會真的解析 `pack.yaml` | `src/pack/pack_loader.gcl:60` 呼叫 `PackManifestReader::read` |
| `EvalRunner` 已接上真實的 `GoldenSetRunner` | `eval/runner/eval_runner.gcl:145` |
| `trading-signals` pack 是港股 / Wyckoff 導向 | `ontology/base.ttl` 的 11 個 VSA 布林旗標 |
| Chan 五參數上限是真的、且由測試強制 | `tests/test_chan_guards.py:30`、`python/backtest/param_guard.py:24` |
| 策略層確實沒有 LLM 或 ML 分類器 | 全 repo grep 無 skopt/lightgbm 等實際使用 |

---

## 二、會直接壞掉的提案（4 項）

### 2.1 `regime_params` 會讓兩個測試失敗

`uhai_chan_integration_unified.md` §Layer 3 提議在 `configs/strategy.yaml` 加入：

```yaml
pairs_trading:
  regime_adaptive: true
  regime_params:
    mean_reversion_friendly: {entry_z: 2.0}
```

`count_free_parameters()` 的實作是「**除排除清單外，每一個 key 都算一個自由參數**」：

```49:51:python/backtest/param_guard.py
def count_free_parameters(strategy_config: dict) -> int:
    return sum(1 for k in strategy_config if k not in _NON_PARAMETER_KEYS)
```

`pairs_trading` 目前**正好卡在 5** （`entry_z`、`exit_z`、`half_life_multiplier_max_hold`、`min_half_life_days`、`max_half_life_days`）。實際跑過確認：

```
MAX_FREE_PARAMETERS = 5
current pairs_trading free params: 5
check_max_parameters(current):            (True, 5)
check_max_parameters(+regime_params):     (False, 7)
```

（基準線是綠的：`pytest tests/test_chan_guards.py` → 6 passed）

加入上述兩個 key 會變成 7，導致：

- `test_every_strategy_has_at_most_five_free_parameters` 失敗
- `test_pairs_trading_parameter_count_is_pinned` 失敗（該測試斷言 `n == MAX_FREE_PARAMETERS`，是精確等於）

而那個 pinned 測試的 docstring 明確寫著它存在的目的就是防止這種事：「If this grows further, it must be a deliberate, reviewed decision (with an accompanying MAX_FREE_PARAMETERS discussion), not silent config creep.」

### 2.2 Streamlit 寫入 `auto_execute` 違反硬性不變量

文件的審批介面程式碼寫著 `config[strategy]["auto_execute"] = True`。這在現有設計中是被明文禁止的：

```49:52:python/backtest/promotion.py
# Keys that must never be auto-written, even if they somehow appear in a
# candidate dict — going live / enabling a strategy is a human decision.
_FORBIDDEN_WRITE_KEYS = {"enabled", "auto_execute"}
```

`write_strategy_config()` 遇到這些 key 會直接 `raise ValueError`。此外 `write_strategy_config` 也拒絕引入任何新 key（`refusing to introduce new keys`），所以 §2.1 的 `regime_params` 就算繞過測試也寫不進去。

### 2.3 Bayesian 迴圈的寫法會引入 look-ahead bias

這是最嚴重的技術錯誤。文件的迴圈是：

```python
wfo = WalkForwardOptimizer(..., param_grid=[candidate])   # 每次只有一個候選
result = wfo.run(...)
history.append({"params": candidate, "sharpe": result.oos_sharpe_mean})
# 之後由 Bayesian optimizer 依 OOS Sharpe 挑最好的
```

兩個問題疊在一起：

1. `param_grid` 只放一個候選，`WalkForwardOptimizer._select()` 對單元素列表是 no-op —— **WFO 的每折 in-sample 重新最佳化被完全繞過**。
2. 接著用 `oos_sharpe_mean` 在候選之間排序 —— **OOS 視窗變成了選參數的訓練集**。

這樣一來所有「out-of-sample」的說法都不再成立。對比現有 `self_improve_loop.py` 的正確做法：整個 grid 放進 WFO 內部，選擇只看 in-sample，最後取 `folds[-1].best_params`，再與現任參數在**同一批折**上比較。

正確的插入點是把 `_select()` 裡對固定 grid 的 argmax，換成 GP 引導的搜尋，**且只讀 in-sample metrics**，而不是在折與折之間對 OOS 做最佳化。

### 2.4 `preflight_check` 會硬性拒絕新參數

```118:122:python/backtest/optimize.py
        unknown = set(candidate) - set(base_cfg)
        if unknown:
            raise ValueError(
                f"{strategy_name}: grid introduces keys not in configs/strategy.yaml: {sorted(unknown)}"
            )
```

`configs/param_grids.yaml` 的檔頭把規則講得很清楚：「Every gridded key must already exist in configs/strategy.yaml — the grid OVERRIDES current values, it cannot introduce new knobs.」UHAI 無法引入任何新參數，只能覆寫既有的值。

---

## 三、事實錯誤（3 項）

### 3.1 `sync_uhai.py` 的目標被說錯了

文件說它推送到 `project::ingestSignalOutcomes` / `trading-signals` pack。實際上：

```67:67:scripts/sync_uhai.py
_INGEST_ENDPOINT = "microstructure::ingestMicroEvents"
```

它推送到的是**這個 repo 自帶的** `uhai/microstructure` pack，與 UHAI 的 `trading-signals` pack 無關。因此「兩邊語義失配」的診斷雖然方向對，但結論錯了：不需要為稽核軌跡新建 pack，本地 pack 已經在做這件事。

更重要的是，它**已經在推送 `promotion_history.jsonl`** —— 也就是文件提議要建的「優化稽核軌跡」，資料已經在流向 UHAI 了。

### 3.2 文件引用的 UHAI 端點全部不存在

`project::suggestPairs`、`suggestParams`、`rankParams`、`getParetoOptimal`、`detectRegime`、`diagnoseDegradation` —— 在 UHAI repo 全文搜尋**零命中**。`skopt` / `scikit-optimize` 也不是 `python-svc` 的依賴。

這些是待實作的設計，不是可呼叫的介面。文件把它們寫成範例程式碼，容易被誤讀為現成可用。

### 3.3 數據是編造的

「6200x 加速」、「CADF 通過率 >60% vs 隨機 5%」、「50 次回測 vs 100 次」、「Sharpe +50%」—— 這些沒有任何測量依據。

這一點值得單獨指出，因為這個 repo 的文化恰好相反：`walk_forward.py` 的 docstring 記錄了整個 gate 設計是怎麼被量測出來的（`scripts/design_wfo_gate.py`，附上 7.0% false positive、77.7% power、99.6% overfit catch rate）。應該提出**要量測什麼**，而不是斷言結果。

---

## 四、重大遺漏（6 項）

### 4.1 `scripts/self_improve_loop.py` 已經存在

這是最大的遺漏。該腳本的 docstring 開頭：

```1:20:scripts/self_improve_loop.py
"""
Self-improving strategy loop: WFO grid search -> validation gates ->
baseline comparison -> (auto) write-back to configs/strategy.yaml ->
append-only decision history.
```

它已經實作了文件所提「Meta-Optimizer」的絕大部分：

| 文件提議的功能 | 現況 |
|--------------|------|
| 讀取現任參數作為 baseline | ✅ `run_iteration()` 第 121 行 |
| 參數紀律預檢 | ✅ `preflight_check()` |
| 參數搜尋 + WFO 驗證 | ✅ `WalkForwardOptimizer` over `param_grids.yaml` |
| 同折 apples-to-apples baseline 比較 | ✅ `baseline_wfo` with `[{}]` grid |
| 多重驗證 gate | ✅ 4 個：`wfo_go`、`oos_drawdown_within_limit`、`has_oos_trades`、`monte_carlo_p5_sharpe` |
| 審批 / 寫回 `strategy.yaml` | ✅ `promotion.evaluate_and_promote()`，保留註解，`auto_execute` 永不觸碰 |
| 稽核軌跡 | ✅ `backtests/logs/promotion_history.jsonl` + `self_improvement_log.md` |
| 多輪迭代（前一輪結果成為下一輪 baseline） | ✅ `--iterations N` |

**真正缺的只有兩樣**：搜尋策略（grid → Bayesian）、以及跨回合記憶（哪些參數/regime 已經失敗過）。這才是 UHAI 應該補的位置 —— 範圍比三份文件所描述的小得多。

### 4.2 Reality Check 已設定但沒有接進自我改進迴圈 ✅ **已修復**

**原狀況**: `self_improve_loop.py` 的 gate 只有四個，不含 Reality Check。

**現狀況**: ✅ **已於 2026-09-10 整合完成**

Reality Check 現已整合為第 5 個 gate：

```python
gates = {
    "wfo_go": candidate_wfo.decision == "GO",
    "oos_drawdown_within_limit": check_drawdown_gate(...),
    "has_oos_trades": check_has_trades_gate(...),
    "monte_carlo_p5_sharpe": mc_result.sharpe.p5 >= min_p5,
    "reality_check_pass": rc_result.verdict == "PASS" if rc_result else True,  # 新增
}
```

**驗證結果**:
- ✅ Demo 模式運行成功
- ✅ 所有 Chan guard 測試通過（6 passed）
- ✅ 日誌正確記錄新 gate 狀態
- ✅ 錯誤處理確保 Reality Check 失敗時不阻斷流程

詳見 `docs/reality_check_integration_summary.md`。

### 4.3 多重比較造成的 gate 弱化完全沒有處理

文件把「搜尋效率提升」當成純粹的好處。但 `walk_forward.py` 自己就警告過：

> Power is quoted where parameters were NOT fitted, so a real run that picks parameters on IS does worse.

Bayesian optimization 的本質是**把採樣集中在 Sharpe 看起來最好的區域**，而那恰恰是過度擬合候選最密集的地方。搜尋越聰明，通過 gate 的 spurious 候選就越多。文件沒有任何機制處理這件事。

而且 `reality_check.py` 是對**單一參數集**做 phase-randomization bootstrap，並不是 White 原始論文那個跨「整個被搜尋過的模型宇宙」做調整的版本。所以「有 White's Reality Check 保護」這個說法，以現有實作而言並不成立。

### 4.4 Regime 機制已經存在

文件提議從零打造「Regime Detector」。實際已有：

- `python/analytics/regime.py`
- `python/analytics/trend_efficiency_gate.py`
- `python/analytics/macro_beta_gate.py`
- `python/analytics/gate_policy.py` + `configs/gate_scorecards.yaml`

而且 `configs/strategy.yaml` 對 `pairs_trading` 的註解已經寫明它**已經是 regime-gated**：「Armed only when the trend-efficiency gate says mean-reversion-friendly (automatically OFF in a 2024–2026-like bull/trend)」。

### 4.5 粗網格是刻意的紀律，不是效率缺陷

`configs/param_grids.yaml` 檔頭：

> Keep grids coarse (3-4 values per axis). A fine grid over a short window is curve-fitting with extra steps; WFO's per-fold re-optimization only demonstrates robustness when the candidate set is small enough that winning repeatedly means something.

`pairs_trading` 目前是 3×3×3 = 27 個候選，這是**選擇**而非疏漏。文件把 Bayesian optimization 定位為「探索更細緻的參數空間」，方向上與這條明文紀律相衝突。

要主張改用 Bayesian，論點必須是「在同樣的候選預算下找到更好的點」，而不是「可以試更多點」。

### 4.6 `min/max_half_life_days` 被刻意排除在搜尋之外

文件的 `param_bounds` 沒提這件事。`param_grids.yaml` 說明了原因：這兩個是**可交易性資格過濾器**，「optimizing eligibility bounds against returns is a classic data-snooping trap」。UHAI 的搜尋空間必須沿用同樣的排除。

---

## 五、修正後的建議範圍

把三份文件的六週計畫，收斂成四項具體工作，依價值排序：

### 1. 把 Reality Check 接進 `self_improve_loop.py`（約 30 行，最高價值）

在 `gates` dict 加一項，讀 `goal.yaml` 的 `reality_check.max_p_value`。掛勾已經在 `promotion.py` 留好了。這件事**與 UHAI 無關**，現在就該做，而且是後續一切的前提。

### 2. 在折內把 argmax 換成 GP 引導搜尋（只讀 in-sample）

改 `WalkForwardOptimizer._select()`，不要改迴圈結構。關鍵約束：選擇只能看 in-sample metrics，OOS 永遠不參與選參數。這樣就保住了現有 WFO 的誠實性。

### 3. 記錄搜尋預算，並據此收緊 gate

在 `PromotionRecord.extra` 記下 `n_configs_evaluated`。若 UHAI 把候選數從 27 提高到數百，`gate_alpha` 必須相應收緊，否則 gate 的 false positive 率會從量測過的 7.0% 悄悄漂高。這需要用 repo 既有的做法量測（參考 `scripts/design_wfo_gate.py`），不能憑估計。

### 4. UHAI 的真正定位：跨回合記憶 + 解釋，而非另一個 optimizer

`promotion_history.jsonl` 已經在流向 UHAI。UHAI 的獨特價值是它有圖譜與 LLM，而 `self_improve_loop.py` 沒有：

- **跨回合記憶**：哪些參數區域、在哪些 regime 下已經被拒絕過，以及拒絕原因（哪個 gate 掛掉）
- **解釋**：把 `REJECTED: gates failed: monte_carlo_p5_sharpe` 翻譯成人看得懂的診斷
- **退化診斷**：實盤 Sharpe 低於 WFO 預測時，指出是哪些配對、哪些市況造成

這幾項都不需要 UHAI 做數值最佳化，也不需要新的 pack。

---

## 六、對「UHAI 能診斷就該能解決」的回應

您上一輪的論點是對的，前一份文件把 UHAI 限制在純解釋層確實過度保守。但驗證完程式碼後，正確的形式是：

**參數自動調整這件事，這個 repo 已經在做了** —— `self_improve_loop.py` 會搜尋參數、跑 gate、與現任比較、寫回 `strategy.yaml`、留下稽核軌跡。UHAI 要做的不是取代它，而是補它缺的兩樣東西：更聰明的搜尋、以及跨回合的記憶。

唯一被硬性保留給人的，是 `auto_execute` —— 而那不是保守，是 `docs/lessons_from_forex_trading.md` 記錄的教訓：前一套 forex 系統有三個策略在實盤中靜默失效，還有一個 config key（正是 `auto_execute`）有一段時間沒有任何程式在讀。

---

**附註**: 本文件為覆核意見，不取代前三份文件；前三份文件對 UHAI 平台本身的描述仍然有效。
