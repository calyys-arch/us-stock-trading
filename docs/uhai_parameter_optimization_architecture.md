# UHAI Parameter Optimization Architecture
**目標**: 讓 UHAI 調整策略參數以提高盈利，同時保持 Chan Discipline 和風控

**日期**: 2026-09-10  
**狀態**: 設計探索 —— **已被實作取代**

> 實際出貨的版本見 [`uhai_parameter_suggester.md`](uhai_parameter_suggester.md)。
> 本文保留設計推演過程，但其中的程式碼片段是示意，與實作有實質差異；
> 特別注意**不要照抄下方的 sidecar 預設 port 8082** —— 那是本 repo dashboard
> 的 port（`scripts/start_dashboard.py`），實作改用 8092 並加了 `/health`
> 身分探測。

---

## 一、核心約束與挑戰

### 1.1 不可妥協的約束

1. **Chan Discipline**: ≤ 5 個自由參數 per strategy
2. **No look-ahead bias**: UHAI 優化過程**不得**使用未來數據
3. **`auto_execute` 雙鑰匙門控**: 
   - `ExecutionGateway.mode = "auto"` AND `strategy.auto_execute = true`
   - UHAI 提出的參數變更必須經**人工審批**才能啟用 `auto_execute`
4. **WFO 仍是最終裁判**: 任何參數變更必須通過 WFO GO 決策 (breadth + decay test)

### 1.2 關鍵挑戰

| 挑戰 | 風險 | 緩解策略 |
|------|------|---------|
| **過度擬合** | LLM 驅動的優化可能記住特定歷史模式 | 使用 **out-of-sample validation** + White's Reality Check |
| **Look-ahead bias** | UHAI 可能洩漏未來數據到回測 | 嚴格的時間切割 + 禁止訪問未來 OHLCV |
| **高維搜索** | 5 個參數的組合空間過大 | 貝葉斯優化 (Bayesian Optimization) 而非暴力網格搜索 |
| **Regime shift** | 參數在一個市場狀態下最優，換狀態後失效 | Regime-gated parameter sets + 自適應切換 |

---

## 二、架構方案: UHAI 作為 Meta-Optimizer

### 2.1 核心理念

UHAI **不直接執行交易**，而是作為 **WFO 的智能搜索引擎**：

```
傳統 WFO:
  param_grid = [
      {"entry_z": 2.0, "exit_z": 0.5},
      {"entry_z": 3.0, "exit_z": 0.5},
      {"entry_z": 4.0, "exit_z": 0.5},
      ...  # 手動窮舉 100+ 組合
  ]
  → 回測每組合 → 選 IS Sharpe 最高 → OOS 驗證

UHAI-增強 WFO:
  initial_params = {"entry_z": 2.0, "exit_z": 0.5}
  → UHAI Bayesian Optimizer 建議下 5 組候選
  → 回測 5 組 (而非 100 組) → 根據結果更新 Gaussian Process
  → 迭代 10 輪 → 找到最優參數 (50 次回測 vs 100 次)
  → 人工審批 → 應用到 strategy.yaml → 啟用 auto_execute
```

### 2.2 系統架構圖

```
┌──────────────────────────────────────────────────────────────────┐
│                     UHAI Control Loop                            │
│  (runs daily at market close, or on-demand via Streamlit)        │
└───────────┬──────────────────────────────────────────────────────┘
            │
            v
   ┌────────────────────┐
   │ 1. Regime Detector │  ← UHAI analyzes VIX, trend metrics, WFO decay
   └────────┬───────────┘
            │ If regime_shift detected → trigger re-optimization
            v
   ┌────────────────────┐
   │ 2. Bayesian Optim  │  ← UHAI (via python-svc sklearn.optimize.gp_minimize)
   │    Suggest Params  │     suggests next 5 param candidates
   └────────┬───────────┘
            │ param_candidates = [{entry_z: 3.2, exit_z: 0.6}, ...]
            v
   ┌────────────────────┐
   │ 3. WFO Validator   │  ← Existing walk_forward.py
   │    (Honest OOS)    │     backtest each candidate, return OOS metrics
   └────────┬───────────┘
            │ metrics = [{sharpe: 1.2, max_dd: -0.15}, ...]
            v
   ┌────────────────────┐
   │ 4. Knowledge Graph │  ← UHAI stores (params → metrics → verdict)
   │    Record          │     in GreyCat triple store
   └────────┬───────────┘
            │ trading:OptimizationRun → trading:ParamSet → trading:WFOResult
            v
   ┌────────────────────┐
   │ 5. Human Review    │  ← Streamlit dashboard shows:
   │    & Approval      │     - Proposed params vs current
   └────────┬───────────┘     - WFO verdict (GO/NO-GO)
            │ [Approve] button → writes to strategy.yaml
            │                 → sets auto_execute = true (if GO)
            v
   ┌────────────────────┐
   │ 6. Apply & Monitor │  ← ExecutionGateway reads new params
   │                    │     UHAI tracks live performance vs WFO prediction
   └────────────────────┘
```

---

## 三、具體實作層

### 3.1 UHAI `chan-quant` Pack 擴展

在原先的 `chan-quant` pack (解釋層) 基礎上，添加**優化層**：

```
chan-quant/
├── ontology/
│   └── base.ttl                       # 新增 OptimizationRun, ParamSet classes
├── connectors/
│   ├── wfo_result_jsonl.gcl           # 原有: 接收 WFO 結果
│   └── optimization_run_jsonl.gcl     # 新增: 接收優化過程追蹤
├── scoring/
│   ├── pair_scoring.gcl               # 原有: 協整對評分
│   └── param_scoring.gcl              # 新增: 參數組合評分 (Pareto front)
├── optimization/                      # 新模組
│   ├── bayesian_optimizer.gcl         # GCL 接口: 調用 python-svc Bayesian Optimizer
│   ├── regime_detector.gcl            # 檢測 regime shift (VIX spike, trend break)
│   └── approval_workflow.gcl          # 審批流程: pending → approved → applied
└── prompts/
    └── tasks/
        ├── explain_optimization.md     # "為什麼 UHAI 建議 entry_z=3.2？"
        └── diagnose_degradation.md     # "為什麼實盤 Sharpe 低於 WFO 預測？"
```

### 3.2 Python Sidecar 新增 Optimizer 端點

`universal-hybrid-ai/python-svc/main.py`:

```python
from skopt import gp_minimize
from skopt.space import Real

@app.post("/optimize/suggest_params")
def suggest_params(request: OptimizationRequest):
    """
    Bayesian Optimization: suggest next parameter candidates.
    
    Input:
      - strategy_id: "pairs_trading"
      - param_bounds: {"entry_z": [1.0, 5.0], "exit_z": [0.0, 1.0]}
      - history: [{params: {...}, sharpe: 1.2}, ...]  # previous evaluations
    
    Output:
      - next_candidates: [{entry_z: 3.2, exit_z: 0.6}, ...]  # top 5 suggestions
    """
    space = [Real(*request.param_bounds[k]) for k in sorted(request.param_bounds)]
    
    # Fit Gaussian Process on history
    X_history = [[h.params[k] for k in sorted(request.param_bounds)] 
                 for h in request.history]
    y_history = [-h.sharpe for h in request.history]  # minimize negative Sharpe
    
    result = gp_minimize(
        func=lambda x: 0.0,  # dummy (we only use the GP, not the minimizer loop)
        dimensions=space,
        x0=X_history, y0=y_history,
        n_calls=5,
        acq_func="EI",  # Expected Improvement
    )
    
    return {
        "next_candidates": [
            {k: float(v) for k, v in zip(sorted(request.param_bounds), x)}
            for x in result.x_iters[-5:]
        ]
    }
```

### 3.3 `us-stock-trading` 新模組

#### `scripts/uhai_optimize_strategy.py`

```python
"""
UHAI-driven parameter optimization for a strategy.

Usage:
  python scripts/uhai_optimize_strategy.py pairs_trading --n-iterations 10
  
Flow:
  1. Load current params from configs/strategy.yaml
  2. Define param bounds (entry_z: [1.0, 5.0], exit_z: [0.0, 1.0])
  3. For i in range(n_iterations):
       a. Ask UHAI for next 5 candidates (POST /optimize/suggest_params)
       b. Run WFO on each candidate (python/backtest/walk_forward.py)
       c. Push results to UHAI (POST /project::ingestOptimizationRun)
  4. UHAI ranks all tested params by Pareto front (Sharpe vs max_dd)
  5. Generate approval report in Streamlit
"""
import requests
from pathlib import Path
from python.backtest.walk_forward import WalkForwardOptimizer, WFOConfig

UHAI_URL = os.getenv("UHAI_GREYCAT_URL", "http://localhost:8080")
PYTHON_SVC_URL = os.getenv("UHAI_PYTHON_SVC_URL", "http://localhost:8082")

def optimize_strategy(strategy_id: str, n_iterations: int = 10):
    # 1. Load current params
    config = yaml.safe_load(Path("configs/strategy.yaml").read_text())
    current = config[strategy_id]
    
    # 2. Define bounds (Chan Discipline: only optimize the 5 free params)
    if strategy_id == "pairs_trading":
        bounds = {
            "entry_z": [1.5, 5.0],
            "exit_z": [0.0, 1.0],
            "half_life_multiplier_max_hold": [2.0, 4.0],
            # coint_lookback_days, revalidate_every_days frozen (structural)
        }
    
    history = []
    
    for iteration in range(n_iterations):
        # 3a. Ask UHAI for next candidates
        resp = requests.post(f"{PYTHON_SVC_URL}/optimize/suggest_params", json={
            "strategy_id": strategy_id,
            "param_bounds": bounds,
            "history": history,
        })
        candidates = resp.json()["next_candidates"]
        
        # 3b. Run WFO on each candidate
        for candidate in candidates:
            wfo = WalkForwardOptimizer(
                backtest_fn=lambda s, e, p: run_pairs_backtest(s, e, {**current, **p}),
                config=WFOConfig(),
                param_grid=[candidate],
            )
            result = wfo.run(start_date, end_date)
            
            # 3c. Push to UHAI
            history.append({
                "params": candidate,
                "sharpe": result.oos_sharpe_mean,
                "max_dd": result.folds[0].oos_metrics.get("max_drawdown", 0.0),
                "decision": result.decision,
            })
            requests.post(f"{UHAI_URL}/project::ingestOptimizationRun", json=[
                "demo", "chan-quant", {
                    "run_id": f"{strategy_id}_opt_{datetime.now().isoformat()}",
                    "params": candidate,
                    "wfo_result": result.to_dict(),
                }
            ])
    
    # 4. UHAI ranks params (Pareto front)
    resp = requests.post(f"{UHAI_URL}/project::rankParams", json=[
        "demo", "chan-quant", strategy_id
    ])
    ranked = resp.json()["pareto_front"]  # [{params, sharpe, max_dd, rank}]
    
    # 5. Generate approval report
    report = {
        "strategy_id": strategy_id,
        "current_params": current,
        "recommended_params": ranked[0]["params"],  # top Pareto candidate
        "sharpe_improvement": ranked[0]["sharpe"] - current_sharpe,
        "wfo_verdict": ranked[0]["wfo_verdict"],
        "approval_status": "pending",
    }
    Path("data/optimization_reports/pending_approval.json").write_text(
        json.dumps(report, indent=2)
    )
    
    print(f"\n✅ Optimization complete. Review at: streamlit run dashboard/optimization_review.py")
```

#### Streamlit 審批界面

`dashboard/optimization_review.py`:

```python
import streamlit as st
import json
from pathlib import Path

report = json.loads(Path("data/optimization_reports/pending_approval.json").read_text())

st.title("UHAI Parameter Optimization Review")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Parameters")
    st.json(report["current_params"])
    st.metric("Current Sharpe", report["current_sharpe"])

with col2:
    st.subheader("UHAI Recommended")
    st.json(report["recommended_params"])
    st.metric("Projected Sharpe", report["recommended_sharpe"], 
              delta=f"+{report['sharpe_improvement']:.2f}")

st.subheader("WFO Verdict")
st.info(f"Decision: {report['wfo_verdict']}")

# Show fold-by-fold breakdown
st.dataframe(report["wfo_folds"])

# Approval buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("✅ Approve & Apply", type="primary"):
        # Write to strategy.yaml
        config = yaml.safe_load(Path("configs/strategy.yaml").read_text())
        config[report["strategy_id"]].update(report["recommended_params"])
        config[report["strategy_id"]]["auto_execute"] = True  # enable if WFO = GO
        Path("configs/strategy.yaml").write_text(yaml.dump(config))
        
        # Log to audit
        requests.post(f"{UHAI_URL}/project::recordApproval", json=[
            "demo", "chan-quant", {
                "run_id": report["run_id"],
                "approver": "human_operator_1",
                "approved_at": datetime.now().isoformat(),
                "applied_params": report["recommended_params"],
            }
        ])
        st.success("Parameters applied! Restart ExecutionGateway to take effect.")

with col2:
    if st.button("⏸️ Approve for Paper Trading Only"):
        config[report["strategy_id"]]["auto_execute"] = False  # paper mode
        st.warning("Parameters applied in paper mode. Review live results before enabling auto_execute.")

with col3:
    if st.button("❌ Reject"):
        report["approval_status"] = "rejected"
        st.error("Optimization rejected. Current parameters retained.")
```

---

## 四、Regime-Adaptive Parameter Switching

### 4.1 問題陳述

**單一參數集無法適應所有市場狀態**：
- 牛市 (2024-2026): `entry_z=4.0` (低頻高確信) 最優
- 震盪市 (2022): `entry_z=2.0` (高頻低確信) 最優

### 4.2 解決方案: Regime-Gated Parameter Sets

```yaml
# configs/strategy.yaml
pairs_trading:
  enabled: true
  regime_adaptive: true  # 新增: 啟用 regime-based 參數切換
  
  # Default params (用於 regime 不確定時)
  entry_z: 3.0
  exit_z: 0.5
  
  # Regime-specific overrides
  regime_params:
    mean_reversion_friendly:  # VIX > 20, Hurst < 0.5
      entry_z: 2.0
      exit_z: 0.3
    trending:  # VIX < 15, Hurst > 0.55
      entry_z: 4.0
      exit_z: 0.7
      auto_execute: false  # 趨勢市關閉 pairs trading
```

**UHAI 的角色**:
1. **每日 regime 分類**: 
   ```python
   regime = uhai.classify_regime(vix=18.5, hurst=0.52, trend_strength=0.3)
   # → "mean_reversion_friendly"
   ```

2. **自動切換參數**:
   ```python
   if regime != current_regime:
       params = config["pairs_trading"]["regime_params"][regime]
       log_audit(f"Regime shift detected: {current_regime} → {regime}")
       apply_params(params, require_approval=True)  # 仍需人工審批
   ```

3. **Regime 預測** (可選, Phase 2):
   - UHAI 通過 LLM 分析宏觀新聞 (Fed 會議、財報季)
   - 預測未來 21 天的 regime 概率分佈
   - 提前建議參數調整 (proactive vs reactive)

---

## 五、風險管控與審計

### 5.1 強制審批流程

| 參數變更類型 | 審批要求 | 理由 |
|------------|---------|------|
| **WFO GO + Sharpe 提升 > 0.5** | 單人審批 | 明確改進 |
| **WFO GO + Sharpe 提升 < 0.5** | 雙人審批 | 邊緣改進，需謹慎 |
| **WFO NO-GO 但 UHAI 堅持** | 禁止 | UHAI 不得違背 WFO 裁判 |
| **Regime 切換** | 自動 (但記錄審計) | 預先批准的 regime_params |

### 5.2 審計日誌 (存入 UHAI Knowledge Graph)

每次優化運行記錄：

```turtle
@prefix opt: <https://uhai.platform/packs/chan-quant/optimization#> .

opt:run_2026_09_10_001 a opt:OptimizationRun ;
    opt:strategy "pairs_trading" ;
    opt:triggeredBy "regime_shift_detected" ;
    opt:startedAt "2026-09-10T09:00:00Z" ;
    opt:nIterations 10 ;
    opt:nCandidatesTested 50 ;
    opt:bestParams [
        opt:entry_z 3.2 ;
        opt:exit_z 0.6 ;
        opt:wfo_verdict "GO" ;
        opt:projected_sharpe 1.8 ;
    ] ;
    opt:approvedBy "human_operator_1" ;
    opt:approvedAt "2026-09-10T16:30:00Z" ;
    opt:appliedAt "2026-09-10T16:35:00Z" .
```

### 5.3 實時性能監控

UHAI 每日檢查：

```python
def monitor_live_vs_predicted():
    # 讀取最近應用的參數
    last_approval = uhai.query("""
        SELECT ?run ?predicted_sharpe WHERE {
            ?run a opt:OptimizationRun ;
                 opt:appliedAt ?applied ;
                 opt:projected_sharpe ?predicted_sharpe .
        } ORDER BY DESC(?applied) LIMIT 1
    """)
    
    # 讀取實盤表現
    live_sharpe = get_live_sharpe_last_21_days()
    
    # 比較
    degradation = (predicted_sharpe - live_sharpe) / predicted_sharpe
    
    if degradation > 0.3:  # 實盤比預測差 30%+
        alert = f"⚠️ Live Sharpe ({live_sharpe:.2f}) significantly below " \
                f"WFO prediction ({predicted_sharpe:.2f}). Possible regime shift."
        send_slack_alert(alert)
        trigger_reoptimization()
```

---

## 六、Phase 實作計劃

### Phase 1: 基礎設施 (1 週)

- [ ] `chan-quant` pack 添加 `optimization/` 模組
- [ ] Python sidecar 添加 `/optimize/suggest_params` 端點
- [ ] `scripts/uhai_optimize_strategy.py` 實作
- [ ] Streamlit 審批界面 `dashboard/optimization_review.py`

### Phase 2: Bayesian Optimizer 整合 (3 天)

- [ ] 測試 `scikit-optimize` 在 5 維參數空間的效率
- [ ] 實作 Pareto front 排序 (Sharpe vs max_dd)
- [ ] 添加約束優化 (entry_z > exit_z, half_life > 1 天)

### Phase 3: Regime Detector (1 週)

- [ ] 實作 VIX + Hurst + Trend 指標計算
- [ ] 訓練 regime 分類器 (可選: LightGBM on historical data)
- [ ] `configs/strategy.yaml` 添加 `regime_params` 支持
- [ ] ExecutionGateway 自動切換參數 (需審批)

### Phase 4: 審計與監控 (3 天)

- [ ] UHAI Knowledge Graph 存儲優化歷史
- [ ] 實時性能監控 (live Sharpe vs WFO prediction)
- [ ] Streamlit dashboard 新增 "Optimization History" 頁面

### Phase 5: Production Hardening (1 週)

- [ ] 添加 Unit tests (`tests/test_uhai_optimizer.py`)
- [ ] Look-ahead bias 審計 (確保 UHAI 無法訪問未來數據)
- [ ] 回測整個 2022-2026 優化循環 (模擬每季度重新優化)
- [ ] 編寫 Runbook (`docs/uhai_optimization_runbook.md`)

---

## 七、關鍵設計決策

### 7.1 為何選 Bayesian Optimization？

| 方法 | 優點 | 缺點 | 適用場景 |
|------|------|------|---------|
| **Grid Search** | 簡單、可重現 | 指數爆炸 (5 個參數 × 10 值 = 10^5 組合) | 參數 ≤ 3 個 |
| **Random Search** | 高效、可並行 | 無記憶、重複測試 | Quick & Dirty |
| **Bayesian Opt** | 記憶過去結果、高效收斂 | 需 Gaussian Process 計算 | **Chan Discipline (≤ 5 參數)** ✅ |
| **Reinforcement Learning** | 可適應 non-stationary | 需大量樣本、難解釋 | 實時交易 (非回測) |

**選擇 Bayesian Optimization** 因為：
- 5 維參數空間適合 GP modeling
- Expected Improvement (EI) acquisition 能平衡 exploration vs exploitation
- Scikit-optimize 成熟、易整合

### 7.2 為何不直接用 LLM 選參數？

**問題**: LLM (GPT-5.5) 可以直接建議 `entry_z=3.2` 嗎？

**答案**: ❌ **不行，原因如下**：

1. **LLM 無法執行數值優化**: GPT 擅長語言推理，不擅長搜索連續參數空間
2. **缺乏回測反饋循環**: LLM 無法直接運行 WFO 並根據 Sharpe 更新搜索方向
3. **幻覺風險**: LLM 可能生成"看起來合理"但實際無效的參數

**LLM 的正確角色** (保留):
- **解釋優化結果**: "UHAI 建議 entry_z=3.2 因為這在最近 8 個 WFO fold 中平均 Sharpe 達 1.8，且 max_dd 控制在 -0.15"
- **Regime 分析**: 分析新聞、宏觀數據，預測 regime shift
- **異常診斷**: "實盤 Sharpe 低於預測可能因為: (1) VIX 突然飆升 (2) 主要配對 AAPL-GOOG 失去協整"

**數值優化交給 Bayesian Optimizer**，**語義推理交給 LLM** — 各司其職。

---

## 八、與原方案對比

| 維度 | 原方案 (UHAI = 解釋層) | 新方案 (UHAI = Meta-Optimizer) |
|------|---------------------|----------------------------|
| **UHAI 角色** | Post-mortem 分析師 | 主動優化引擎 + 分析師 |
| **能否調參** | ❌ 僅解釋現有參數 | ✅ 通過 Bayesian Opt 建議新參數 |
| **能否執行** | ❌ 不參與交易決策 | ❌ 仍不執行 (需人工審批) |
| **WFO 地位** | 外部流程 | **仍是最終裁判** (UHAI 只能加速搜索) |
| **風險** | 低 (純讀取) | 中 (可能過擬合，需嚴格審計) |
| **開發工作量** | 2-3 天 | **2-3 週** (含 Bayesian Opt + Streamlit) |

---

## 九、成功標準

### 9.1 技術指標

- [ ] **參數搜索效率**: Bayesian Opt 在 50 次回測內找到最優解 (vs Grid Search 100+ 次)
- [ ] **WFO 通過率**: UHAI 建議的參數在 WFO 中 GO 比例 > 70%
- [ ] **Sharpe 提升**: 相比手動調參，平均 OOS Sharpe 提升 > 0.3
- [ ] **Regime 切換準確率**: Regime detector 在回測中準確率 > 75%

### 9.2 風控指標

- [ ] **無 look-ahead bias**: `tests/test_lookahead_bias.py` 通過
- [ ] **審批覆蓋率**: 100% 的參數變更有審計日誌
- [ ] **實盤一致性**: Live Sharpe 與 WFO 預測誤差 < 30%

### 9.3 業務指標

- [ ] **盈利能力**: 應用 UHAI 優化後的策略在 paper trading 中連續 3 個月盈利
- [ ] **Sharpe ratio**: Paper trading Sharpe > 1.5 (vs 當前 pairs_trading ~1.0)
- [ ] **Max drawdown**: 控制在 -20% 以內 (vs 當前 regime-gated limit -25%)

---

## 十、總結

### 關鍵理念

**UHAI 不取代 Chan Discipline，而是增強它**：
- ✅ **保留 ≤ 5 參數約束**: UHAI 優化的仍是這 5 個參數
- ✅ **保留 WFO 裁判**: UHAI 只能建議參數，必須通過 WFO GO
- ✅ **保留人工審批**: 啟用 `auto_execute` 仍需人類決策
- ✅ **增強搜索效率**: Bayesian Opt 替代暴力網格搜索
- ✅ **增強適應性**: Regime detector 自動切換參數集

### 風險與緩解

| 風險 | 緩解 |
|------|------|
| 過擬合 | Out-of-sample WFO + White's Reality Check |
| Look-ahead bias | 嚴格時間切割 + 禁止訪問未來 OHLCV |
| 參數漂移 | 實時監控 (live vs predicted Sharpe) |
| 審批疲勞 | 僅在 Sharpe 提升 > 0.5 時觸發審批 |

### 下一步

**立即行動** (如採納此方案):
1. **Review 此架構文檔** → 確認技術可行性與業務目標一致
2. **Phase 1 Prototype** (1 週): 實作 `scripts/uhai_optimize_strategy.py` + Streamlit 審批界面
3. **回測驗證** (3 天): 在 2022-2026 數據上模擬優化循環
4. **Paper Trading** (1 個月): 在 paper account 運行 UHAI 優化的參數
5. **Live 部署** (僅在 paper Sharpe > 1.5 後): 啟用 `auto_execute`

---

**附錄**: 此架構與 `docs/uhai_integration_analysis.md` (解釋層方案) 互補，不衝突。
UHAI 同時扮演兩個角色：
1. **事後分析師** (解釋為何拒絕)
2. **主動優化器** (建議更好的參數)
