# UHAI × Chan 統計套利：統一架構
**核心理念**: UHAI **增強** Chan 方法論，而非取代或分離

**日期**: 2026-09-10  
**狀態**: 修正版架構

---

## 一、核心誤解的澄清

### 1.1 之前的錯誤假設

**我之前認為**:
- ❌ UHAI (LLM + Knowledge Graph) 與 Chan 統計套利在"哲學上不兼容"
- ❌ UHAI 應該只做"解釋層"（事後診斷），不參與參數優化
- ❌ UHAI 的 AI 模型會違反 Chan Discipline (≤ 5 參數)

### 1.2 正確的理解

**用戶的洞察** ✅:
> "既然 UHAI 能找出為什麼（診斷問題），那麼 UHAI 就應該有解決方案"

**關鍵認知**:
1. **UHAI 不引入新的交易信號** → 不違反 Chan 方法論
2. **UHAI 在 meta-level 優化現有的 5 個參數** → 不違反 ≤ 5 參數約束
3. **UHAI 使用 Knowledge Graph 記錄歷史** → 避免重複失敗，這是 **Chan 回測精神的延伸**
4. **UHAI 的 LLM 分析宏觀事件** → 預測 regime shift，這是 **量化交易的自然演進**

---

## 二、統一架構：UHAI 作為 Chan 框架的智能層

### 2.1 架構對比

#### ❌ 錯誤架構 (分離式)

```
Chan 統計套利 (獨立)              UHAI (獨立)
    ↓                              ↓
協整檢測 → WFO → 交易         解釋為何失敗
    ↑                              ↓
    └──────────────────────────────┘
         (弱連接，僅事後診斷)
```

**問題**: UHAI 只能"馬後砲"，無法主動改進。

#### ✅ 正確架構 (統一式)

```
┌─────────────────────────────────────────────────────────────┐
│              UHAI-Enhanced Chan Framework                   │
│  (UHAI 是 Chan 方法論的智能增強層，非外部附加)               │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        v                   v                   v
   [協整檢測]          [參數優化]           [執行決策]
   (Chan 原理)       (UHAI 增強 WFO)      (Chan 風控)
        │                   │                   │
        │   ┌───────────────┘                   │
        │   │   UHAI Knowledge Graph:           │
        │   │   記錄 (pair, params, regime)     │
        │   │        → (sharpe, max_dd)         │
        │   │                                    │
        v   v                                    v
    [信號生成] ────────> [WFO 驗證] ────────> [執行]
    (仍是 5 參數)      (仍是最終裁判)      (仍需審批)
```

**關鍵**: UHAI **嵌入**到 Chan 框架的每個環節，而非並行運作。

---

## 三、UHAI 增強 Chan 方法論的四個層次

### Layer 1: 協整對發現 (Pair Discovery)

#### Chan 原始方法
```python
# 手動挑選配對
pairs = [("AAPL", "GOOG"), ("JPM", "BAC"), ...]

# 逐對檢測協整
for pair in pairs:
    p_value = cadf_test(pair)
    if p_value < 0.05:
        add_to_universe(pair)
```

#### UHAI 增強
```python
# UHAI 從 Knowledge Graph 學習歷史成功配對的特徵
uhai_features = uhai.query("""
    SELECT ?pair ?sector ?correlation ?half_life ?success_rate
    WHERE {
        ?pair a coint:Pair ;
              coint:sector ?sector ;
              coint:avgCorrelation ?correlation ;
              coint:avgHalfLife ?half_life ;
              coint:wfoSuccessRate ?success_rate .
        FILTER (?success_rate > 0.7)
    } ORDER BY DESC(?success_rate)
""")

# 根據歷史模式，優先測試高概率配對
suggested_pairs = uhai.suggest_pairs(
    current_universe=sp500_stocks,
    historical_features=uhai_features,
    regime=current_regime,  # 牛市 vs 震盪市
)

# 仍然用 CADF 驗證 (Chan 原理不變)
for pair in suggested_pairs:
    p_value = cadf_test(pair)
    if p_value < 0.05:
        add_to_universe(pair)
```

**價值**: 
- ✅ 從歷史 1000+ 配對測試中學習 → 優先測試成功率高的組合
- ✅ 節省計算資源 (不需測試所有 C(500,2) = 124,750 種組合)
- ✅ **仍保持 Chan 驗證** (CADF p-value < 0.05)

---

### Layer 2: 參數優化 (Parameter Tuning)

#### Chan 原始方法
```python
# 手動網格搜索
param_grid = [
    {"entry_z": 2.0, "exit_z": 0.5},
    {"entry_z": 2.5, "exit_z": 0.5},
    {"entry_z": 3.0, "exit_z": 0.5},
    # ... 100+ combinations
]

# WFO 測試每組參數
for params in param_grid:
    wfo_result = run_wfo(params)
    if wfo_result.decision == "GO":
        best_params = params
```

#### UHAI 增強
```python
# UHAI Bayesian Optimizer 智能搜索
optimizer = UHAIBayesianOptimizer(
    param_bounds={
        "entry_z": [1.5, 5.0],
        "exit_z": [0.0, 1.0],
        "half_life_mult": [2.0, 4.0],
    },
    knowledge_graph=uhai.graph,  # 記錄所有歷史測試
)

# 迭代搜索 (50 次 vs 手動 100+ 次)
for iteration in range(10):
    # UHAI 根據歷史結果建議下 5 組候選
    candidates = optimizer.suggest_next_batch(n=5)
    
    # 仍用 WFO 驗證 (Chan 原理不變)
    for candidate in candidates:
        wfo_result = run_wfo(candidate)
        optimizer.update(candidate, wfo_result.oos_sharpe)
        
        # 記錄到 Knowledge Graph
        uhai.record_optimization_run(
            params=candidate,
            wfo_result=wfo_result,
            regime=current_regime,
        )

# UHAI 從 Pareto front 推薦最優參數
best_params = optimizer.get_pareto_optimal()
```

**價值**:
- ✅ **50 次回測找到最優解** (vs 手動 100+ 次)
- ✅ **記憶過去失敗** → 不重複測試無效區域
- ✅ **仍保持 WFO 驗證** (honest out-of-sample)

---

### Layer 3: Regime 適應 (Regime Adaptation)

#### Chan 原始方法
```python
# 單一參數集用於所有市場狀態
params = {"entry_z": 2.0, "exit_z": 0.5}

# 問題: 牛市與震盪市的最優參數不同
# 2024-2026 牛市: entry_z=4.0 最優 (低頻高確信)
# 2022 震盪市: entry_z=2.0 最優 (高頻低確信)
```

#### UHAI 增強
```python
# UHAI 自動檢測 regime 並切換參數集
regime = uhai.detect_regime(
    vix=current_vix,
    hurst_exponent=compute_hurst(spy_returns),
    trend_strength=compute_trend(spy_prices),
    macro_events=uhai.analyze_news(days=7),  # LLM 分析 Fed 會議、財報季
)

# 從 Knowledge Graph 載入該 regime 的歷史最優參數
optimal_params = uhai.query(f"""
    SELECT ?params ?avg_sharpe WHERE {{
        ?run a coint:OptimizationRun ;
             coint:regime "{regime}" ;
             coint:wfoVerdict "GO" ;
             coint:params ?params ;
             coint:oosSharpe ?sharpe .
    }}
    GROUP BY ?params
    HAVING AVG(?sharpe) AS ?avg_sharpe
    ORDER BY DESC(?avg_sharpe)
    LIMIT 1
""")

# 應用參數 (仍需人工審批)
if optimal_params != current_params:
    trigger_approval_workflow(
        reason=f"Regime shift detected: {old_regime} → {regime}",
        recommended_params=optimal_params,
        historical_sharpe=optimal_params.avg_sharpe,
    )
```

**價值**:
- ✅ **自動適應市場狀態** → 不需每次手動重新優化
- ✅ **LLM 分析宏觀事件** → 預測 regime shift（如 Fed 加息 → 震盪市）
- ✅ **仍保持人工審批** → 參數切換需確認

---

### Layer 4: 執行監控 (Execution Monitoring)

#### Chan 原始方法
```python
# 回測 Sharpe = 1.8
# 實盤 Sharpe = 1.2 (偏差 -33%)
# 問題: 不知道為何偏差？何時該重新優化？
```

#### UHAI 增強
```python
# UHAI 每日監控實盤 vs 回測
monitor_report = uhai.compare_live_vs_backtest(
    live_sharpe=get_live_sharpe(days=21),
    wfo_predicted_sharpe=1.8,
    live_trades=get_recent_trades(),
)

if monitor_report.degradation > 0.30:  # 偏差 > 30%
    # UHAI 診斷根本原因
    diagnosis = uhai.diagnose_degradation(
        live_trades=monitor_report.live_trades,
        failed_pairs=monitor_report.losing_pairs,
        market_conditions=get_current_vix_hurst(),
    )
    
    # LLM 生成解釋報告
    explanation = uhai.llm_explain(f"""
    為何實盤 Sharpe ({monitor_report.live_sharpe}) 低於 WFO 預測 ({1.8})？
    
    證據:
    - 失敗配對: {diagnosis.failed_pairs}
    - VIX: {diagnosis.vix_current} (WFO 時: {diagnosis.vix_backtest})
    - 主要損失: {diagnosis.top_losses}
    
    請分析根本原因並建議行動。
    """)
    
    # 發送警報 + 建議
    send_alert(f"""
    ⚠️ Performance Degradation Detected
    
    {explanation}
    
    建議行動:
    1. {diagnosis.suggested_actions[0]}
    2. {diagnosis.suggested_actions[1]}
    """)
```

**價值**:
- ✅ **自動偵測性能退化** → 不需人工每日檢查
- ✅ **LLM 診斷根本原因** → "AAPL-GOOG 失去協整因 Apple 財報意外暴跌"
- ✅ **可操作的建議** → "移除 AAPL-GOOG，重新測試 MSFT-GOOG"

---

## 四、UHAI 與 Chan Discipline 的完全兼容性

### 4.1 Chan Discipline 檢查表

| Chan 原則 | UHAI 是否遵守？ | 說明 |
|-----------|----------------|------|
| **≤ 5 自由參數** | ✅ **完全遵守** | UHAI 優化的仍是 entry_z, exit_z, half_life_mult 等 5 個參數 |
| **No look-ahead bias** | ✅ **完全遵守** | UHAI 的訓練數據僅來自已完成的 WFO 折 (out-of-sample) |
| **WFO 驗證** | ✅ **完全遵守** | UHAI 建議的參數必須通過 WFO GO 決策 |
| **Transaction costs** | ✅ **完全遵守** | UHAI 不改變回測引擎的成本模型 |
| **Survivorship bias** | ✅ **完全遵守** | UHAI 使用 point-in-time universe (configs/universe.yaml) |

### 4.2 UHAI 的增強點 (不違反 Chan)

| 增強功能 | 技術 | 為何不違反 Chan？ |
|---------|------|------------------|
| **智能參數搜索** | Bayesian Optimization | 只是更高效的搜索方法，仍在 5 維空間內，仍用 WFO 驗證 |
| **Regime 適應** | VIX/Hurst + LLM 新聞分析 | 切換的仍是預先回測過的參數集，非即時生成 |
| **配對建議** | Knowledge Graph 歷史學習 | 建議後仍用 CADF p-value 驗證，非直接交易 |
| **性能監控** | 實時 Sharpe 對比 + LLM 診斷 | 僅監控與診斷，不自動修改參數 |

**結論**: UHAI 是 **Chan 方法論的智能化工具**，不是替代方案。

---

## 五、具體實作：統一的 `chan-quant` Pack

### 5.1 Pack 結構 (整合式)

```
universal-hybrid-ai/industry-packs/chan-quant/
├── pack.yaml
├── ontology/
│   └── base.ttl                    # 協整對、WFO 結果、Regime 的本體論
├── connectors/
│   ├── pair_discovery.gcl          # 接收 CADF 測試結果
│   ├── wfo_results.gcl             # 接收 WFO 折的詳細數據
│   └── live_trades.gcl             # 接收實盤執行記錄
├── optimization/
│   ├── bayesian_optimizer.gcl      # 參數優化引擎
│   ├── regime_detector.gcl         # Regime 檢測與切換
│   └── pair_suggester.gcl          # 配對建議引擎
├── monitoring/
│   ├── degradation_detector.gcl    # 性能退化偵測
│   └── live_vs_backtest.gcl        # 實盤 vs 回測對比
├── scoring/
│   ├── pair_scoring.gcl            # 配對評分 (half-life 穩定性)
│   └── param_scoring.gcl           # 參數評分 (Pareto front)
└── prompts/
    └── tasks/
        ├── explain_why_failed.md        # "為何 AAPL-GOOG 失去協整？"
        ├── suggest_new_pairs.md         # "建議替代 AAPL-GOOG 的配對"
        ├── optimize_params.md           # "在當前 regime 下優化參數"
        └── diagnose_degradation.md      # "為何實盤 Sharpe 低於預測？"
```

### 5.2 本體論 (完整版)

`ontology/base.ttl`:

```turtle
@prefix coint: <https://uhai.platform/packs/chan-quant#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# ── 協整對 ────────────────────────────────────────────────────────
coint:Pair a owl:Class .
coint:symbol1 a owl:DatatypeProperty ; rdfs:domain coint:Pair ; rdfs:range xsd:string .
coint:symbol2 a owl:DatatypeProperty ; rdfs:domain coint:Pair ; rdfs:range xsd:string .
coint:halfLife a owl:DatatypeProperty ; rdfs:domain coint:Pair ; rdfs:range xsd:decimal .
coint:hurst a owl:DatatypeProperty ; rdfs:domain coint:Pair ; rdfs:range xsd:decimal .
coint:cadfPValue a owl:DatatypeProperty ; rdfs:domain coint:Pair ; rdfs:range xsd:decimal .
coint:hedgeRatio a owl:DatatypeProperty ; rdfs:domain coint:Pair ; rdfs:range xsd:decimal .
coint:sector a owl:DatatypeProperty ; rdfs:domain coint:Pair ; rdfs:range xsd:string .

# ── WFO 結果 ──────────────────────────────────────────────────────
coint:WFOResult a owl:Class .
coint:pair a owl:ObjectProperty ; rdfs:domain coint:WFOResult ; rdfs:range coint:Pair .
coint:params a owl:ObjectProperty ; rdfs:domain coint:WFOResult ; rdfs:range coint:ParamSet .
coint:regime a owl:DatatypeProperty ; rdfs:domain coint:WFOResult ; rdfs:range xsd:string .
coint:oosSharpe a owl:DatatypeProperty ; rdfs:domain coint:WFOResult ; rdfs:range xsd:decimal .
coint:maxDrawdown a owl:DatatypeProperty ; rdfs:domain coint:WFOResult ; rdfs:range xsd:decimal .
coint:wfoVerdict a owl:DatatypeProperty ; rdfs:domain coint:WFOResult ; rdfs:range xsd:string .
coint:nFolds a owl:DatatypeProperty ; rdfs:domain coint:WFOResult ; rdfs:range xsd:integer .
coint:passRatio a owl:DatatypeProperty ; rdfs:domain coint:WFOResult ; rdfs:range xsd:decimal .

# ── 參數集 ────────────────────────────────────────────────────────
coint:ParamSet a owl:Class .
coint:entryZ a owl:DatatypeProperty ; rdfs:domain coint:ParamSet ; rdfs:range xsd:decimal .
coint:exitZ a owl:DatatypeProperty ; rdfs:domain coint:ParamSet ; rdfs:range xsd:decimal .
coint:halfLifeMult a owl:DatatypeProperty ; rdfs:domain coint:ParamSet ; rdfs:range xsd:decimal .
coint:cointLookback a owl:DatatypeProperty ; rdfs:domain coint:ParamSet ; rdfs:range xsd:integer .
coint:revalidateDays a owl:DatatypeProperty ; rdfs:domain coint:ParamSet ; rdfs:range xsd:integer .

# ── Regime ────────────────────────────────────────────────────────
coint:Regime a owl:Class .
coint:vix a owl:DatatypeProperty ; rdfs:domain coint:Regime ; rdfs:range xsd:decimal .
coint:hurstExponent a owl:DatatypeProperty ; rdfs:domain coint:Regime ; rdfs:range xsd:decimal .
coint:trendStrength a owl:DatatypeProperty ; rdfs:domain coint:Regime ; rdfs:range xsd:decimal .
coint:classification a owl:DatatypeProperty ; rdfs:domain coint:Regime ; rdfs:range xsd:string .
  # "mean_reversion_friendly" | "trending" | "volatile"

# ── 實盤交易 ──────────────────────────────────────────────────────
coint:LiveTrade a owl:Class .
coint:tradedPair a owl:ObjectProperty ; rdfs:domain coint:LiveTrade ; rdfs:range coint:Pair .
coint:entryDate a owl:DatatypeProperty ; rdfs:domain coint:LiveTrade ; rdfs:range xsd:date .
coint:exitDate a owl:DatatypeProperty ; rdfs:domain coint:LiveTrade ; rdfs:range xsd:date .
coint:realizedPnL a owl:DatatypeProperty ; rdfs:domain coint:LiveTrade ; rdfs:range xsd:decimal .
coint:usedParams a owl:ObjectProperty ; rdfs:domain coint:LiveTrade ; rdfs:range coint:ParamSet .

# ── 優化歷史 ──────────────────────────────────────────────────────
coint:OptimizationRun a owl:Class .
coint:triggeredBy a owl:DatatypeProperty ; rdfs:domain coint:OptimizationRun ; rdfs:range xsd:string .
  # "manual" | "regime_shift" | "performance_degradation"
coint:testedParams a owl:ObjectProperty ; rdfs:domain coint:OptimizationRun ; rdfs:range coint:ParamSet .
coint:bestParams a owl:ObjectProperty ; rdfs:domain coint:OptimizationRun ; rdfs:range coint:ParamSet .
coint:approvedBy a owl:DatatypeProperty ; rdfs:domain coint:OptimizationRun ; rdfs:range xsd:string .
coint:appliedAt a owl:DatatypeProperty ; rdfs:domain coint:OptimizationRun ; rdfs:range xsd:dateTime .
```

### 5.3 Python 整合腳本

`us-stock-trading/scripts/uhai_unified_workflow.py`:

```python
"""
UHAI-Enhanced Chan Workflow
統一的參數優化、配對發現、實盤監控流程
"""
import requests
from datetime import datetime, timedelta
from pathlib import Path
import yaml

UHAI_URL = "http://localhost:8080"
UHAI_TENANT = "demo"
UHAI_PACK = "chan-quant"

class UHAIChanWorkflow:
    
    # ── Layer 1: 配對發現 ─────────────────────────────────────────
    def discover_pairs(self, universe: list[str]) -> list[tuple]:
        """UHAI 建議高概率協整配對"""
        resp = requests.post(f"{UHAI_URL}/project::suggestPairs", json=[
            UHAI_TENANT, UHAI_PACK, {
                "universe": universe,
                "current_regime": self.detect_regime(),
                "top_k": 20,
            }
        ])
        suggested = resp.json()["suggested_pairs"]
        
        # 仍用 CADF 驗證 (Chan 原理)
        verified_pairs = []
        for pair in suggested:
            p_value = self.cadf_test(pair["symbol1"], pair["symbol2"])
            if p_value < 0.05:
                verified_pairs.append((pair["symbol1"], pair["symbol2"]))
                
                # 記錄到 UHAI
                requests.post(f"{UHAI_URL}/project::recordPairDiscovery", json=[
                    UHAI_TENANT, UHAI_PACK, {
                        "pair": pair,
                        "cadf_p_value": p_value,
                        "half_life": self.compute_half_life(pair),
                        "discovered_at": datetime.now().isoformat(),
                    }
                ])
        
        return verified_pairs
    
    # ── Layer 2: 參數優化 ─────────────────────────────────────────
    def optimize_params(self, pair: tuple, n_iterations: int = 10):
        """UHAI Bayesian Optimization"""
        optimizer_state = {"iteration": 0, "history": []}
        
        for i in range(n_iterations):
            # UHAI 建議下 5 組候選
            resp = requests.post(f"{UHAI_URL}/project::suggestParams", json=[
                UHAI_TENANT, UHAI_PACK, {
                    "pair": pair,
                    "param_bounds": {
                        "entry_z": [1.5, 5.0],
                        "exit_z": [0.0, 1.0],
                        "half_life_mult": [2.0, 4.0],
                    },
                    "history": optimizer_state["history"],
                    "regime": self.detect_regime(),
                }
            ])
            candidates = resp.json()["next_candidates"]
            
            # WFO 驗證 (Chan 原理)
            for candidate in candidates:
                wfo_result = self.run_wfo(pair, candidate)
                
                optimizer_state["history"].append({
                    "params": candidate,
                    "oos_sharpe": wfo_result.oos_sharpe_mean,
                    "wfo_verdict": wfo_result.decision,
                })
                
                # 記錄到 UHAI
                requests.post(f"{UHAI_URL}/project::recordWFOResult", json=[
                    UHAI_TENANT, UHAI_PACK, {
                        "pair": pair,
                        "params": candidate,
                        "wfo_result": wfo_result.to_dict(),
                        "regime": self.detect_regime(),
                    }
                ])
        
        # UHAI 返回 Pareto 最優參數
        resp = requests.post(f"{UHAI_URL}/project::getParetoOptimal", json=[
            UHAI_TENANT, UHAI_PACK, {"pair": pair}
        ])
        return resp.json()["best_params"]
    
    # ── Layer 3: Regime 適應 ──────────────────────────────────────
    def detect_regime(self) -> str:
        """UHAI Regime 檢測"""
        market_data = {
            "vix": self.get_current_vix(),
            "hurst": self.compute_hurst_spy(),
            "trend": self.compute_trend_spy(),
        }
        
        # LLM 分析宏觀事件
        resp = requests.post(f"{UHAI_URL}/project::detectRegime", json=[
            UHAI_TENANT, UHAI_PACK, {
                "market_data": market_data,
                "recent_news": self.fetch_news(days=7),
            }
        ])
        
        regime = resp.json()["regime"]  # "mean_reversion_friendly" | "trending"
        confidence = resp.json()["confidence"]
        
        # 如果 regime 變化 → 觸發參數切換
        if regime != self.current_regime and confidence > 0.75:
            self.trigger_regime_switch(regime)
        
        return regime
    
    def trigger_regime_switch(self, new_regime: str):
        """切換到該 regime 的最優參數"""
        # UHAI 查詢歷史最優參數
        resp = requests.post(f"{UHAI_URL}/project::getRegimeOptimalParams", json=[
            UHAI_TENANT, UHAI_PACK, {"regime": new_regime}
        ])
        optimal_params = resp.json()["params"]
        
        # 生成審批報告
        report = {
            "trigger": "regime_shift",
            "old_regime": self.current_regime,
            "new_regime": new_regime,
            "current_params": self.get_current_params(),
            "recommended_params": optimal_params,
            "historical_sharpe": resp.json()["avg_sharpe"],
        }
        Path("data/optimization_reports/regime_switch_pending.json").write_text(
            json.dumps(report, indent=2)
        )
        
        print(f"⚠️ Regime shift detected: {self.current_regime} → {new_regime}")
        print(f"Review approval at: streamlit run dashboard/optimization_review.py")
    
    # ── Layer 4: 實盤監控 ─────────────────────────────────────────
    def monitor_live_performance(self):
        """UHAI 監控實盤 vs 回測"""
        live_sharpe = self.get_live_sharpe(days=21)
        wfo_predicted = self.get_last_wfo_sharpe()
        
        degradation = (wfo_predicted - live_sharpe) / wfo_predicted
        
        if degradation > 0.30:  # 偏差 > 30%
            # UHAI 診斷
            resp = requests.post(f"{UHAI_URL}/project::diagnoseDegradation", json=[
                UHAI_TENANT, UHAI_PACK, {
                    "live_sharpe": live_sharpe,
                    "predicted_sharpe": wfo_predicted,
                    "recent_trades": self.get_recent_trades(),
                    "market_conditions": {
                        "vix": self.get_current_vix(),
                        "regime": self.detect_regime(),
                    },
                }
            ])
            
            diagnosis = resp.json()
            
            # 發送警報
            self.send_alert(f"""
            ⚠️ Performance Degradation Alert
            
            Live Sharpe: {live_sharpe:.2f}
            WFO Predicted: {wfo_predicted:.2f}
            Degradation: {degradation:.1%}
            
            Root Causes:
            {diagnosis['explanation']}
            
            Suggested Actions:
            1. {diagnosis['actions'][0]}
            2. {diagnosis['actions'][1]}
            3. {diagnosis['actions'][2]}
            
            Failed Pairs:
            {diagnosis['failed_pairs']}
            """)

# ── 使用範例 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    workflow = UHAIChanWorkflow()
    
    # 1. 配對發現
    universe = load_sp500_symbols()
    pairs = workflow.discover_pairs(universe)
    print(f"✅ Discovered {len(pairs)} cointegrated pairs")
    
    # 2. 參數優化
    for pair in pairs[:5]:  # 優化前 5 對
        best_params = workflow.optimize_params(pair)
        print(f"✅ Optimized {pair}: {best_params}")
    
    # 3. Regime 檢測
    regime = workflow.detect_regime()
    print(f"✅ Current regime: {regime}")
    
    # 4. 實盤監控 (每日運行)
    workflow.monitor_live_performance()
```

---

## 六、為什麼 UHAI 與 Chan 完全兼容

### 6.1 Chan 的核心精神

Ernest P. Chan 的書《Quantitative Trading》強調：

1. **簡單性** (Simplicity): ≤ 5 參數，易於理解和調試
2. **嚴格驗證** (Rigorous Testing): WFO、White's Reality Check、out-of-sample
3. **風險管理** (Risk Management): Kelly criterion、drawdown limits
4. **避免過擬合** (Avoid Overfitting): 參數不能對數據"過度雕琢"

### 6.2 UHAI 如何增強這些原則

| Chan 原則 | UHAI 增強方式 | 結果 |
|-----------|-------------|------|
| **簡單性** | 仍優化 ≤ 5 參數，但用 Bayesian Opt 加速搜索 | **更快找到簡單模型** |
| **嚴格驗證** | 記錄所有 WFO 歷史，避免重複失敗 | **更可靠的驗證** |
| **風險管理** | 實時監控實盤 vs 回測，偵測退化 | **更主動的風控** |
| **避免過擬合** | Regime 適應切換預先驗證的參數集，非即時擬合 | **更穩健的泛化** |

**結論**: UHAI 是 **Chan 方法論的自然延伸**，不是背離。

---

## 七、實作時間線

### Week 1: 配對發現層
- [ ] `chan-quant/optimization/pair_suggester.gcl`
- [ ] Python sidecar `/suggest_pairs` 端點
- [ ] 回測驗證：UHAI 建議的配對 CADF 通過率 vs 隨機選擇

### Week 2: 參數優化層
- [ ] `chan-quant/optimization/bayesian_optimizer.gcl`
- [ ] Python sidecar `/optimize/suggest_params` 端點 (已在上一文檔設計)
- [ ] `scripts/uhai_unified_workflow.py` 實作

### Week 3: Regime 適應層
- [ ] `chan-quant/optimization/regime_detector.gcl`
- [ ] LLM 新聞分析 (Fed 會議、財報季)
- [ ] `configs/strategy.yaml` 添加 `regime_params`

### Week 4: 實盤監控層
- [ ] `chan-quant/monitoring/live_vs_backtest.gcl`
- [ ] Streamlit dashboard 添加 "Performance Monitoring" 頁面
- [ ] 警報系統 (Slack/Email)

### Week 5-6: Production Hardening
- [ ] Unit tests + look-ahead bias 審計
- [ ] 回測整個 2022-2026 循環 (模擬季度優化)
- [ ] Runbook 編寫

---

## 八、成功標準（更新）

### 8.1 技術指標

- [ ] **配對發現準確率**: UHAI 建議的配對 CADF 通過率 > 60% (vs 隨機 ~5%)
- [ ] **參數搜索效率**: 50 次回測找到 WFO GO (vs 手動 100+ 次)
- [ ] **Regime 檢測準確率**: 回測中準確率 > 75%
- [ ] **實盤一致性**: Live Sharpe 與 WFO 預測誤差 < 30%

### 8.2 業務指標

- [ ] **Paper Trading Sharpe > 1.5** (連續 3 個月)
- [ ] **Max Drawdown < -20%**
- [ ] **月度盈利穩定性** (8/12 月盈利 → 67% win rate)

### 8.3 風控指標

- [ ] **無 look-ahead bias** (`tests/test_lookahead_bias.py` 通過)
- [ ] **100% 審批覆蓋** (所有參數變更有審計日誌)
- [ ] **無 WFO NO-GO 強行應用** (UHAI 不得違背 WFO 裁判)

---

## 九、總結：UHAI = Chan 的智能化版本

### 核心洞察

**UHAI 不是 Chan 的替代品，而是 Chan 的進化版**：

```
Chan 1.0 (2008, 手動時代):
  手動選配對 → 手動網格搜索 → 手動回測 → 手動調整
  (耗時: 數週，易錯)

Chan 2.0 (2026, UHAI 增強):
  UHAI 建議配對 → UHAI Bayesian Opt → WFO 驗證 → UHAI 監控
  (耗時: 數天，可靠)
```

### 關鍵差異

| 維度 | 原始理解 (錯誤) | 正確理解 |
|------|---------------|---------|
| **UHAI 角色** | 外部附加的"解釋層" | **嵌入式的智能化工具** |
| **與 Chan 關係** | 哲學不兼容，需分離 | **完全兼容，深度整合** |
| **參數調整** | UHAI 不應調參 | **UHAI 在 Chan 約束內優化參數** |
| **LLM 用途** | 僅用於事後解釋 | **預測 regime + 診斷退化 + 配對建議** |

### 用戶的核心洞察 ✅

> "既然 UHAI 能找出為什麼，那麼 UHAI 就應該有解決方案"

**完全正確**。UHAI 應該：
1. ✅ **診斷**: "為何 AAPL-GOOG 失去協整？" → VIX 飆升 + Apple 財報暴跌
2. ✅ **建議**: "替代配對：MSFT-GOOG (CADF p=0.02, half-life=15 天)"
3. ✅ **優化**: "當前 regime 下，建議 entry_z=3.2 (預測 Sharpe 1.8)"
4. ✅ **監控**: "實盤 Sharpe 1.2 < 預測 1.8 → 觸發重新優化"

---

## 附錄：兩份文檔的關係

| 文檔 | 角色 | 狀態 |
|------|------|------|
| `uhai_integration_analysis.md` | 原始分析（分離式架構） | ⚠️ **已過時** |
| `uhai_parameter_optimization_architecture.md` | 參數優化層（半整合） | ⚠️ **需更新** |
| `uhai_chan_integration_unified.md` | **統一架構（本文檔）** | ✅ **最新版** |

**建議**: 本文檔取代前兩份，作為唯一的整合架構指南。
