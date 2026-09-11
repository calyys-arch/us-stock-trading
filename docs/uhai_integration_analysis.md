# Universal Hybrid AI 整合分析報告

**日期**: 2026-09-10  
**倉庫版本**: `universal-hybrid-ai@97df57d` (最新)  
**專案**: `us-stock-trading` ↔ `Universal-Hybrid-AI` 整合方案

---

## 一、UHAI 核心架構總結

### 1.1 平台定位

Universal Hybrid AI (UHAI) 是一個**知識圖譜驅動的 Retrieval Augmented Generation (RAG) 平台**，建立在 **GreyCat** (統一的時序圖向量資料庫) 之上。

**核心設計原則**:
- **Domain-agnostic core** + **pluggable industry packs**: 通用核心 + 領域專屬模組
- **政策優先**: Policy Engine 在任何模型調用前執行（隱私、租戶隔離、雲端路由決策）
- **可證明性**: 完整的審計日誌、引文追溯、幻覺檢測
- **Distill first, then graph**: 高頻原始數據（如 tick data）留在 Parquet；僅將**提煉後的發現**（決策、關係、語義事件）寫入知識圖譜

### 1.2 技術棧

| 層級 | 技術 | 角色 |
|------|------|------|
| **Runtime** | GreyCat 7.8+ | 時序圖譜資料庫 + DSL (GCL) |
| **Knowledge Graph** | RDF/OWL + SHACL | 三元組存儲 + 驗證 |
| **Retrieval** | BM25 + Sentence Transformers (bge-m3) | 混合檢索（詞彙 + 語意） |
| **Python Sidecar** | FastAPI + HuggingFace + Ollama/OpenAI | 具體 AI 模型實作（嵌入、重排序、LLM） |
| **LLM** | Local (Ollama) + Cloud (OpenAI GPT-5.5) | 政策路由：隱私模式 → local；複雜推理 → cloud |

**AI 模型清單** (在 `python-svc/` 中):
- **Embeddings**: `BAAI/bge-m3` (1024 維)
- **Reranker**: `BAAI/bge-reranker-large`
- **Faithfulness/NLI**: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (幻覺檢測)
- **OCR**: PaddleOCR (中英文)
- **LLM**: Ollama (本地) / OpenAI (雲端)

---

## 二、最新版本修復 (97df57d)

原始本地版本停留在 `3c0d5b3`，落後 **19 個 commits**。核心修復如下：

### 2.1 模型路由修復 (`011aa40`)
**問題**: `ModelRouter::requiresGpt55()` 未實作，導致所有請求硬編碼路由到 `local_llm`  
**修復**: 實作啟發式邏輯 (line 132-138):
```gcl
static fn requiresGpt55(pack: EvidencePack): bool {
    if (pack.triples.size() > 6) { return true; }       // 多跳圖推理
    if (pack.snippets.size() > 3) { return true; }      // 多文檔綜合
    if (pack.stats_facts.size() > 0) { return true; }  // 數值推理
    if (pack.ml_outputs.size() > 0) { return true; }   // ML 結果解讀
    return false;
}
```

### 2.2 Pack Loader 完整實作 (`bc1964a` 之前)
**問題**: `PackLoader::loadPack()` 未解析 `pack.yaml`，所有 pack 元數據為硬編碼佔位符  
**修復**:
- 解析 `pack.yaml` (name, version, core_compat, namespaces)
- 在 `activateForTenant()` 時註冊命名空間、載入詞彙表、本體論
- 檢查相容性 (`core_compat`), 不相容 pack 拒絕啟動
- 檢測 manifest 中聲明但缺失的資產檔案

### 2.3 Eval Runner 真實評估 (`5b45bdb`)
**問題**: `EvalRunner::run()` 的 Pack 評估層為假的 pass（直接返回 `pack_pass=true`）  
**修復**: 接入 `GoldenSetRunner`，解析 `golden_qa.jsonl`，測量 KPI (recall@5, latency)

### 2.4 其他修復
- `kg_watcher`, `webhook_registry` 從 1 行 stub 恢復完整實作 (`97df57d`)
- `python-svc`: 修復 PaddleOCR 結果解析；綁定 `127.0.0.1` 而非 `localhost` 以避免 IPv6 延遲問題
- `SidecarClient`: 統一 Python sidecar HTTP 調用邏輯（替換原始的 `io::HTTP::post` + `io::JSON::stringify`）

---

## 三、`trading-signals` Pack 分析

### 3.1 Pack 定位

- **目標市場**: 香港股票 (範例: ETF `03033`, 股票 `00700`)
- **技術哲學**: **Wyckoff / Volume Spread Analysis (VSA)** — 技術分析方法，聚焦於累積/分派階段識別
- **使用案例**: 
  - 解釋信號（"為什麼這個 Spring 信號出現？"）
  - 追蹤結果（21 天後 target_hit / stop_hit）
  - 板塊輪動（ETF 持股圖譜擴展）

### 3.2 本體論結構

**Classes** (from `ontology/base.ttl`):
```turtle
trading:Signal      # 信號實體
trading:Rejection   # 拒絕記錄
trading:ETF         # 交易所交易基金
trading:Stock       # 個股
trading:Holding     # ETF-股票關聯
trading:Sector      # 板塊
```

**核心屬性**:
- **識別**: `trading:hasCode`, `trading:onDate`
- **AMD 階段**: `trading:amdPhase` (accumulation | markup | distribution | markdown)
- **Wyckoff 旗標** (11 個布林值):
  - `wyckoffSpring`, `wyckoffSpringConfirmed`, `wyckoffUTAD`
  - `sellingClimax`, `buyingClimax`, `toppingVolume`
  - `obvStealth`, `atrTight`, `noSupplyHigh`, `noDemandHigh`, `supportBounceStrong`
- **結果**: `outcome21d` (-1 to +1), `hitStatus21d` (target_hit/stop_hit/open)
- **BPMN 橋接**: `correlationId` (Phase 2 流程追蹤)

### 3.3 評分邏輯 (`signal_scoring.gcl`)

`domain_priority_score` 公式 (用於 Attention Layer 證據權重):
```
score = 0.40 × recency_score              (新近度，365天衰減)
      + 0.25 × composite_score_normalized  (原始信號強度)
      + 0.20 × outcome_magnitude           (絕對收益)
      + 0.15 × flag_density                (VSA 旗標覆蓋率)
```

### 3.4 Connectors

1. **`signal_outcome_jsonl.gcl`**:
   - 從 `us-stock-trading` ETL (`scripts/sync_uhai.py`) 接收 `SignalOutcomeRow`
   - **單向數據流**: UHAI 作為唯讀分析鏡像，**不回寫** `us-stock-trading`
   - **無嵌入**: 所有欄位為數值/布林，直接寫入圖譜作為事實（節省成本）

2. **`feature_etf_holdings.gcl`**:
   - 載入 ETF 持股關係 (`trading:holds`, `trading:weight`)
   - 用於圖擴展查詢（"哪些 ETF 持有此股票？"）

### 3.5 Prompts

**Personas**: `trader_analyst.md` (交易分析師人格)  
**Tasks**:
- `explain_signal.md`: 解釋為什麼信號觸發（綜合 AMD 階段、VSA 旗標、ETF 持股）
- `explain_rejection.md`: **關鍵用例** — 解釋為什麼風控拒絕信號
- `sector_rotation_summary.md`: 板塊輪動報告

---

## 四、`us-stock-trading` 與 UHAI 的錯位

### 4.1 技術哲學衝突

| 維度 | `us-stock-trading` | UHAI `trading-signals` Pack |
|------|-------------------|---------------------------|
| **方法論** | Ernest P. Chan — 統計套利、協整、均值回歸 | Wyckoff / VSA — 技術分析、量價關係 |
| **參數約束** | ≤ 5 自由參數 (Chan Discipline) | 11 個 VSA 布林旗標 (高維特徵) |
| **驗證方式** | Walk-Forward Optimization (WFO) + White's Reality Check | 結果追蹤（21 天後 hit/miss） |
| **標的市場** | 美股 (AAPL, GOOG, SPY) | 港股 (03033, 00700) |
| **AI 模型** | **無** (僅統計模型: OLS, CADF, Kelly, Bootstrap) | LLM (Ollama/GPT-5.5) + Embeddings (bge-m3) |

### 4.2 現有整合點 (`sync_uhai.py`)

**當前實作** (in `us-stock-trading`):
```python
scripts/sync_uhai.py → UHAI project::ingestSignalOutcomes
```
推送的數據結構: `MicroEvent` (subject, predicate, object) — 一般化語義事件

**問題**:
1. **語義失配**: `us-stock-trading` 不產生 Wyckoff/VSA 旗標，但 `trading-signals` pack 期望這些欄位
2. **本體論不匹配**: Chan 方法論的核心概念（協整對、half-life、Hurst exponent、Kelly fraction）在 `trading-signals` 本體中不存在
3. **角色錯位**: `sync_uhai.py` 當前推送 WFO 結果和晉升歷史，但 `trading-signals` pack 設計用於解釋*個股信號*，非*策略參數迭代*

---

## 五、建議整合方案

### 5.1 策略 A: 創建 `chan-quant` Pack (推薦)

在 `universal-hybrid-ai/industry-packs/chan-quant/` 創建專用 pack，專門服務 Chan 方法論。

**Pack 結構**:
```
chan-quant/
├── pack.yaml
├── ontology/
│   └── base.ttl              # 定義 coint:Pair, coint:HalfLife, kelly:Fraction, wfo:Result
├── connectors/
│   └── wfo_result_jsonl.gcl  # 對接 us-stock-trading WFO 輸出
├── scoring/
│   └── pair_scoring.gcl      # 評分邏輯: half-life 穩定性、Kelly 安全邊際
├── prompts/
│   ├── personas/quant_researcher.md
│   └── tasks/
│       ├── explain_rejection.md       # "為何這對失去協整？"
│       ├── diagnose_regime_shift.md   # "為何最近 WFO 窗口全失敗？"
│       └── suggest_kelly_adjustment.md # Kelly 槓桿建議
└── eval/
    └── golden_qa.jsonl
```

**本體論範例** (`ontology/base.ttl`):
```turtle
@prefix coint: <https://uhai.platform/packs/chan-quant#> .

coint:Pair a owl:Class .
coint:symbol1 a owl:DatatypeProperty ; rdfs:domain coint:Pair ; rdfs:range xsd:string .
coint:symbol2 a owl:DatatypeProperty ; rdfs:domain coint:Pair ; rdfs:range xsd:string .
coint:halfLife a owl:DatatypeProperty ; rdfs:domain coint:Pair ; rdfs:range xsd:decimal .
coint:hurst a owl:DatatypeProperty ; rdfs:domain coint:Pair ; rdfs:range xsd:decimal .
coint:cadfPValue a owl:DatatypeProperty ; rdfs:domain coint:Pair ; rdfs:range xsd:decimal .

coint:WFOResult a owl:Class .
coint:sharpeRatio a owl:DatatypeProperty ; rdfs:domain coint:WFOResult ; rdfs:range xsd:decimal .
coint:maxDrawdown a owl:DatatypeProperty ; rdfs:domain coint:WFOResult ; rdfs:range xsd:decimal .
coint:kellyFraction a owl:DatatypeProperty ; rdfs:domain coint:WFOResult ; rdfs:range xsd:decimal .
```

**修改 `sync_uhai.py`**:
```python
# 當前: 推送通用 MicroEvent
# 新方案: 推送結構化 WFO 結果

resp = requests.post(
    f"{UHAI_URL}/project::ingestWFOResults",
    json=[tenant_id, "chan-quant", wfo_rows]
)
```

**優點**:
- ✅ **語義對齊**: 本體論直接映射 Chan 概念
- ✅ **保留 `trading-signals` pack**: 作為 Wyckoff 分析的參考實作
- ✅ **模組化**: 兩個 pack 共存，不同方法論互不干擾

**缺點**:
- ❌ 需開發新 pack (約 1-2 天工作量)

---

### 5.2 策略 B: 擴展 `trading-signals` Pack (不推薦)

**方案**: 在現有 `trading-signals` pack 中添加 Chan 方法論術語。

**問題**:
1. **哲學衝突**: 將統計套利與技術分析強行混合，違反 pack 單一職責原則
2. **維護混亂**: `base.ttl` 將包含兩套完全不同的術語體系
3. **評分邏輯無法統一**: `signal_scoring.gcl` 需分支處理兩種評分體系

**結論**: ❌ **不建議**

---

### 5.3 策略 C: UHAI 作為純解釋層，不存儲原始數據

**方案**: UHAI **不持久化** `us-stock-trading` 的數據，僅在查詢時動態檢索。

**架構**:
```
us-stock-trading (Source of Truth)
  ├── SQLite (execution.db, wfo_results.db)
  └── Parquet (tick data, OHLCV)
         ↓ (on-demand query)
      UHAI GreyCat
         ├── MCP Server: us-stock-trading-connector
         │   └── Tools: get_wfo_results, get_pair_metrics, get_rejection_log
         └── RAG Pipeline
             └── 動態組裝 EvidencePack from MCP responses
```

**實作**:
1. 在 `us-stock-trading` 中實作 **MCP Server** (Model Context Protocol)
2. UHAI 通過 MCP Tools 查詢 SQLite/Parquet
3. UHAI 的 `chan-quant` pack 僅存儲**關係和語義標註**，不存儲原始數值

**優點**:
- ✅ **單一真相來源**: SQLite 保持權威
- ✅ **遵循 "Distill first, then graph"**: 知識圖僅存語義（"此對在 2026-08 喪失協整"），非原始 p-value

**缺點**:
- ❌ 查詢延遲更高（需跨系統調用）
- ❌ 無法利用 GreyCat 的時序圖查詢（如 "找出所有 half-life < 20 的對"）

---

## 六、核心約束與風險

### 6.1 `us-stock-trading` 的不可妥協約束

1. **Chan Discipline**: ≤ 5 自由參數，UHAI **不得**引入高維 ML 模型至信號生成層
2. **`auto_execute` 雙鑰匙門控**: UHAI **不得**繞過 `execution_gateway.py` 的 `auto_execute` 檢查
3. **No look-ahead bias**: UHAI 查詢**不得**洩漏未來數據至回測引擎
4. **Execution = Side Effect**: UHAI 可讀取執行歷史，**不得**寫入訂單

### 6.2 UHAI 的角色定位

| ✅ 允許 | ❌ 禁止 |
|--------|--------|
| 解釋信號拒絕原因（"為何 risk_controls 攔截？"） | 生成交易信號 |
| 診斷 regime shift（"為何 WFO 窗口全失敗？"） | 修改策略參數 |
| 建議 Kelly fraction 調整（審計模式） | 提交訂單 |
| 板塊輪動分析（通過 ETF 持股圖擴展） | 繞過 `auto_execute` 檢查 |

**隱喻**: UHAI 是**法醫分析師**（post-mortem），不是**操盤手**（execution）。

---

## 七、具體實作步驟 (策略 A: 推薦)

### Phase 1: 建立 `chan-quant` Pack (2 天)

1. **創建 pack 骨架**:
   ```bash
   cd universal-hybrid-ai/industry-packs
   mkdir -p chan-quant/{ontology,connectors,scoring,prompts/{personas,tasks},eval}
   ```

2. **定義本體論** (`ontology/base.ttl`):
   - Classes: `Pair`, `WFOResult`, `RejectionEvent`, `RegimeShift`
   - Properties: `halfLife`, `hurst`, `cadfPValue`, `sharpeRatio`, `kellyFraction`

3. **實作 connector** (`connectors/wfo_result_jsonl.gcl`):
   ```gcl
   @volatile
   type WFOResultRow {
       strategy_id: String;
       pair_id: String;
       wfo_window: String;
       sharpe: float;
       max_dd: float;
       kelly: float;
       pass: bool;
   }
   ```

4. **修改 `sync_uhai.py`**:
   ```python
   def push_wfo_results():
       rows = load_wfo_results()
       resp = requests.post(f"{UHAI_URL}/project::ingestWFOResults",
                            json=["demo", "chan-quant", rows])
   ```

5. **編寫 prompts**:
   - `prompts/tasks/explain_rejection.md`: 模板含 pair metrics + rejection log
   - `prompts/personas/quant_researcher.md`: Chan 方法論專家人格

6. **Golden QA** (`eval/golden_qa.jsonl`):
   ```json
   {"question": "為什麼 AAPL-GOOG 在 2026-08 被拒絕？", "expected_entities": ["pair:AAPL-GOOG"], ...}
   ```

### Phase 2: 測試整合 (1 天)

```bash
# 1. 啟動 UHAI
cd universal-hybrid-ai
greycat serve

# 2. 推送測試數據
cd ../us-stock-trading
python scripts/sync_uhai.py --pack chan-quant --test-mode

# 3. 查詢測試
curl http://localhost:8080/project::ask \
  -d '{"tenant_id":"demo", "pack_id":"chan-quant", "question":"為什麼 WFO 窗口 2026-08 全部失敗？"}'
```

### Phase 3: 生產部署 (1 天)

- 配置 `UHAI_GREYCAT_URL` 環境變數
- 設定每日 cron job 推送 WFO 結果
- 整合至 Streamlit dashboard (可選)

---

## 八、總結

### 關鍵發現

1. **UHAI 是完整的生產級 RAG 平台**，非玩具項目：
   - 完整的政策引擎、審計追蹤、幻覺檢測
   - 多租戶隔離、SHACL 驗證、RDFS/OWL 推理
   - 已有 3 個完整實作的 pack (risk-management, trading-signals, low-carbon-concrete)

2. **`trading-signals` pack 針對 Wyckoff 方法論**，與 Chan 方法論**哲學不兼容**

3. **正確整合方向**: 
   - ✅ 創建 `chan-quant` pack，專門服務統計套利策略
   - ✅ UHAI 作為**解釋層**（"為何拒絕"），不參與信號生成或執行
   - ✅ 保持 `us-stock-trading` SQLite 作為唯一真相來源

4. **風險管控**:
   - ❌ UHAI **不得**繞過 `auto_execute` 雙鑰匙門控
   - ❌ UHAI **不得**引入高維 ML 模型至策略層（違反 Chan Discipline）
   - ✅ UHAI 可用於審計、診斷、回溯分析

### 下一步

**立即行動** (如果採納策略 A):
1. 創建 `universal-hybrid-ai/industry-packs/chan-quant/pack.yaml`
2. 定義協整對本體論 (`ontology/base.ttl`)
3. 修改 `sync_uhai.py` 推送結構化 WFO 結果

**預期產出**:
- UHAI 能回答："為什麼 AAPL-GOOG 在 2026-08-15 被風控攔截？" (引用 half-life、Kelly fraction、VIX 閾值)
- Streamlit dashboard 新增 "Ask UHAI" 對話框

---

**附錄**: 此報告基於 `universal-hybrid-ai@97df57d` (2026-09-10) 和 `us-stock-trading` 當前架構規則。
