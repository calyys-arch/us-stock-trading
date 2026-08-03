# 系統轉型計畫：盤中市場微觀結構交易系統（Microstructure Pivot Plan）

> 狀態：**計畫草案，尚未動工** — 待使用者確認後才開始實作。
> 日期：2026-07-29
> 背景：現有兩個日線策略（pairs trading、cross-sectional mean reversion）在
> 20 檔科技股宇宙上經過完整回測 + WFO 自我改進迴圈驗證，全部 NO-GO/REJECTED。
> 使用者決定轉向「與機構訂單流對齊」的盤中微觀結構系統（liquidity sweep、
> VWAP、order-flow imbalance、L2 absorption），本文件是完整的轉型藍圖。

---

## 0. 目標與定位

**從**：日線頻率、統計套利（均值回歸）——被驗證在當前市場結構下無效。
**到**：盤中（1 分鐘級）流動性導向系統——不預測價格，改為辨識機構執行行為
（掃流動性、吸收、失衡回補）並站在同一邊。

**不變的原則**（沿用現有系統的紀律，這是本系統最大的資產）：
- Observe-only 優先：新訊號一律先記錄不下單，通過驗收門檻才逐步升級。
- 驗收門檻制：WFO + Monte Carlo + 參數紀律（Chan ≤5 自由參數）照舊適用。
- 誠實回報：資料來源、滑價假設、樣本不足一律標註在報告裡。
- 雙鑰上線：`auto_execute` + dashboard arm 的機制保留。

**宇宙**：沿用現有 20 檔高流動性科技巨頭（`configs/universe.yaml`）。
研究材料明確建議「只交易 mega-cap 高流動性股票」——現有宇宙正好符合，
不需要重選。

---

## 1. 現有資產盤點（重用地圖）

### 1a. 直接重用（不改或小改）

| 元件 | 路徑 | 在新系統的角色 |
|---|---|---|
| Tick/L2 擷取 | `python/interfaces/ibkr_tick_capture.py` | 資料護城河：每日 RTH 錄製 tick + 深度，供 Phase 3 回放驗證 |
| 交易日曆/RTH | `python/core/calendar.py` | session 開收盤、提前收盤、尾盤 flatten 窗口 |
| Broker 轉接層 | `python/interfaces/ibkr_broker.py`、`python/core/sim_broker.py` | market/limit 下單殼（需擴充訂單類型，見 §5） |
| 執行閘道 | `python/core/execution_gateway.py` | observe/auto 雙鑰、EOD flatten、bus 事件流 |
| 風險引擎 | `python/core/risk_engine.py` + `configs/risk.yaml` | ADV/PDT/Reg T/Kelly 硬上限（需補盤中規則，見 §6） |
| Rate limiter | `python/core/rate_limiter.py` | IB 歷史資料 pacing |
| 事件資料 | `python/data/finnhub_client.py`、`edgar_client.py`、`python/interfaces/finnhub_calendar.py` | 財報/總經事件的「暫停交易」閘門 |
| 驗證框架 | `python/backtest/walk_forward.py`、`monte_carlo.py`、`param_guard.py`、`optimize.py`、`promotion.py` | 盤中策略的驗收與晉升管線（餵入盤中報酬序列即可） |
| Dashboard 殼 | `dashboard/app.py`、`engine_bridge.py` | FastAPI + 引擎啟停 + observe/auto API |

### 1b. 需要修改

| 元件 | 現況 | 需要的改動 |
|---|---|---|
| `IbkrFeed` | 2 秒輪詢 bid/ask 合成 mid tick | 改用 `reqRealTimeBars`（5 秒 bar）聚合成 1m + 對重點股訂 tick-by-tick；L2 限 ≤3 檔 |
| `ibkr_price_source` / `price_cache` | 硬編碼日線 `ADJUSTED_LAST` | 新增 1 分鐘 `TRADES` 歷史抓取路徑與獨立快取（見 §3） |
| 回測引擎 | 日線假設寫死（vector: 日開→日收；pairs: 日收盤序列） | 新建盤中事件回測器（見 §4），舊引擎保留給存量策略 |
| `MarketSnapshot.level2_*` | 欄位存在但從未填值 | live 路徑填入深度快照，供 absorption 訊號用 |
| trap_detector | 日線 report-only | `order_book_churn`、`marking_the_close` 抽成即時串流特徵 |

### 1c. 完全缺失（需新建）

1. 1 分鐘歷史資料管線（IB pacing-aware 批量抓取 + 本地快取）
2. Session VWAP / Anchored VWAP / Volume Profile（POC/VAH/VAL）計算模組
3. Context Engine：流動性池地圖（YDH/YDL、PMH/PML、EQH/EQL、整數關卡）
4. Signal Engine：sweep-reclaim、FVG、ORB 三個可回測訊號 + L2 absorption
5. 盤中事件回測器（1m bar 迴圈、滑價模型、session 感知）
6. Tick/L2 回放引擎（讀取已錄 JSONL 重建訂單簿）
7. Live 訊號迴路（dashboard 目前只餵 DataEngine，沒有真正跑策略）
8. 訊號日誌（signal journal）：observe-only 期間記錄每個訊號的完整上下文

---

## 2. 系統架構

沿用研究材料的四引擎架構，映射到現有 bus 事件流：

```
                         ┌──────────────────────────────────────┐
  1m bars / ticks / L2 → │ Context Engine                       │
  (feed 或 回測資料)      │ python/microstructure/context.py     │
                         │ 流動性池 + VWAP 家族 + Volume Profile │
                         └──────────────┬───────────────────────┘
                                        │ ContextState（每 bar 更新）
                                        ▼
                         ┌──────────────────────────────────────┐
                         │ Signal Engine                        │
                         │ python/microstructure/signals/*.py   │
                         │ sweep_reclaim / fvg_retest / orb /   │
                         │ l2_absorption（Phase 3）              │
                         └──────────────┬───────────────────────┘
                                        │ RawSignal → bus
                                        ▼
                         ┌──────────────────────────────────────┐
                         │ Risk Engine（現有 RiskEngine 擴充）    │
                         │ 1% 風險部位、ATR 停損距離、時間停損、  │
                         │ 事件暫停、日虧損 kill-switch          │
                         └──────────────┬───────────────────────┘
                                        │ qualified_order → bus
                                        ▼
                         ┌──────────────────────────────────────┐
                         │ Execution Engine（現有 gateway 擴充）  │
                         │ limit/stop-limit、observe/auto 雙鑰、  │
                         │ EOD flatten（已有）                    │
                         └──────────────────────────────────────┘
```

### Context Engine（新建 `python/microstructure/context.py`）

每個交易日開盤前計算、盤中逐 bar 更新：

- **流動性池（Liquidity Target Zones）**：昨日高低（YDH/YDL）、盤前高低
  （PMH/PML）、近 N 根等高/等低（EQH/EQL，容差 = k×ATR）、整數關卡
  （$5/$10 級距，依價位自適應）。
- **VWAP 家族**：session VWAP（09:30 錨定）+ anchored VWAP（開盤、
  盤中極值、財報日錨點），皆從 1m bar 的 Σ(price×vol)/Σ(vol) 計算，
  並附 ±1σ/±2σ 帶。
- **Volume Profile**：當日滾動 POC/VAH/VAL（bar 近似，非 tick 精度——
  誠實標註）。
- **Opening Range**：09:30–09:45 的高低（15 分鐘 ORB）。

### Signal Engine（新建 `python/microstructure/signals/`）

每個訊號一個模組、各自 ≤5 個自由參數、統一輸出 `MicroSignal`
（symbol、方向、進場價、停損價、目標價、有效期、觸發上下文）：

**S1 — Liquidity Sweep & Reclaim（`sweep_reclaim.py`）**
1m bar 邏輯：價格刺穿流動性池 ≥ `sweep_min_atr` × ATR → `reclaim_bars`
根內收回關卡內側 → 反向進場，停損 = 掃出的極值外 `stop_atr_mult` × ATR，
目標 = 對側流動性池或 session VWAP。
自由參數（3）：`sweep_min_atr`、`reclaim_bars`、`stop_atr_mult`。

**S2 — Fair Value Gap Retest（`fvg_retest.py`）**
3 根 bar 序列：bar2 為大幅方向 bar（量 > `vol_mult` × 20-bar 均量）且
bar1 高點與 bar3 低點不重疊 → 在缺口 50% 掛限價單順勢回補進場，
停損 = 缺口起點外側，時間停損 = `expiry_bars` 根內未觸發即撤單。
自由參數（3）：`vol_mult`、`entry_pct`（預設 0.5）、`expiry_bars`。

**S3 — Opening Range + VWAP 方向過濾（`orb_vwap.py`）**
09:30–09:45 不進場（研究材料的黃金法則，直接寫成硬規則而非參數）。
09:45 後：突破 OR 高/低 且 價格與 VWAP 同側 才順勢進場;
若盤前跳空方向與 OR 突破方向相反 → 視為出貨陷阱、只做反向。
自由參數（2）：`or_minutes`（15/30）、`vwap_side_filter`（bool）。

**S4 — L2 Absorption / Iceberg（`l2_absorption.py`，Phase 3）**
Time & Sales + 深度：同一價位持續成交大量但價格不穿越（吸收），
且 `order_book_churn_score` 低（非幌騙）→ 該價位視為機構防守位。
先以「特徵 + 訊號日誌」形式跑 observe-only，等錄滿足夠深度資料再回測。

**全域過濾器（所有訊號共用，不算自由參數——是操作規則）**：
- 開盤前 15 分鐘與收盤前 flatten 窗口不進場（後者已有）。
- FOMC/CPI/NFP 等總經事件前後 10 分鐘暫停（Finnhub economic calendar,
  已有快取層）；個股財報日整日暫停（Finnhub earnings calendar）。
- 當沖限定：不留倉（gateway 的 EOD flatten 已實作）。

---

## 3. 資料層計畫

### 3a. 1 分鐘歷史 bar 管線（新建 `python/data/intraday_cache.py`）

- 來源：IB `reqHistoricalData`，`barSizeSetting="1 min"`、`whatToShow="TRADES"`、
  `useRTH=True`。注意：`ADJUSTED_LAST` 不支援 intraday，改用 TRADES 就緒
  （除息日的價格跳空需在回測中標註，不做回溯調整——誠實處理）。
- IB 限制：1 分鐘資料單次請求最多約 1 個月、pacing 約 60 req/10min。
  20 檔 × 12 個月 × 12 次請求 ≈ 2,880 個請求 ≈ 8 小時全量回補——
  用現有 `rate_limiter` + 斷點續傳，跑一次後增量更新。
- 快取格式：`data/history_1m/<SYMBOL>/<YYYY-MM>.parquet`（parquet 比 CSV
  小一個量級、載入快；需在 requirements 加 `pyarrow`）。
- 初期範圍：**近 12 個月**（20 檔 × ~390 bar/日 × ~250 日 ≈ 200 萬列,
  完全可處理）。夠 WFO 切出 8–10 個 fold。

### 3b. Tick / L2 資料護城河（沿用 + 排程化）

- `scripts/capture_market_microstructure.py` 改為每個交易日自動啟動
  （RTH 期間錄製、收盤自動停止），L2 依 IB 上限選 3 檔輪替
  （建議固定 NVDA、AAPL + 1 檔輪換）。
- 這是 S4 的前置投資：**IB 不提供歷史深度資料**，只能自己往前錄。
  錄滿 4–6 週才有第一批可回放驗證的樣本。

### 3c. 明確不做（誠實聲明）

- **Dark pool ratio / 大宗 print 追蹤**：IB 不提供逐筆 off-exchange 歸屬,
  FINRA ATS 週報延遲兩週、只能當背景參考。若日後要做，需訂 Polygon.io
  （~$200/月）或 Databento——列為 Phase 3 之後的選配，先不承諾。
- **官方 SIP 全市場 tape**：成本與必要性不符，mega-cap 用 IB feed 已足夠。

---

## 4. 回測層計畫

### 4a. 盤中事件回測器（新建 `python/backtest/intraday_engine.py`）

- 1m bar 逐根事件迴圈，每根 bar：更新 Context → 評估 Signal → 模擬委託。
- **防前視**：訊號只用「已收 bar」判定，成交發生在下一根 bar
  （限價單需下一根 bar 的範圍觸及掛單價才成交）。
- **滑價/成本模型**（比日線嚴格,因為盤中成本是主要殺手）：
  - 進出各收 `half_spread + impact`：half_spread 用該股近期平均買賣價差
    （從錄到的 BidAsk tick 統計，沒有就用保守常數）；impact 依委託佔
    當根 bar 成交量比例遞增。
  - 佣金用 IB 分層費率。
  - 驗收時額外跑 **2× 滑價壓力測試**——過不了就是 NO-GO。
- session 感知：只在 RTH 交易、尾盤強制平倉、提前收盤日處理（日曆已有）。
- 輸出與現有引擎相同的 metrics 契約（sharpe、max_dd、daily_returns、trades）,
  讓 WFO/MC/promotion 管線**零修改直接餵**。

### 4b. Tick/L2 回放引擎（新建 `python/backtest/depth_replay.py`，Phase 3）

- 讀 `data/ticks/`、`data/depth/` JSONL，重建每檔的 top-10 訂單簿時間序列,
  供 S4 absorption 訊號離線驗證。
- 樣本量誠實標註：只有錄了幾週的資料就說幾週，不擴大解讀。

### 4c. 驗證紀律（沿用 + 調整）

- WFO 視窗改用盤中資料量級：例如 IS = 3 個月（~60 交易日 × 390 bar）,
  OOS = 1 個月，roll 1 個月——12 個月資料可得 8–9 個 fold。
- 每策略自由參數 ≤5（param_guard 照用）；`configs/param_grids.yaml` 新增
  三個訊號的粗網格（每軸 2–3 值）。
- `configs/goal.yaml` 新增盤中門檻：最少成交筆數（≥100 筆/OOS 才有統計
  意義,盤中頻率高、樣本這點反而比日線有利）、扣費後 profit factor ≥ 1.3、
  MC p5 ≥ 0、2× 滑價壓力測試仍為正。

---

## 5. 執行層計畫

- `IbkrBroker` 擴充：`stop_limit` 訂單、可選 TIF（DAY/IOC）、
  可選直接路由（`exchange="IEX"`/`"NASDAQ"`,預設仍 SMART——IBKR Pro
  無 PFOF 問題,直接路由列為可調參數而非必要）。
- **禁用市價單**進場（研究材料規則）：訊號全部用限價/停損限價執行,
  gateway 增加「進場單型白名單」檢查。
- Observe 模式的訊號日誌（新建 `python/microstructure/signal_journal.py`）：
  每個訊號記錄完整上下文（觸發規則、關卡價、VWAP 距離、假想進出場與
  盈虧）→ `data/signal_journal/<date>.jsonl`,累積「紙上實測 vs 回測」
  的落差統計——這是 go-live 前最重要的證據。

## 6. 風險層計畫（`configs/risk.yaml` 擴充）

- **部位規模 = 1% 帳戶風險 ÷ 停損距離**（研究材料規則,取代固定金額）。
- ATR 停損（訊號自帶）+ **時間停損**：進場後 `time_stop_minutes`（5–10 分鐘）
  未朝有利方向移動即出場。
- 日虧損 kill-switch：當日累計虧損達帳戶 X% → 全平 + 停止當日交易
  （gateway 已有 emergency flatten,加觸發器即可）。
- 同時持倉上限（初期 ≤3）；PDT 檢查沿用現有 RiskEngine。

## 7. Live 層計畫

- `IbkrFeed` 升級：20 檔全訂 `reqRealTimeBars`（5 秒）聚合 1m;
  重點 3 檔加訂 tick-by-tick + L2（clientId 分離慣例照舊）。
- **把訊號迴路真正接上 dashboard**（現在缺的最後一哩）：
  feed → Context/Signal → RiskEngine → gateway（observe）→ signal journal,
  dashboard 新增「今日訊號」面板與 VWAP/流動性池疊圖。
- 全程 observe-only,直到 §4c 門檻通過 + 訊號日誌與回測吻合度達標,
  才依現有雙鑰機制在 paper 帳戶開 auto。

---

## 8. 分階段里程碑

| Phase | 內容 | 完成定義 | 預估 |
|---|---|---|---|
| **0. 資料地基** | 1m 歷史管線 + intraday cache + VWAP/Context 模組 + 單元測試 | 20 檔 × 12 個月 1m bar 落地,Context 輸出經 spot-check | 先做 |
| **1. 可回測訊號** | S1/S2/S3 + 盤中回測器 + 滑價模型 + WFO/MC 驗證報告 | 三個訊號各出一份含壓力測試的驗收報告（GO 或誠實 NO-GO） | Phase 0 後 |
| **2. Live observe** | feed 升級 + 訊號迴路接 dashboard + signal journal;tick/L2 錄製排程化 | 連續 2 週每日訊號日誌,回測 vs 紙上落差報告 | 與 Phase 1 部分並行 |
| **3. 深度訊號** | L2 回放引擎 + S4 absorption + churn/mark-close 即時化 | 錄滿 ≥4 週深度資料後出 S4 驗證報告 | 資料到位後 |
| **4. Gated 上線** | 通過門檻的訊號在 paper 帳戶開 auto（雙鑰） | 30 個交易日 paper 實測績效 ≥ 回測的 70% | 最後 |

每個 Phase 結束出一份報告，**由驗收門檻決定是否前進**——和現有
self-improve 迴圈同樣的精神：系統可以誠實地說「這個訊號不行」。

## 9. 新增/修改檔案清單

**新建**
```
python/microstructure/__init__.py
python/microstructure/context.py            # 流動性池 + VWAP 家族 + volume profile
python/microstructure/signals/sweep_reclaim.py
python/microstructure/signals/fvg_retest.py
python/microstructure/signals/orb_vwap.py
python/microstructure/signals/l2_absorption.py   # Phase 3
python/microstructure/signal_journal.py
python/data/intraday_cache.py               # 1m bar 抓取 + parquet 快取
python/backtest/intraday_engine.py          # 盤中事件回測器 + 滑價模型
python/backtest/depth_replay.py             # Phase 3
scripts/backfill_intraday.py                # 1m 歷史回補 CLI
scripts/run_intraday_backtest.py            # 盤中回測 + 驗收報告 CLI
tests/test_context_engine.py
tests/test_intraday_signals.py              # 每訊號合成資料單測（含防前視測試）
tests/test_intraday_engine.py
tests/test_intraday_cache.py
```

**修改**
```
python/interfaces/ibkr_feed.py        # reqRealTimeBars + tick-by-tick 訂閱
python/interfaces/ibkr_broker.py      # stop-limit / TIF / 路由參數
python/core/execution_gateway.py      # 進場單型白名單、日虧損 kill-switch
python/core/risk_engine.py            # 1% 風險部位、時間停損
configs/risk.yaml                     # 盤中風險參數
configs/goal.yaml                     # 盤中驗收門檻
configs/param_grids.yaml              # S1/S2/S3 網格
configs/strategy.yaml                 # 新策略區塊（enabled/auto_execute 慣例照舊）
dashboard/engine_bridge.py            # 訊號迴路接入
requirements.txt                      # + pyarrow
```

**保留不動**：現有日線策略、回測引擎、self-improve 迴圈全部保留
（作為存量與對照組），不刪除。

## 10. 誠實風險聲明

1. **SMC/流動性概念的學術證據薄弱**。sweep/FVG/ORB 是交易圈流行概念,
   嚴謹研究支持有限。本計畫的立場：把它們當「待驗證的假設」,
   讓現有驗收門檻決定生死——**要有多數訊號被 NO-GO 的心理準備**,
   這正是系統在做它該做的事。
2. **盤中成本是第一殺手**。1m 頻率下,價差+滑價很容易吃掉全部毛利,
   所以滑價模型從嚴、外加 2× 壓力測試。
3. **IB 資料天花板**：L2 同時最多 ~3 檔、無歷史深度、無 dark pool 歸屬。
   S4 與 dark pool 相關功能受此限制,計畫已按此設計（forward capture）。
4. **1m TRADES 無除權息調整**：跨除息日的隔日跳空需要在回測中排除或
   標註（本系統當沖不留倉,影響被大幅緩解,但 anchored VWAP 跨日錨點
   需注意）。
5. **PDT 與帳戶規模**：日內頻繁進出需 ≥$25k 帳戶,現有 RiskEngine 的
   PDT 檢查會擋,live 前需確認帳戶狀態。

## 11. 待使用者確認的決策點

1. **起手範圍**：建議 Phase 0 + Phase 1 一起做（資料地基 + 三個可回測
   訊號),Phase 2 之後再依結果排。是否同意？
2. **L2 錄製的 3 檔**：建議 NVDA、AAPL 固定 + 第 3 檔每週輪換。可指定。
3. **外部資料預算**：dark pool / 更長歷史 tick 需要 Polygon/Databento
   （~$200+/月）。先不訂、用 IB 自錄——除非你想加速 Phase 3。
4. **1m 歷史深度**：預設回補 12 個月。要更長（IB 上限內可到 ~2 年,
   回補時間加倍）可以說。
