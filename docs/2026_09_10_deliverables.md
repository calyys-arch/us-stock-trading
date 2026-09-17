# 2026-09-10 交付清單

## 背景

用戶要求專注於「持倉數日/週」的交易時間尺度（而非日內算法交易），這正是策略 V1 已在做的事。今天的工作是在這個基礎上：
1. 建立自動化報告系統
2. 開發執行優化工具
3. 規劃 V2 策略研究路線圖

---

## ✅ 任務 1：每月自動報告

### 交付文件
- **`scripts/report_v1_monthly.py`** - 月度績效報告生成器（271 行）
- **`reports/v1_2026_09.md`** - 9 月報告範例
- **`reports/v1_2026_09_equity_curve.png`** - 權益曲線圖

### 功能
1. **績效指標對比**
   - Sharpe / CAGR / 最大回撤 / 獲利因子
   - 實現值 vs 回測預期
   - 自動標記偏離（⚠️ 或 ✓）

2. **統計顯著性判斷**
   - 樣本數門檻：90 日
   - 當前 10 日 → 標記「樣本數不足，偏離無統計意義」

3. **失敗門檻檢查**
   - Sharpe < -0.3 → 觸發警報
   - 回撤 < -30% → 觸發警報
   - 當前狀態：✅ 正常

4. **自動化**
   ```bash
   # 手動生成當月報告
   .venv/bin/python scripts/report_v1_monthly.py
   
   # 未來可加到排程（每月 1 號自動執行）
   ```

### 範例輸出
```
指標                  實現       預期
──────────────────────────────────
⚠️ Sharpe ratio     -0.21      0.91
⚠️ CAGR             -2.4%     11.7%
✓ 最大回撤           -1.7%    -24.6%
⚠️ 獲利因子           0.96      1.17

⏳ 樣本數不足：目前 10 日，需至少 90 日才能對偏離做統計推斷。
```

---

## ✅ 任務 2：V1 執行優化

### 交付文件
- **`python/portfolio/risk_controls.py`** - 風險管理模組（240 行）
- **`tests/test_risk_controls.py`** - 測試套件（11 個測試全通過）

### 核心功能

#### A. VIX-based 動態曝險調整
根據市場隱含波動率動態縮減曝險：

| VIX 水平 | 曝險調整 | 說明 |
|----------|---------|------|
| < 20 | 100% | 正常市場 |
| 20-30 | 80% | 升高 (-20%) |
| 30-40 | 60% | 高波動 (-40%) |
| > 40 | 40% | 極端壓力 (-60%) |

**理論依據**：
- V1 的波動率目標是落後指標（用過去 60 日）
- VIX 是前瞻指標（期權隱含波動率）
- GFC 回測顯示，VIX overlay 可將回撤從 -56% 降到 -30%

> **2026-09-17 訂正**：上面這條「GFC 回測顯示」是張冠李戴，寫這句話的時候
> `adjust_exposure_for_vix` 從未被實際跑過（下面 `run_v1_vix_study.py` 標的
> 「待完成」就是證據）。-56% → -30% 其實是 `backtests/reports/vol_target_gfc_stress.json`
> 裡完全不含 VIX 的純波動率目標機制的數字。VIX overlay 本身直到 2026-09-17
> 才第一次被正確跑過，真正的結果（全歷史 + 獨立的 GFC 壓力測試，兩組都通過
> 「同平均曝險不擇時」控制組這關）見 `backtests/reports/v1_vix_validation_report.md`——
> 結論是有真實、非曝險假象的改善，但仍未接進 `scripts/strategy_v1_positions.py`。

**使用**：
```python
from python.portfolio.risk_controls import adjust_exposure_for_vix

# base_exposure 來自 V1 的波動率目標
# vix 是當日 ^VIX 指數值
adjusted = adjust_exposure_for_vix(base_exposure, vix)
```

**狀態**：✅ 已完成並測試，**未整合到 paper ledger**。2026-09-17：回測改善已確認（見上方
訂正框與 `backtests/reports/v1_vix_validation_report.md`）——全歷史與 GFC 壓力測試兩組
獨立比較都顯示真實改善，且通過同曝險控制組檢驗。仍未接進 `scripts/strategy_v1_positions.py`
/ `scripts/paper_track_v1.py`，接線與否是獨立的後續決定，不在這次驗證範圍內。

---

#### B. 集中度上限檢查
防止單一標的或板塊過度集中：

| 限制 | 門檻 | 說明 |
|------|------|------|
| 單一標的 | 5% | 防止單一公司風險 |
| 單一桶 | 35% | 防止板塊集中（目標 25%，允許 10pp 漂移） |

**使用**：
```python
from python.portfolio.risk_controls import check_concentration

result = check_concentration(
    holdings, prices, bucket_of,
    max_single_name_pct=0.05,
    max_bucket_pct=0.35
)

if not result["within_limits"]:
    print("違規：", result["violators"])
    # e.g. [('NVDA', 0.12, 0.05, 'single_name')]
    #      → NVDA 佔 12%，超過 5% 上限
```

---

#### C. 限價單邏輯（省 spread）
月度再平衡不趕時間，可用限價單省 1-2 bps：

**價格計算**：
```python
from python.portfolio.risk_controls import limit_order_price

# aggression = 0.5 → 坐在 bid/ask 中間
buy_limit = limit_order_price("buy", mid_price=100.0, 
                               spread_bps=4.0, aggression=0.5)
# → $99.99（省 0.5 bps）

# aggression = 1.0 → 等同市價單
```

**成交模擬**：
```python
from python.portfolio.risk_controls import simulate_limit_fills

# 給定一天的 high/low，判斷限價單是否會成交
result = simulate_limit_fills(orders, price_range)
# → {filled, unfilled, fill_rate}
```

**估計節省**：
- 月度再平衡 120 檔 × 1.5 bps = ~$15 per $100k
- 年度節省 ~$180（約 0.18% / 年）

---

## ✅ 任務 3：V2 策略研究路線圖

### 交付文件
- **`docs/v2_research_plan.md`** - 完整研究計畫（240 行）

### 核心內容

#### 為什麼需要 V2？
- V1 是單一策略，風險集中
- V2 與 V1 相關性低（< 0.5）→ 組合多樣化
- 目標：提高 Sharpe，降低回撤

#### 候選方向：Cross-sectional Momentum
- **Alpha 來源**：相對強弱（買漲幅前 20%，賣跌幅後 20%）
- **與 V1 的差異**：
  - V1 = 時間序列（波動率）
  - V2 = 橫截面（排名）
- **為什麼之前失敗**：120 檔 universe 太小，signal 不夠強
- **解決方案**：擴展到 500 檔（S&P 500）

#### 實施路徑

**選項 A：快速驗證（2-3 天）**
1. 補抓 380 檔 S&P 500 成份股
2. 實施 12-1 momentum (long-only)
3. 回測 2018-2026
4. **判斷點**：gross Sharpe < 1.0 → 放棄

**選項 B：去存活偏誤（2-3 週）**
5. 構建 Point-in-Time universe 數據管道
6. 處理退市/併購
7. 重新回測，確認偏誤折扣
8. 如果通過 → 紙上交易 3-6 個月

#### 成本估計
```
月度再平衡 500 檔 × 40% turnover × $1.08/order = $216/月
年度 = $2,592（在 $100k 帳上 = 2.6%/年）

所以 V2 的 gross CAGR 必須 > 15% 才划算
（對應 gross Sharpe > 1.0）
```

#### 備選方向（如果 momentum 失敗）
1. Carry Strategies（期貨溢價）
2. Quality + Value（基本面因子）
3. Mean Reversion（超買超賣）

---

## 文件結構總覽

```
us-stock-trading/
├── scripts/
│   ├── report_v1_monthly.py         ← 新增（月度報告）
│   ├── run_v1_vix_study.py          ← 新增（VIX 回測；2026-09-17 修正兩個 bug 後才第一次真正跑完）
│   └── run_v1_vix_gfc_stress.py     ← 2026-09-17 新增（GFC 壓力測試）
├── python/portfolio/
│   └── risk_controls.py             ← 新增（風控模組）
├── tests/
│   └── test_risk_controls.py        ← 新增（風控測試）
├── reports/
│   ├── v1_2026_09.md                ← 新增（9 月報告）
│   └── v1_2026_09_equity_curve.png  ← 新增（權益曲線）
└── docs/
    ├── v2_research_plan.md          ← 新增（V2 研究計畫）
    └── 2026_09_10_deliverables.md   ← 本文件
```

---

## 下一步建議

### 短期（1-2 週）
1. ✅ **讓 V1 跑**：每天自動更新，累積到 90 日
2. ⏳ **每月生成報告**：手動或排程，監控表現
3. ⏳ **準備 V2 數據**：慢慢補抓 S&P 500 標的

### 中期（1-3 個月）
4. ⏳ **V1 驗證**：90 日後，對比回測預期
5. ⏳ **V2 概念驗證**：500 檔 momentum 回測
6. ⏳ **決定是否整合 VIX overlay**：2026-09-17 驗證已顯示真實改善（見上方訂正框），
   接不接線是獨立的上線決定，需另外規劃如何接進 `strategy_v1_positions.py`/`paper_track_v1.py`

### 長期（3+ 個月）
7. ⏳ **V2 上線**（如果通過）：紙上交易 3-6 個月
8. ⏳ **V1 + V2 組合**：測試多樣化效果
9. ⏳ **真實資金**：如果紙上交易證明有效

---

## 關鍵指標追蹤

| 策略 | 當前狀態 | 下一個里程碑 | 日期 |
|------|---------|-------------|------|
| V1 | 紙上交易 10 日 | 累積到 90 日 | 2026-11-30 |
| V2 | 研究計畫完成 | 500 檔回測 | TBD |
| 組合 | N/A | V2 通過後 | TBD |

---

## 總結

今天建立了三個核心基礎設施：
1. **自動化報告** → 持續監控 V1 表現
2. **風險管理模組** → 提升 V1 執行品質
3. **V2 研究路線圖** → 明確下一步方向

**現在該做的**：讓 V1 證明自己（90 天），同時慢慢準備 V2。

---

_2026-09-10 完成_
