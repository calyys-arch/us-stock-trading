# Strategy V2 研究計畫

## 目標

開發與 V1 相關性低（< 0.5）的第二條策略，用於組合多樣化。

## 候選方向：Cross-sectional Momentum

### 為什麼選這個？

1. **Alpha 來源不同**：
   - V1 = 波動率目標 + 多資產配置（時間序列特徵）
   - V2 = 相對強弱（橫截面特徵）
   
2. **已有初步測試**：
   - 在 120 檔 universe 上測過，無邊際
   - 診斷結果：universe 太小，cross-sectional signal 不夠強

3. **文獻支持**：
   - 12-1 momentum 是學術界最穩健的因子之一
   - 在大 universe（500+ 檔）上效果明顯

### 問題：為什麼 120 檔不夠？

Cross-sectional momentum 需要「贏家」和「輸家」的分化足夠明顯：
- 120 檔中，前 20% 和後 20% 的報酬差可能只有 5-10%
- 500 檔中，這個差會擴大到 20-30%（尾部效應）
- 更大的 universe 讓 signal 更穩定（不會被單一板塊主導）

## 實施路徑

### Step 1：擴展 Universe（數據工程）

#### 選項 A：使用現有 S&P 500 數據（快速但有偏誤）

**做法**：
```python
# 已有的 120 檔 + 再抓 380 檔
# data/history 已有部分 S&P 500 成份股
# 補齊到 500 檔
```

**優點**：
- 快（只需補抓價格）
- 可立即測試 signal 是否在大 universe 有效

**缺點**：
- **存活偏誤**：2026 年的 500 檔 ≠ 2016-2026 各年的 500 檔
- 會高估真實表現（排除了失敗的公司）

**適用場景**：概念驗證（proof of concept）

---

#### 選項 B：Point-in-Time (PIT) Universe（正確但費時）

**做法**：
1. 取得每年 S&P 500 成份股變動（已有部分在 `data/history_pit2016/`）
2. 對每個回測日期，只使用「那一天真實存在於指數中」的標的
3. 處理退市/併購：
   - 退市 → 最後一筆價格視為清算價
   - 併購 → 用併購價（通常是溢價）

**優點**：
- 無存活偏誤
- 回測結果可信
- 這是機構標準做法

**缺點**：
- 需要構建完整的 PIT 數據管道
- yfinance 不提供退市股票的部分歷史（已確認）
- 需要付費數據源（如 Sharadar、Alpha Vantage、Norgate）

**適用場景**：正式策略（如果 V2 測試通過）

---

### Step 2：實施 12-1 Momentum

**邏輯**：
```python
# 每月月初
past_12m_return = price[-252:-21] / price[-252] - 1  # 排除最近 1 個月
ranks = past_12m_return.rank()

# Long top 20%, short bottom 20%
long = ranks > 0.8
short = ranks < 0.2

# Equal weight within each side
long_weight = long / long.sum() * 0.5   # 50% long
short_weight = short / short.sum() * -0.5  # 50% short
```

**關鍵設定**：
- 月度再平衡（不是日度，避免 overtrading）
- 排除最近 1 個月（避免短期反轉）
- Long-short（市場中性）或 Long-only（方向性）？

**建議先測試 Long-only**，原因：
1. 做空美股在零售端成本高（借券費 + 短倉利息）
2. Long-only 可直接與 V1 組合（都是多頭方向）
3. 如果 long-only 有效，再考慮 long-short

---

### Step 3：回測與評估

**回測窗口**：2018-2026（與 V1 對齊）

**評估指標**：
| 指標 | 門檻 | 說明 |
|------|------|------|
| Sharpe ratio | > 0.5 | V2 自身要有正期望 |
| Correlation with V1 | < 0.5 | 確保多樣化效果 |
| Max drawdown | < -30% | 單獨拿出來也能接受的風險 |
| 組合 Sharpe | > V1 alone | 組合後要更好 |

**組合測試**：
```python
# 50% V1 + 50% V2
combined = 0.5 * v1_returns + 0.5 * v2_returns

# 期望：
# - Sharpe(combined) > max(Sharpe(V1), Sharpe(V2))
# - DD(combined) < max(DD(V1), DD(V2))
```

---

### Step 4：成本模型

**V2 的交易頻率**：
- 月度再平衡 = 每月交易 1 次 × 500 檔 × 40% turnover = 200 筆/月
- 年度 = 2,400 筆訂單

**估計成本**：
```
Per-trade commission: $1.00 (IBKR minimum)
Spread: 4 bps * $200 avg order = $0.08
Total per order: ~$1.08

Annual cost: 2,400 * $1.08 = $2,592
On $100k book: 2.6% / year
```

**這是致命傷嗎？**
- 如果 V2 gross CAGR = 15%，net = 12.4%（還行）
- 如果 V2 gross CAGR = 8%，net = 5.4%（不值得）

所以 V2 的 gross Sharpe 必須 **> 1.0** 才划算（遠高於 V1 的 0.82）。

---

## 時間估計

| 任務 | 選項 A（快速） | 選項 B（PIT） |
|------|---------------|--------------|
| 擴展 universe | 2-4 小時 | 1-2 週 |
| 實施策略邏輯 | 1 天 | 1 天 |
| 回測與評估 | 1 天 | 1 天 |
| 成本模型 | 4 小時 | 4 小時 |
| **總計** | **2-3 天** | **2-3 週** |

---

## 建議執行順序

### 階段 1：概念驗證（選項 A）
1. 補抓 380 檔 S&P 500 成份股到 `data/history/`
2. 實施 12-1 momentum (long-only)
3. 回測 2018-2026，測 Sharpe 和與 V1 的相關性
4. **判斷點**：如果 gross Sharpe < 1.0 → 放棄這條路

### 階段 2：去偏誤（選項 B，如果階段 1 通過）
5. 構建 PIT universe 數據管道
6. 重新回測，確認存活偏誤折扣
7. 如果 net Sharpe > 0.5 → 紙上交易 3-6 個月

---

## 風險與備選方案

### 如果 Cross-sectional Momentum 不行？

**備選 V2 方向**：

1. **Carry Strategies（期貨溢價）**
   - Alpha 來源：期貨曲線斜率
   - 數據需求：期貨連續合約價格
   - 相關性：與 V1 低（完全不同市場）
   
2. **Quality + Value（基本面因子）**
   - Alpha 來源：ROE、P/E、負債率
   - 數據需求：季報財務數據
   - 相關性：與 V1 低（基本面 vs 技術面）
   
3. **Mean Reversion（超買超賣）**
   - Alpha 來源：短期過度反應
   - 數據需求：現有價格數據即可
   - 風險：容易過擬合

**優先級**：如果 momentum 失敗，下一個測 **carry**（因為數據較容易取得）。

---

## 產出清單

完成 V2 研究後，你會有：

- [ ] `scripts/run_v2_momentum.py` - V2 回測腳本
- [ ] `backtests/v2_momentum_study.json` - 回測結果
- [ ] `python/portfolio/v2_momentum.py` - V2 策略邏輯
- [ ] `scripts/v2_positions.py` - V2 部位表生成器
- [ ] `data/paper_v2/journal.jsonl` - V2 紙上帳本（如果通過）
- [ ] `reports/v1_v2_correlation_study.md` - V1 vs V2 相關性分析

---

## 下一步

**立即可做**：
```bash
# 1. 補抓 S&P 500 標的到 500 檔
.venv/bin/python scripts/expand_universe_to_sp500.py --fetch

# 2. 實施並回測 V2
.venv/bin/python scripts/run_v2_momentum.py --wfo --save

# 3. 評估組合效果
.venv/bin/python scripts/compare_v1_v2_portfolio.py
```

**如果時間有限**：
- 先讓 V1 跑滿 90 天
- 同時慢慢準備 PIT universe
- 3 個月後如果 V1 證明有效，再全力投入 V2

---

_2026-09-10 建立_
