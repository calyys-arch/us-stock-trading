# Strategy V2 研究計畫 (v2 — 2026-09-11 修正版)

## 修訂記錄

| 版本 | 日期 | 改動 |
|---|---|---|
| v1 | 2026-08-25 | 初版：12-1 cross-sectional momentum，選項 A/B universe 擴充路徑 |
| v2（本文件）| 2026-09-11 | 基於 2026 年最新動能因子文獻的系統性複查，修正 v1 兩個未經驗證、且文獻上已知會出問題的假設：universe 選擇、缺少崩盤風控。新增：net-of-cost 硬門檻、Chan 五參數上限下的具體參數表、可快速證偽的 kill-test 順序。 |

**v1 沒有錯的部分（保留）**：long-only（不做 short leg）、月度再平衡、12-1 動能定義本身、"先概念驗證再做 PIT" 的分階段思路。這些跟文獻結論一致，不用改。

**v1 有問題的部分（本次修正）**：
1. 「選項 A」預設抓到 500 檔＝抓 S&P 500 這種大型股——但兩篇 2024 年的復現研究**剛好測的就是這個確切組合**（100-500 檔美國大型股，2000-2024），結論是樣本外（2015 年後）淨 Sharpe +0.02、5 個滾動窗口 0 個顯著，"the easily harvested version is gone"。這跟今天 `xsection_mean_reversion` 死在大型股上的死法幾乎一樣。
2. 完全沒有處理「動能崩盤」風險——雖然 long-only 已經避開了文獻裡最致命的那部分（short leg 在反轉時暴力回彈），但沒理由不用一個文獻驗證過、幾乎零成本的風控手段。
3. 沒有把「小型股的動能溢酬其實是交易成本的幻覺」這個 2024 年的發現算進 universe 篩選條件——這點沒處理的話，Phase 1 擴大 universe 時很可能會把太小、太貴的名字也加進來，做出一個好看的 gross 數字，然後在 net-of-cost 才發現全部被交易成本吃掉。

---

## 目標（不變）

開發與 V1 相關性低（< 0.5）的第二條策略，用於組合多樣化。V1 現況：development window（2018-2026，1948 天）Sharpe 1.08 / CAGR 15.8%；holdout（664 天）Sharpe 1.71 / CAGR 25.8%，兩者都過 drawdown gate。V2 的目的是「加一條真正不同的 alpha 來源」，不是取代 V1。

---

## 一、2026 年文獻複查結論

### 1.1 動能因子本身：沒死，但持續衰減

- *Journal of Portfolio Management*（2025）綜述：150 年跨市場資料下動能穩健，「against data mining and arbitrage concerns」——跟已被證明死掉的 1 日橫斷面反轉不是同一個等級的問題。
- 但 2025 年一篇用 Fama-French 8 因子（1963-2024）做的衰減建模發現：動能的 alpha 衰減完美符合雙曲線衰減模型（R²=0.65，優於線性/指數模型），且**2015 年後衰減明顯加速**，與動能 ETF/因子基金規模增長高度相關（ρ=-0.63）。樣本外模型甚至還高估了剩餘 alpha（預測 0.30，實際只剩 0.15）。
- **結論**：動能是一個「真實但持續在被套利掉」的效應，不是「已經死掉」的效應——這跟 `xsection_mean_reversion` 的 1 日反轉（在大型股上已被證明是負的，不是「不夠正」）在證據強度上完全不同等級,但也不能天真地假設 12-1 動能會有多好的邊際。

### 1.2 Universe 選擇：v1 計畫的「選項 A」踩到了文獻的坑

兩個 2024 年直接復現「12-1 動能 + 美股大型股」這個組合的研究：

| 研究 | Universe | 結果 |
|---|---|---|
| 100 檔美國大型股，2000-2024 | 大型股，long-short | 樣本外（2015-24）淨 Sharpe **+0.02**；5 個滾動窗口 Bonferroni 校正後 **0 個顯著**；breakeven cost **-8.8bps**（零成本都虧錢）；結論原文："the easily harvested version is gone" |
| Survivorship-bias-free S&P 500，2005-2024 | 大型股，long-short | Long leg 本身 +7.9%/年（還活著），但 long-short 組合淨報酬 **-2.79%/年**、最大回撤 **-81.2%**——幾乎全部虧損來自 short leg 在崩盤反彈（2009-04、2020-11）時的損失，FF5+UMD 迴歸 alpha = **-4.03%**（顯著為負） |

**這兩個結果分別對應到 v1 計畫的兩個決定**：

- ✅ v1 已經選「long-only」——文獻證實這完全正確：long leg 單獨測是 +7.9%/年，虧錢的是 short leg。Long-only 天生躲開了這篇研究裡最致命的部分。
- ❌ v1「選項 A」預設抓 500 檔大型股（S&P 500 主體）——這正是第一篇研究測的確切 universe，樣本外已被證明是 null result。**繼續照抄選項 A 大概率會複現今天 `xsection_mean_reversion` 同樣的死法，只是換一個因子名字。**

### 1.3 Universe 廣度的正確方向：中型股，不是小型股

一份 2024 年针對 CRSP 1990-2023 的分層研究（依市值分三層）給出了非常明確、可直接拿來設計篩選條件的數字：

| 市值層 | Gross 報酬/月 | Net 報酬/月（扣交易成本後）| 顯著性 |
|---|---|---|---|
| Large（>$10B） | 0.42% (t=2.8) | 0.31% (t=2.1) | 顯著但小 |
| **Mid（$500M–$10B）** | **0.78% (t=3.9)** | **0.38% (t=1.9)** | **邊際顯著（最佳淨值）** |
| Small（<$500M） | 1.34% (t=4.7) | **-0.12% (t=-0.4)** | **淨值為負，不顯著** |

原文結論：小型股動能的 gross 報酬看起來最漂亮（每月 1.34%），但换手率是大型股的 1.8 倍、價差是 4.2 倍寬，**扣成本後完全被吃光，breakeven 市值約 $480M**——「a pure illusion of gross returns」。

**這給了 V2 一個明確、可執行的 universe 篩選規則**：目標中型股（約 $500M–$10B 市值,或用流動性代理如 20 日均額美元成交量），**排除**兩端——太小的（交易成本吃光淨值，這是今天新學到的、直接可用的硬約束）和太集中在超大型股的（樣本外已被驗證是 null result，跟今天死掉的 `xsection_mean_reversion` 同一個坑）。

### 1.4 動能崩盤風控：有文獻驗證的做法，且幾乎零額外自由參數成本

- **Volatility-scaled momentum**（Barroso & Santa-Clara 2015; Daniel & Moskowitz 2016）：用動能因子自身的已實現波動率去反向縮放曝險（波動率上升時降低倉位）——這個波動率具有可預測性，能在崩盤發生「前」就先降槓桿。概念上與 V1 `strategy_v1.py` 已經在用的 vol-targeting 是同一族技巧，風格上一致，不是新發明。
- **Residual/idiosyncratic momentum**（Blitz et al. 2011/2020）：用「扣掉市場/風格因子曝險後」的殘差報酬排序，而非總報酬排序——實證上波動率只有傳統動能的一半，且沒有顯著犧牲報酬,長期也沒有反轉現象。2020-2021 年最新研究顯示 vol-scaled residual momentum 效果最穩定（US 年化 alpha 1.08-1.20%，Sharpe 到 0.68）。
- 兩篇比較研究都指出：**沒有單一方法「一定」最好**，但都比什麼都不做（raw total-return momentum, unscaled）好；dynamic vol-scaling 對市場反轉的敏感度較低。
- **重要但容易忽略的一點**：因為 V2 是 long-only，已經避開了文獻裡崩盤傷害最大的 short leg，所以崩盤風控在這裡的迫切性遠低於原始 long-short 動能文獻的語境——不需要為了風控把 Phase 1 搞得過度複雜。建議把 residual momentum（需要額外算因子暴露迴歸，複雜度較高）留到 Phase 2 才考慮，Phase 1 先用最簡單的 vol-scaling（一個乘數，一個 lookback，兩個參數就能做）。

### 1.5 擁擠度（Crowding）：對 V2 的啟示是「別選最熱門的名字」，不是「別做動能」

- 2026 年一篇研究發現「crowded momentum 的崩盤機率反而更低（0.38x, p=0.006）」——擁擠度預測的是尾部風險的方向，不是均值報酬本身，別誤讀成「擁擠=更好」。
- MSCI 的研究則發現：在同一個高動能十分位裡，**排除掉最擁擠的名字**（用一個擁擠度分數做約束，而非額外做多空）能改善結果——但這是一個「錦上添花」的優化，不是 Phase 1 的必要條件，且這種擁擠度分數在這個 repo 目前沒有現成的資料源可以算（需要持倉/資金流數據，yfinance 拿不到）,列為 Phase 2+ 才評估的方向，不列入 Phase 1 的參數表。

---

## 二、修正後的策略設計

### 2.1 訊號定義（Phase 1，維持簡單）

```python
# 每月月初再平衡
lookback_months = 12   # formation period（12-1 慣例，鎖定不掃）
skip_months = 1         # 排除最近 1 個月（避免短期反轉汙染訊號，鎖定不掃）
past_return = price[t - lookback_months] / price[t - skip_months] - 1

# Universe 篩選（新增，直接回應 1.3 節的成本懸崖發現）
# 用 20 日均額美元成交量做流動性代理（市值資料 yfinance 不穩定，
# ADV 代理更貼近「交易成本會不會吃掉淨值」這個真正關心的問題）
eligible = (adv_20d_usd >= MIN_ADV_USD) & (adv_20d_usd <= MAX_ADV_USD)
# MIN 篩掉小型股成本懸崖（1.3 節：<$500M 市值淨值為負）
# MAX 篩掉最擁擠的超大型股（1.2 節：純大型股樣本外是 null result）

ranks = past_return[eligible].rank(pct=True)
long = ranks > (1 - top_quantile)          # top_quantile 預設 0.2（quintile）

# 風險平價式的簡單波動率縮放（1.4 節，Barroso-Santa Clara 精神，非逐股）
book_vol_target = ...                       # 沿用 V1 的 vol-targeting 風格
exposure_scale = book_vol_target / realized_vol_trailing(momentum_book_returns, vol_lookback_days)
long_weight = (long / long.sum()) * exposure_scale
```

### 2.2 Chan 五參數上限盤點

| 參數 | 是否為自由參數 | 說明 |
|---|---|---|
| `lookback_months=12` | 否（鎖定） | 12-1 是文獻慣例值，不掃格，避免用回測結果反推出剛好賺錢的 lookback（過擬合） |
| `skip_months=1` | 否（鎖定） | 同上 |
| `top_quantile` | **是（1）** | 預設 0.2（quintile），可掃 {0.1, 0.2, 0.3} |
| `min_adv_usd` / `max_adv_usd` | **是（2）** | 定義中型股窗口，需要網格驗證窗口邊界對結果的敏感度（避免抓到剛好賺錢的窗口） |
| `vol_lookback_days` | **是（1）** | 波動率縮放的回看天數 |
| `book_vol_target` | **是（1）** | 目標年化波動率（可比照 V1 的 15%起跳） |

共 5 個自由參數，剛好貼齊 `param_guard.py` 的 `MAX_FREE_PARAMETERS=5` 上限，不留餘裕——如果 Phase 2 要加 residual momentum 或擁擠度過濾，必須先移除或鎖定掉其中一個現有參數（例如把 `top_quantile` 鎖定為 0.2 不再掃）。

### 2.3 Net-of-cost 是新的硬門檻，不是事後檢查

沿用 `python/core/fees_equity.py`（已有的 commission + SEC Section 31 + FINRA TAF + square-root market impact + 半價差模型）。基於 1.3 節的發現，**net Sharpe 門檻不再只是「> 0」**，而是：

- Gross Sharpe 必須顯著為正 **而且**
- Net Sharpe（用 `fees_equity.py` 算完整套成本後）也必須為正，門檻抓 > 0.5（沿用 v1 原本的邏輯：月度再平衡换手率不算太高，但中型股價差比大型股寬，不能只用大型股的成本假設）
- **如果 gross 顯著為正但 net 轉負或接近於零 → 直接 NO-GO，不做參數調整去「救」它**（`strategy_review_summary.md` §4.5 的教訓：grid search 救不回一個 net 為負的效應）

---

## 三、Universe 建構：分階段，先便宜的證偽測試

### Phase 0（可立即執行，~0.5 天）：用現有快取資料做第一輪 kill-test

`data/history/` 已經快取了 **592 檔**股票 2019-2026 的日線資料（比 v1 計畫設想的還多），且 `python/core/fees_equity.py` 的成本模型已經現成可用。**不需要新抓資料就能先跑一輪**：

1. 用現有 592 檔算 ADV（用快取的 volume×close 估計，非精確值，但足夠做第一輪篩選）
2. 套用中型股 ADV 窗口，排除掉最集中的大型股尾部（現有 592 檔可能仍以現存大型股居多，需要先看分布，若中型股樣本太少則此階段只能算 partial validation）
3. 跑 12-1 long-only + vol-scaling，gross 和 net Sharpe 都算出來
4. **判斷點（比 v1 原計畫更早、更嚴格）**：如果 net Sharpe ≤ 0 甚至只是 gross Sharpe 就不顯著 → 直接 NO-GO，不進入 Phase 1的資料擴充，省下抓資料跟建 PIT pipeline 的力氣

這一步的**目的是快速證偽**，不是為了拿到能上線的結果——現有 592 檔是存活偏誤資料（今天存在的股票的完整歷史），任何 Phase 0 的 GO 結果都只能當作「值得繼續」的訊號，不能當作「可以上線」的證據。

### Phase 1（如果 Phase 0 沒被直接證偽，~2-3 天）：正確定義中型股 universe

- 如果 Phase 0 顯示現有 592 檔裡中型股樣本太少，需要額外抓 S&P 400 MidCap 成份股（約 400 檔，跟現有 592 檔有部分重疊）
- 這階段仍是存活偏誤資料（選項 A 的精神），但至少 universe 的市值分佈是正確對準文獻建議的窗口，而不是像 v1 原計畫一樣預設抓「500 檔大型股」
- 判斷點同 v1 原計畫：net Sharpe ≤ 0.5 或跟 V1 相關性 > 0.5 → 放棄

### Phase 2（只有 Phase 1 通過才做，~1-2 週）：Point-in-time universe 去偏誤

- `data/history_pit2016/` 現況只有 **22 檔**、且不是真正的逐年成份股變動記錄，這個 PIT 基礎建設實質上不存在，需要從頭建（v1 計畫的「選項 B」原本的工作量估計，1-2 週，仍然成立）
- 這步驟只有在 Phase 1 的存活偏誤結果夠好、值得投入這個力氣時才做——這正是 Phase 0/1 存在的意義：不要一開始就直接跳進最貴的 PIT pipeline

---

## 四、評估管線（沿用既有基礎設施，不重新發明）

- WFO / Monte Carlo / Reality Check（Bonferroni）：直接沿用 `self_improve_loop.py` 已經驗證過的管線（這次 session 才剛在 `pairs_trading`/`xsection_mean_reversion` 上端到端測過，管線本身沒問題）
- 與 V1 的相關性檢查：`scripts/compare_v1_v2.py`（已存在,尚未跑過 V2 的真實數據）
- 成本模型：`python/core/fees_equity.py`（已存在，不需要重新建）
- 回測窗口：沿用 v1 計畫的 2018-2026，與 V1 對齊

---

## 五、什麼情況會讓我們快速放棄這個方向

明確寫下來，避免重蹈今天 pairs/xsection 那種「一路做到底才發現早就該停」的覆轍：

1. Phase 0 net Sharpe ≤ 0 → 立即停，不做任何參數搜尋救援
2. Phase 0/1 的 top-quintile 和 bottom-quintile 報酬差距在統計上不顯著（t < 2）→ 停,這代表訊號本身在這個 universe 沒有區分力,不是成本問題
3. 即使 Phase 1 通過，如果 WFO 通過率 < 50%（`xsection_mean_reversion` 今天是 27.8%，直接 NO-GO 的量級）→ 停
4. 與 V1 相關性 > 0.5 → 就算自身表現不錯也停，因為失去了組合多樣化的意義（這是 V2 存在的唯一理由）

---

## 六、下一步

需要決定：先跑 **Phase 0**（用現有 592 檔快取資料 + 現成成本模型,半天內可以有第一個 gross/net Sharpe 數字），確認這個方向在被 3 天/3 週的資料工程投入之前，值不值得繼續。

```bash
# Phase 0 大致執行方式（尚未寫成腳本）
.venv/bin/python scripts/run_v2_momentum.py --universe data/history --phase 0 --vol-scale --wfo
```

_2026-08-25 建立，2026-09-11 依 2026 年動能因子文獻複查修正_
