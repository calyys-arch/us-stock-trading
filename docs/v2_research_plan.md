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

## 六、Phase 0 結果（2026-09-11 執行）

```bash
.venv/bin/python scripts/run_v2_momentum_phase0.py --universe both
```

（注意：檔名是 `run_v2_momentum_phase0.py`，不是 `run_v2_momentum.py`——後者是 2026-09-10 就已存在、`scripts/compare_v1_v2.py` 還在依賴的另一支腳本，見下方「與既有 2026-09-10 結果的關係」。）

| Universe | N | 月數 | Gross Sharpe | Net Sharpe | 最大回撤 | Top-bottom 價差 t-stat | 判定 |
|---|---|---|---|---|---|---|---|
| 中型股（`data/history_altuni_midcap_full`，71 檔）| 71 | 103 | 0.53 | 0.51 | -26.5% | **-0.60** | **NO-GO** |
| 大型股（`data/history` 可用長歷史,57 檔）| 57 | 115 | 0.94 | 0.92 | -28.6% | **0.37** | **NO-GO** |

**機械式套用第五節的證偽規則,兩個 universe 都是 NO-GO**——但關鍵不是成本（月均成本只有 2-3bps,遠低於警戒),而是 **top quintile 跟 bottom quintile 的差距在統計上跟零沒有差異**（|t|<2）。換句話說：這兩個回測看起來的正 Sharpe，比較可能是「這段時間剛好整體在漲」（beta），不是「動能訊號真的能挑出贏家」（alpha）。

**誠實的但書**：57-71 檔的橫斷面樣本數偏小，這個顯著性檢定的統計檢定力（power）不高——文獻做這類檢定通常用幾百檔的橫斷面。所以「NO-GO」比較準確的講法是「這個小樣本沒有偵測到訊號」,而不是「訊號在更大 universe 上也一定沒用」的鐵證。這跟原始計畫設想的（用大 universe 才能看出分化）恰好呼應。

### 與既有 2026-09-10 結果的關係（`docs/v2_momentum_results.md`）

寫這支腳本時才發現，2026-09-10 已經有人做過一次 V2 動能測試，結果是 Gross Sharpe 0.96、CAGR 24.3%,看起來很不錯——但那次測試：
1. Universe 直接用 V1 的 120 檔,**混合了個股跟債券/大宗商品/國際股市 ETF** 一起做橫斷面排名（例如 GLD 這種可能單純因為資產類別本身在噴,而不是「動能」訊號真的在挑股票）。
2. **沒有做 vol-scaling**——這解釋了它 -40.4% 的最大回撤跟 26.2% 的實現波動,遠比今天這次（vol-scaled 後 -26~-29%）大。
3. **從未檢查 winner 是否真的顯著贏過 loser**——這正是今天新增、也是兩次測試中最關鍵的差異。

那份報告自己也誠實地寫了「universe 太小，可能高估了真實效果」，並建議「先讓 V1 證明自己，再考慮 V2」。今天用更嚴謹的方法（單純個股、有 vol-scaling、有顯著性檢定）重測,結論比原本更保守——不是推翻原結論，而是用不同方法得到一致但更謹慎的答案。

### 追加測試（2026-09-11,回應「樣本太小、檢定力不足」的質疑）

把中型股（71 檔）跟大型股（57 檔）兩個 universe 直接合併（去重後 127 檔,只有 1 檔重複）重跑一次,直接檢驗「是不是因為樣本太小才測不出訊號」：

```bash
.venv/bin/python scripts/run_v2_momentum_phase0.py --universe combined
```

| Universe | N | Net Sharpe | Top-bottom 價差 t-stat |
|---|---|---|---|
| 中型股（單獨） | 71 | 0.51 | -0.60 |
| 大型股（單獨） | 57 | 0.92 | 0.37 |
| **合併** | **127** | **0.79** | **0.16** |

**結果比單獨測試更明確地指向零**，不是更明確地指向有訊號。合併後價差均值只剩 0.10%/月（遠小於任一單獨 universe 的 |0.34%|~|0.36%|），t-stat 更接近 0。如果真的是「樣本太小蓋住了真訊號」，合併後應該讓訊號變清楚（往同一個方向收斂）；實際上兩個子樣本的雜訊方向相反,合併後互相抵消，這是「這裡真的沒有訊號」比「樣本不夠大」更合理的解釋。

### 建議下一步

兩個獨立方法都指向同一個方向：**現階段不值得投入 Phase 1/2 的資料工程（S&P 400 完整成份股 + PIT pipeline）**。比較合理的選項：
1. 採納 2026-09-10 報告原本的建議——先讓 V1 累積更多真實紙上交易數據，V2 動能方向擱置
2. 如果仍想繼續動能方向，下一步應該是解決「樣本數太小、統計檢定力不足」這個問題本身（抓一個真正幾百檔的 universe），而不是在現有 57-71 檔上調參數

## 七、Option D — 完全不同訊號家族的 Phase 0（2026-09-11，回應「換一個機制」的要求）

在確認動能（合併後也收斂到零）之後，使用者要求不要再測「同一個機制、換個 universe」，改測機制上真正不同的訊號家族。先盤點了 repo 裡的另類資料基礎設施，發現大多是「程式碼殼、沒有真實歷史資料」：

| 候選 | 基礎設施 | 實際歷史資料 | 可行性 |
|---|---|---|---|
| PEAD（財報驚喜） | `python/interfaces/finnhub_calendar.py` + `data/calendar/` | 2018-2025 全部 0 筆，2026 年 1500 筆但 EPS 欄位全 `None` | 不可行 |
| GEX / 選擇權莊家倉位 | `python/microstructure/gex.py` | 無歷史選擇權鏈快取 | 不可行 |
| 新聞情緒 | `data/news/{SYM}/*.json` | 只有 2025-08~2025-12（5 個月） | 不可行 |
| **8-K 申報事件** | `data/filings/8k/{SYM}.json` | **20 檔真實回溯至 2018-01-01** | **可行，唯一現成可測的** |

實作 `scripts/run_v2_8k_drift_phase0.py`：用 20 檔（跟微結構訊號同一批大型股/半導體名單）的 item 2.02（財報結果）8-K 申報，以申報當天的價格反應幅度當「意外程度」代理（沒有真實 EPS 驚喜數字），檢驗反應後 10 個交易日是否有延續漂移（PEAD 的核心假設：正意外持續漂移、負意外持續下跌——跟已死亡的 `xsection_mean_reversion` 的「反轉」假設方向相反）。

**結果（554 個事件，184 top / 184 bottom，2019-01-03 至 2026-08-04）：**
```
GROSS: Sharpe 0.43, CAGR 0.6%, MaxDD -1.8%
NET:   Sharpe 0.33, CAGR 0.5%, MaxDD -1.9%
Top-bottom 月度漂移價差: 均值 0.921%, t-stat 0.84（66 個月兩邊都有事件）
VERDICT: NO-GO — spread t-stat 0.84（|t| < 2）
```

**跟動能測試的差異值得記錄**：價差均值方向正確且經濟量級不小（0.92%/月，遠大於動能任一 arm 的 |0.34%|~|0.36%|），t-stat 0.84 也比動能合併測試的 0.16 更靠近顯著——**這比動能的「收斂到零」更像是「方向可能對、但樣本（僅 20 檔 x 8 年 x 每年約 4 次財報 ≈ 554 個事件）還是不夠大」**，跟動能測試「越合併越趨零」的證據型態不同。但目前 8-K 歷史資料就只覆蓋這 20 檔，沒有便宜的方法擴大樣本（需要新的資料抓取工程才能加更多symbol的 8-K 歷史，跟 Phase 1 的成本量級相近）。

**結論：Phase 0 NO-GO，但不是「死透」的 NO-GO**——如果之後真的要投入資料工程擴大 universe，8-K 事件漂移排在動能之前，因為它的訊號方向未被現有測試證偽，只是還沒被證實。目前建議跟動能一樣：先擱置，優先看 V1 的紙上交易紀錄。

_2026-08-25 建立，2026-09-11 依 2026 年動能因子文獻複查修正 + Phase 0 實測結果 + Option D（8-K 事件漂移）Phase 0 結果_
