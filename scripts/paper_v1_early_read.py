"""
Early-read for Strategy V1's paper ledger — answers "is 11 days of noise
unusual" WITHOUT waiting for scripts/paper_track_v1.py --report's
MIN_DAYS_FOR_VERDICT = 250 threshold.

WHY 250 IS NOT WRONG, JUST BLUNT. It is a round-number floor ("roughly a
year"), not derived from anything about V1 specifically. No amount of
statistical cleverness shrinks the fundamental uncertainty of a Sharpe
estimate — its standard error scales like 1/sqrt(years of data), so 11 days
genuinely carries very little information about whether a Sharpe-0.91-style
edge is real. That is not fixable by a smarter test. What IS fixable: a flat
day-count threshold treats "-1.23% over 11 days" and "-1.23% over 11 days
during a historically unprecedented stretch for this exact strategy" as
identical, when they are not. This script replaces "is N >= 250?" with "how
does this window compare to every other N-day window this strategy has ever
had?", which is informative from day 1 and gets MORE informative as N grows,
rather than being silent below a cutoff and suddenly vocal above it.

METHOD. Reconstruct V1's full historical daily return path (2018 -> today)
using the exact same rule the paper ledger runs — bucket_weighted_gross,
volatility-targeted exposure, the rebalance band from
python/portfolio/strategy_v1.py — then look at EVERY overlapping N-day
window in that history (N = however many trading days the live paper
ledger has accumulated) and see where the live ledger's actual N-day
cumulative return falls in that empirical distribution.

DISCLOSED SIMPLIFICATIONS (acceptable for an early-warning read, NOT a
substitute for the costed paper ledger or the WFO promotion gates):
  - Turnover cost is a flat one-way bps charge on the traded exposure
    change (same convention scripts/run_vol_target_study.py uses), not the
    per-order IBKR/Futu commission schedule scripts/paper_track_v1.py
    actually charges. Cheap to compute over 2000+ days; the real ledger's
    own $-costed numbers are still the authoritative record.
  - Overlapping windows are NOT independent draws (adjacent windows share
    almost all their days), so "5th percentile" here is a descriptive
    goodness-of-fit statement ("how rare has this shape of stretch been"),
    not a p-value from an i.i.d. test. Still far more informative than a
    flat day-count gate, and gets MORE trustworthy as N grows relative to
    the ~2000-day history (small-N windows are the noisiest to interpret
    in EITHER direction, which is exactly the caveat this script prints).

Usage:
    .venv/bin/python scripts/paper_v1_early_read.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.portfolio.strategy_v1 import (  # noqa: E402
    band_exposure, bucket_weighted_gross, load_prices, load_universe,
)
from scripts.run_vol_target_study import (  # noqa: E402
    MAX_EXPOSURE, ONE_WAY_COST_BPS, REBALANCE_BAND, VOL_LOOKBACK,
)

JOURNAL = ROOT / "data/paper_v1/journal.jsonl"
TARGET_VOL = 0.15
TRADING_DAYS = 252


def reconstruct_history(target_vol: float = TARGET_VOL) -> pd.Series:
    """Full daily net-return path V1's rule would have produced over all
    cached history — same weighting/exposure/band logic the live paper
    ledger runs, approximated cost model (see module docstring)."""
    symbols, bucket_of = load_universe()
    panel = load_prices(symbols)
    gross, _ = bucket_weighted_gross(panel, bucket_of, "equal", VOL_LOOKBACK,
                                      REBALANCE_BAND, ONE_WAY_COST_BPS)
    gross = gross.dropna()

    applied = None
    net_rets = []
    dates = []
    for i in range(len(gross)):
        window = gross.iloc[max(0, i - VOL_LOOKBACK):i]
        if len(window) < VOL_LOOKBACK:
            continue
        realized = float(window.std(ddof=1) * np.sqrt(TRADING_DAYS))
        want = 0.0 if realized <= 0 else min(MAX_EXPOSURE, target_vol / realized)
        target, _ = band_exposure(want, applied, REBALANCE_BAND)
        if applied is None:
            applied = target
        turnover_cost = abs(target - applied) * (ONE_WAY_COST_BPS / 10_000.0)
        net_rets.append(applied * gross.iloc[i] - turnover_cost)
        applied = target
        dates.append(gross.index[i])
    return pd.Series(net_rets, index=dates)


def window_distribution(daily: pd.Series, n_days: int) -> pd.Series:
    """Cumulative return of every overlapping n_days-long window in `daily`."""
    cumret = (1.0 + daily).rolling(n_days).apply(np.prod, raw=True) - 1.0
    return cumret.dropna()


def main() -> None:
    if not JOURNAL.exists():
        print("找不到 data/paper_v1/journal.jsonl，先跑 paper_track_v1.py --init。")
        return
    journal = [json.loads(line) for line in JOURNAL.read_text().splitlines() if line.strip()]
    marks = [j for j in journal if j["action"] != "init"]
    n_days = len(marks)
    if n_days < 3:
        print(f"只有 {n_days} 個交易日，連早期讀數都還太少，晚點再跑。")
        return

    start_equity = float(journal[0]["equity"])
    live_cumret = float(marks[-1]["equity"]) / start_equity - 1.0

    print(f"重建 V1 全歷史每日報酬路徑（同樣的 bucket_weighted_gross + 波動目標 + "
          f"再平衡帶規則，簡化成本模型）...")
    daily = reconstruct_history()
    print(f"重建歷史：{daily.index[0].date()} ~ {daily.index[-1].date()}，"
          f"{len(daily)} 個交易日\n")

    dist = window_distribution(daily, n_days)
    if len(dist) < 30:
        print(f"歷史上只有 {len(dist)} 個 {n_days} 天視窗可比較，樣本太少，"
              f"這個早期讀數本身也不可靠。")
        return

    percentile = float((dist < live_cumret).mean() * 100)
    p5, p25, p50, p75, p95 = dist.quantile([0.05, 0.25, 0.5, 0.75, 0.95])

    print(f"目前紙上帳本：{n_days} 個交易日，累計報酬 {live_cumret:+.2%}")
    print(f"歷史上所有 {n_days} 天視窗（重疊，共 {len(dist)} 個）的累計報酬分布：")
    print(f"  5th 百分位: {p5:+.2%}   25th: {p25:+.2%}   中位數: {p50:+.2%}"
          f"   75th: {p75:+.2%}   95th 百分位: {p95:+.2%}")
    print(f"\n==> 目前這段落在歷史分布的第 {percentile:.0f} 百分位")

    if percentile <= 5:
        print("    低於歷史 5th 百分位 — 值得認真看待，即使天數還很少（但見下方警語）。")
    elif percentile >= 95:
        print("    高於歷史 95th 百分位 — 表現異常好，同樣值得留意是否合理（見下方警語）。")
    else:
        print("    落在歷史上典型的雜訊範圍內，跟過去任何一段隨機挑的同樣長度期間沒有明顯差異——")
        print("    這正是 --report 在 250 天門檻前拒絕下結論的原因：現在這個數字不算異常。")

    print(f"\n警語：這 {len(dist)} 個視窗互相重疊（不是獨立樣本），天數越少雜訊越大，"
          f"這是輔助早期監控用的描述性比較，不是正式的顯著性檢定，也不能取代 250 天"
          f"的正式否證門檻或完整的 WFO/Monte Carlo 驗證。")


if __name__ == "__main__":
    main()
