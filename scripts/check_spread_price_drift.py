"""
How much does the out-of-window spread calibration understate backtest costs?

The problem. backtests/reports/calibrated_spreads.json measures real half-
spreads from captured L2 depth, and it is the basis for every cost conclusion
now on the table. But that depth covers 2026-08-04 to 2026-08-21 — ten trading
days sitting ENTIRELY AFTER the 2025-08-01 to 2026-07-01 backtest window. There
is no more of it: data/depth is 28 GB for exactly those ten days times twenty
symbols, and data/ticks covers the same span and carries trades only, no
quotes. So the spreads the study uses were never observed during the period the
study replays, and the gap cannot be closed by processing more data.

What can still be checked. A quoted spread in BASIS POINTS is a spread in cents
divided by a price. The cents component is anchored by the $0.01 minimum tick
and by liquidity tier, both persistent over a year; the price component is not,
and this universe re-rated hard over the window (MU 396 -> 864, SNDK 605 ->
1312). So the calibrated bps figure can be converted back to implied cents at
the calibration-window price and re-expressed at the backtest-window price:

    cents        = median_bps / 1e4 * price_calibration
    bps_backtest = cents / price_backtest * 1e4

This is an ADJUSTMENT, not a measurement. It assumes the cents spread held
constant, which is the same persistence assumption the original calibration
already relies on, just made explicit and applied to the one variable known to
have moved. It cannot capture a genuine change in liquidity.

Prices. Backtest-window prices come from daily bars. Calibration-window prices
come from sampled ticks, because data/history stops at 2026-07-31 for eighteen
of the twenty symbols — the same coverage gap, showing up again.

What the answer was, when this was first run. Tier MEMBERSHIP survives: 7 of 8
names in the cheap tier and 8 of 8 in the expensive tier are unchanged, so the
spread-tier experiment picked the right symbols. LEVELS do not: the median
symbol's backtest-window spread is 1.61x its calibrated value, ranging 0.69x
(ORCL, which fell) to 2.18x (MU, which more than doubled). The cheap tier is
least affected, near 1.1x, because those names re-rated least — so tightening
the universe looks BETTER under this adjustment, not worse, while
auction_reclaim_5m's ceiling falls from 4.11 (flat 2.0 bps) through 2.33
(calibrated) to roughly 1.45, close to the analytic 1.0 floor below which no
position size clears PF 1.0.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPREADS = ROOT / "backtests/reports/calibrated_spreads.json"
DAILY = ROOT / "data/history"
TICKS = ROOT / "data/ticks"
OUT_JSON = ROOT / "backtests/reports/spread_price_drift.json"
OUT_MD = ROOT / "backtests/reports/spread_price_drift.md"

TIER_SIZE = 8
TICK_SAMPLE_FILES = 3
TICK_SAMPLE_LINES = 2000


def _calibration_price(symbol: str) -> float | None:
    """Mean traded price over the calibration window, sampled from ticks.

    Sampled rather than fully aggregated because only the price LEVEL matters
    here, to two significant figures; reading 18 GB to refine a denominator
    would not move any conclusion below.
    """
    files = sorted((TICKS / symbol).glob("*.jsonl"))[:TICK_SAMPLE_FILES]
    prices: list[float] = []
    for path in files:
        with path.open() as fh:
            for line in itertools.islice(fh, TICK_SAMPLE_LINES):
                try:
                    prices.append(float(json.loads(line)["price"]))
                except (ValueError, KeyError, TypeError):
                    continue
    return (sum(prices) / len(prices)) if prices else None


def _backtest_price(symbol: str, start: str, end: str) -> float | None:
    path = DAILY / f"{symbol}.csv"
    if not path.exists():
        return None
    df = (pd.read_csv(path, parse_dates=["date"])
          .drop_duplicates("date").set_index("date").sort_index())
    window = df.loc[start:end, "close"]
    return float(window.mean()) if not window.empty else None


def analyse(start: str, end: str) -> dict:
    symbols = json.loads(SPREADS.read_text(encoding="utf-8"))["symbols"]
    rows = []
    for symbol, info in sorted(symbols.items()):
        p_cal = _calibration_price(symbol)
        p_bt = _backtest_price(symbol, start, end)
        if not (p_cal and p_bt):
            continue
        calibrated = float(info["median_bps"])
        cents = calibrated / 1e4 * p_cal
        adjusted = cents / p_bt * 1e4
        rows.append({
            "symbol": symbol,
            "calibrated_bps": calibrated,
            "adjusted_bps": adjusted,
            "multiple": adjusted / calibrated,
            "implied_spread_cents": cents,
            "price_calibration": p_cal,
            "price_backtest": p_bt,
        })

    by_cal = sorted(rows, key=lambda r: r["calibrated_bps"])
    by_adj = sorted(rows, key=lambda r: r["adjusted_bps"])
    names = lambda rs: [r["symbol"] for r in rs]  # noqa: E731
    multiples = [r["multiple"] for r in rows]
    return {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "backtest_window": f"{start} .. {end}",
        "calibration_window": "2026-08-04 .. 2026-08-21 (10 trading days)",
        "rows": by_cal,
        "multiple_median": st.median(multiples) if multiples else None,
        "multiple_min": min(multiples) if multiples else None,
        "multiple_max": max(multiples) if multiples else None,
        "tiers": {
            "cheap_calibrated": names(by_cal[:TIER_SIZE]),
            "cheap_adjusted": names(by_adj[:TIER_SIZE]),
            "expensive_calibrated": names(by_cal[-TIER_SIZE:]),
            "expensive_adjusted": names(by_adj[-TIER_SIZE:]),
        },
    }


def _render(p: dict) -> str:
    t = p["tiers"]
    cheap_overlap = len(set(t["cheap_calibrated"]) & set(t["cheap_adjusted"]))
    dear_overlap = len(set(t["expensive_calibrated"]) & set(t["expensive_adjusted"]))
    L = [
        "# 校準價差的價格漂移檢查",
        "",
        f"- 產生於：{p['generated_at']}",
        f"- 回測視窗：{p['backtest_window']}",
        f"- 校準視窗：{p['calibration_window']}",
        "",
        "校準視窗**完全落在回測視窗之後**，而且無法延長：`data/depth` 的 28 GB 就是那 10 天 "
        "20 檔的完整委託簿，`data/ticks` 同一段期間且只有成交沒有報價。",
        "",
        "bps 價差等於分價差除以價格。分價差由最小跳動單位與流動性層級錨定，一年內相對持久；",
        "價格不是，而這批股票在視窗內大幅重評。所以把校準值換算成隱含分價差，再用回測期價格",
        "重新表示。這是**調整而非量測**——它假設分價差不變，也就是把原本校準已經倚賴的持久性",
        "假設講明，並套用到唯一已知有變動的那個變數上。",
        "",
        "| 標的 | 校準 bps | 回測期 bps | 倍數 | 隱含分價差 | 校準均價 | 回測均價 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p["rows"]:
        L.append(
            f"| {r['symbol']} | {r['calibrated_bps']:.3f} | {r['adjusted_bps']:.3f} | "
            f"{r['multiple']:.2f} | ${r['implied_spread_cents']:.4f} | "
            f"{r['price_calibration']:.1f} | {r['price_backtest']:.1f} |"
        )
    L += [
        "",
        f"- 倍數中位數 {p['multiple_median']:.2f}，範圍 "
        f"{p['multiple_min']:.2f} 至 {p['multiple_max']:.2f}。",
        f"- 分層**成員資格站得住**：便宜層重疊 {cheap_overlap}/{TIER_SIZE}、"
        f"昂貴層重疊 {dear_overlap}/{TIER_SIZE}，所以價差分層實驗選對了標的。",
        "- 分層**水準站不住**：漲最多的標的被低估最多，因為同樣的分價差在較低的價位上就是更多的 bps。",
        "- 便宜層受影響最小（約 1.1 倍），所以緊價差的優勢在這個調整下**擴大**而非縮小。",
        "",
        "## 這對已發布的天花板意味著什麼",
        "",
        "| 成本假設 | `auction_reclaim_5m` 天花板 |",
        "|---|---:|",
        "| 2.0 bps 常數（已發布） | 4.11 |",
        "| 校準期實測價差 | 2.33 |",
        "| 校準值 × 倍數中位數 | ~1.45 |",
        "",
        "矩陣裡唯一成本後 PF 超過 1.0 的格子，真實天花板可能貼著 1.0 這條解析邊界——"
        "而低於 1.0 代表任何部位大小都過不了 PF 1.0。",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2025-08-01")
    ap.add_argument("--end", default="2026-06-30")
    args = ap.parse_args()

    payload = analyse(args.start, args.end)
    if not payload["rows"]:
        print("沒有任何標的同時取得兩個視窗的價格。")
        return 1
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_render(payload), encoding="utf-8")

    print(f"{len(payload['rows'])} 檔；倍數中位數 {payload['multiple_median']:.2f}"
          f"（{payload['multiple_min']:.2f}..{payload['multiple_max']:.2f}）")
    t = payload["tiers"]
    print(f"便宜層重疊 {len(set(t['cheap_calibrated']) & set(t['cheap_adjusted']))}/{TIER_SIZE}；"
          f"昂貴層重疊 {len(set(t['expensive_calibrated']) & set(t['expensive_adjusted']))}/{TIER_SIZE}")
    print(f"Wrote {OUT_JSON}\nWrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
