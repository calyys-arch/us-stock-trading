"""
Ad-hoc "pick N names to trade today" report — reuses the REAL
CrossSectionalMeanReversionStrategy (python/core/strategies/xsection_mean_reversion.py)
and the REAL configs/strategy.yaml parameters, so the ranking below is not a
separate ad-hoc heuristic; it is exactly Chan's eq. 3.7 signal the live system
would generate for today, run against a liquid large-cap subset of the
point-in-time S&P 500 universe (full 500-name yfinance pull is rate-limited
and slow for an interactive check, so this uses --limit names by default).

Universe-builder exclusions (PortfolioStrategy.evaluate()'s docstring is
explicit that this is the universe builder's job, not the strategy's): any
symbol reporting earnings today (python/interfaces/finnhub_calendar.py) or
carrying a same-day company-specific headline (python/interfaces/finnhub_news.py)
is dropped from `universe` before the strategy ever sees it — both fail safe
(no exclusions applied) if FINNHUB_API_KEY is unset. Today's general market
headlines are printed for context only; there is no principled threshold for
turning "N general-news items" into an exclusion (see finnhub_news.py).

Usage:
    .venv/bin/python scripts/pick_10.py --top 10 --limit 120
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from python.data.sp500_universe import fetch_current_constituents
from python.core.strategies.xsection_mean_reversion import CrossSectionalMeanReversionStrategy
from python.interfaces.finnhub_calendar import FinnhubEarningsCalendar
from python.interfaces.finnhub_news import FinnhubNewsSignal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--limit", type=int, default=60, help="cap universe size for a fast interactive pull")
    ap.add_argument("--lookback-calendar-days", type=int, default=10)
    ap.add_argument("--no-news-filter", action="store_true",
                     help="skip the earnings/company-news universe exclusion (debugging only)")
    args = ap.parse_args()

    with open("configs/strategy.yaml", encoding="utf-8") as f:
        strat_cfg = yaml.safe_load(f)["xsection_mean_reversion"]

    earnings_calendar = FinnhubEarningsCalendar()
    news_signal = FinnhubNewsSignal()

    headlines = news_signal.general_market_headlines_today()
    if headlines:
        print(f"=== Today's general market headlines ({len(headlines)}) — context only, not a trading filter ===")
        for h in headlines[:8]:
            print(f"  [{h['source']}] {h['headline']}")
        print()

    print("Fetching current S&P 500 constituent list (Wikipedia)...")
    constituents = fetch_current_constituents()
    sector_map = dict(zip(constituents["symbol"], constituents["sector"]))
    all_symbols = sorted(sector_map)
    # Evenly-spaced sample across the full alphabetically-sorted list rather
    # than the first N — avoids biasing the interactive sample toward
    # early-alphabet names/sectors (a straight [:limit] slice landed almost
    # entirely on A/B, over-weighting Info Tech and Health Care).
    if args.limit < len(all_symbols):
        step = len(all_symbols) / args.limit
        symbols = sorted({all_symbols[int(i * step)] for i in range(args.limit)})
    else:
        symbols = all_symbols
    print(f"Universe: {len(symbols)} symbols (evenly sampled from the full {len(all_symbols)}-name universe)")

    print("Downloading recent daily bars (bulk yfinance call)...")
    # threads=True observed to hang indefinitely in this environment (likely a
    # curl_cffi/session thread-safety issue under the sandboxed network path);
    # sequential requests are slower per-symbol but reliably complete.
    raw = yf.download(
        symbols, period=f"{args.lookback_calendar_days}d",
        auto_adjust=True, progress=False, group_by="ticker", threads=False,
    )

    frames = []
    for sym in symbols:
        try:
            sub = raw[sym][["Close"]].rename(columns={"Close": "close"})
        except (KeyError, TypeError):
            continue
        if sub.empty:
            continue
        sub = sub.copy()
        # yfinance's bulk multi-ticker endpoint occasionally drops a single
        # day's print for a subset of tickers (observed: ~46/60 names missing
        # only one specific date) even though the same date is fine via a
        # single-ticker call. Forward-fill isolated single-day gaps rather
        # than dropping the whole row, which would otherwise exclude most of
        # the universe from the return calc just because they hit that one
        # glitched date.
        sub["close"] = sub["close"].ffill(limit=1)
        sub = sub.dropna()
        sub["code"] = sym
        sub.index.name = "date"
        frames.append(sub.reset_index().set_index(["date", "code"]))

    if not frames:
        print("No data fetched — aborting.")
        sys.exit(1)

    panel = pd.concat(frames).sort_index()
    as_of_date = panel.index.get_level_values(0).max()
    loaded_symbols = sorted(panel.index.get_level_values(1).unique())
    print(f"Loaded {len(loaded_symbols)}/{len(symbols)} symbols, most recent bar: {as_of_date.date()}")

    # Universe-builder exclusions — see module docstring. Deliberately done
    # HERE (before strategy.evaluate()), never inside the strategy itself,
    # per PortfolioStrategy.evaluate()'s docstring contract.
    trade_universe = loaded_symbols
    if not args.no_news_filter:
        excluded: dict[str, str] = {}
        for code in loaded_symbols:
            if earnings_calendar.is_earnings_today(code):
                excluded[code] = "earnings today"
            elif news_signal.has_company_news_today(code):
                excluded[code] = "company news today"
        if excluded:
            trade_universe = [c for c in loaded_symbols if c not in excluded]
            print(f"Excluding {len(excluded)} name(s) with earnings/news today: "
                  + ", ".join(f"{c} ({reason})" for c, reason in excluded.items()))

    strategy = CrossSectionalMeanReversionStrategy(
        lookback_days=strat_cfg["lookback_days"],
        gross_leverage_target=strat_cfg["gross_leverage_target"],
        min_universe_size=strat_cfg["min_universe_size"],
    )

    target = strategy.evaluate(panel, as_of=as_of_date.to_pydatetime(), universe=trade_universe)

    if not target.weights:
        print(f"Strategy produced no signal today: {target.metadata}")
        sys.exit(0)

    ranked = sorted(target.weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[: args.top]

    print(f"\n=== Top {len(ranked)} names by |signal weight| — {target.strategy}, as of {as_of_date.date()} ===")
    print(f"(n_names in cross-section: {target.metadata['n_names']}, "
          f"cross-sectional mean return: {target.metadata['cross_sectional_mean_return']:.4%})\n")
    print(f"{'Symbol':<8}{'Sector':<28}{'Side':<6}{'Weight':>10}")
    for code, w in ranked:
        side = "LONG" if w > 0 else "SHORT"
        sector = sector_map.get(code, "?")[:26]
        print(f"{code:<8}{sector:<28}{side:<6}{w:>+10.4f}")

    print(
        "\nInterpretation (Chan eq. 3.7): LONG names underperformed the "
        "cross-sectional average yesterday and are expected to partially "
        "revert upward today; SHORT names outperformed and are expected to "
        "partially revert downward. This is an intraday-only signal — "
        "positions are meant to flatten by the close, per "
        "configs/strategy.yaml (allow_overnight: false)."
    )


if __name__ == "__main__":
    main()
