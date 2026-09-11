"""
Option D Phase 0 kill-test — 8-K filing-date event drift ("PEAD proxy").

Context: docs/v2_research_plan.md's momentum Phase 0 (2026-09-11, both
individually and combined into one 127-name cross-section) came up
NO-GO — the winner/loser spread converges toward zero as N grows rather
than sharpening, evidence the null is genuine rather than an
underpowered test. In the same conversation the user asked for a
GENUINELY DIFFERENT signal family (not another momentum/reversal
variant), specifically ruling out re-litigating the same mechanism with
more data.

Data-availability audit before writing this script (see chat, not
re-derived here) found most "alternative data" infra in this repo is a
code SHELL with no real historical backing:
  - data/calendar/earnings_*.json: 2018-2025 are ALL EMPTY; only 2026 has
    entries and epsActual/epsEstimate are None. No historical EPS
    surprise data exists -> classic PEAD (needs analyst surprise) is
    NOT feasible without a new paid data-acquisition project.
  - GEX (python/microstructure/gex.py): no historical options-chain
    cache exists at all -> not feasible.
  - data/news/{SYMBOL}/*.json: only 2025-08..2025-12 (5 months) -> too
    short.
  - data/filings/8k/{SYMBOL}.json: REAL, genuine backfill since
    2018-01-01 for 20 symbols (the same 20-symbol intraday-microstructure
    universe: AAPL/AMAT/AMD/AVGO/GOOGL/INTC/LITE/LRCX/META/MRVL/MSFT/MU/
    NBIS/NVDA/ORCL/PLTR/QCOM/SNDK/STX/WDC), with real filing dates and
    item codes (no EPS numbers, but real event timestamps). This is the
    ONE alternative-data source usable today without a new fetch.

Design (event-study, PRICE REACTION as the surprise proxy since we have
no EPS numbers):
  1. Filter each symbol's 8-K filings to those tagged item "2.02"
     ("Results of Operations and Financial Condition" — SEC's earnings-
     release item code).
  2. Classify each filing's reaction trading day from its
     acceptance_datetime: UTC hour < 14 (~before 9:30am ET) -> same
     calendar day is the reaction day (pre-market release, whole session
     reacts); otherwise -> the NEXT trading day (after-market-close
     release, market reacts the next session).
  3. reaction_return = close-to-close return from the trading day BEFORE
     the reaction day to the reaction day itself (captures the jump).
     This is the "surprise" proxy — we have no analyst-estimate number,
     so we use the market's own immediate repricing, same logic
     event-studies use when survey/estimate data is unavailable.
  4. fwd_return = close-to-close cumulative return from the reaction day
     to reaction day + HOLD_DAYS trading days (the drift window PEAD
     tests).
  5. Pool ALL events (symbol x time), rank by reaction_return, split
     into top/bottom terciles. PEAD hypothesis: top (biggest positive
     jump) should keep drifting up, bottom (biggest negative jump)
     should keep drifting down — i.e. CONTINUATION, the opposite of the
     already-dead xsection_mean_reversion's contrarian bet.
  6. Same falsification rule as the momentum Phase 0 script for direct
     comparability: monthly-aggregated top-minus-bottom spread t-stat
     and net Sharpe of a "long top tercile / short bottom tercile" book,
     held HOLD_DAYS, costed via python/core/fees_equity.py.

Usage:
    .venv/bin/python scripts/run_v2_8k_drift_phase0.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.core.fees_equity import commission, finra_taf, market_impact, sec_section_31_fee

log = logging.getLogger(__name__)

FILINGS_DIR = Path("data/filings/8k")
PRICE_DIR = Path("data/history")
OUT_DIR = Path("backtests/v2_momentum")

HOLD_DAYS = 10          # PEAD drift window, trading days
CAPITAL = 10_000_000.0
NOTIONAL_PER_EVENT = 200_000.0   # fixed single-name position size per event
IMPACT_COEFFICIENT_BPS = 10.0
HALF_SPREAD_BPS = 5.0
MIN_EVENTS_PER_MONTH_SIDE = 1     # need >=1 top and >=1 bottom event in a month to count its spread


def _load_events(symbol: str) -> list[dict]:
    path = FILINGS_DIR / f"{symbol}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    events = []
    for f in data.get("filings", []):
        items = f.get("items", "") or ""
        if "2.02" not in items.split(","):
            continue
        try:
            acc_dt = datetime.fromisoformat(f["acceptance_datetime"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        events.append({
            "symbol": symbol,
            "filing_date": pd.Timestamp(f["filing_date"]),
            "acceptance_utc_hour": acc_dt.astimezone(timezone.utc).hour,
        })
    return events


def _load_price(symbol: str) -> pd.DataFrame | None:
    path = PRICE_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if df.empty or "close" not in df.columns:
        return None
    return df


def _next_trading_day_on_or_after(index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    pos = index.searchsorted(day)
    if pos >= len(index):
        return None
    return index[pos]


def build_events(symbols: list[str]) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        px = _load_price(sym)
        if px is None:
            log.warning("skip %s: no price CSV", sym)
            continue
        idx = px.index
        close = px["close"].astype(float)
        for ev in _load_events(sym):
            reaction_target = ev["filing_date"] if ev["acceptance_utc_hour"] < 14 else ev["filing_date"] + timedelta(days=1)
            reaction_day = _next_trading_day_on_or_after(idx, reaction_target)
            if reaction_day is None:
                continue
            pos = idx.get_loc(reaction_day)
            if pos < 1 or pos + HOLD_DAYS >= len(idx):
                continue  # need a prior close and a full forward window
            prior_close = close.iloc[pos - 1]
            reaction_close = close.iloc[pos]
            fwd_close = close.iloc[pos + HOLD_DAYS]
            if prior_close <= 0 or reaction_close <= 0:
                continue
            reaction_return = reaction_close / prior_close - 1.0
            fwd_return = fwd_close / reaction_close - 1.0
            rows.append({
                "symbol": sym,
                "reaction_day": reaction_day,
                "entry_price": reaction_close,
                "exit_price": fwd_close,
                "reaction_return": reaction_return,
                "fwd_return": fwd_return,
            })
    return pd.DataFrame(rows)


def _one_sample_t_stat(x: pd.Series) -> float:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(n)))


def _event_cost(shares: int, price: float, adv: float, side: str) -> float:
    notional = shares * price
    cost = commission(shares, price)
    cost += market_impact(notional, adv, IMPACT_COEFFICIENT_BPS)
    cost += notional * (HALF_SPREAD_BPS / 10_000.0)
    if side == "sell":
        cost += sec_section_31_fee(shares, price, "sell")
        cost += finra_taf(shares, "sell")
    return cost


def run_phase0(events: pd.DataFrame) -> dict:
    n = len(events)
    if n < 20:
        raise ValueError(f"only {n} events found — too few to test")

    ranked = events.sort_values("reaction_return")
    n_tercile = max(1, n // 3)
    bottom = ranked.iloc[:n_tercile].copy()
    top = ranked.iloc[-n_tercile:].copy()
    bottom["bucket"] = "bottom"
    top["bucket"] = "top"
    buckets = pd.concat([bottom, top])

    # --- Monthly-aggregated spread (comparable methodology/verdict rule
    # to scripts/run_v2_momentum_phase0.py) ---
    buckets["month"] = buckets["reaction_day"].dt.to_period("M")
    monthly_bucket_mean = buckets.groupby(["month", "bucket"])["fwd_return"].mean().unstack("bucket")
    monthly_bucket_count = buckets.groupby(["month", "bucket"])["fwd_return"].count().unstack("bucket")
    valid = (monthly_bucket_count.get("top", 0) >= MIN_EVENTS_PER_MONTH_SIDE) & \
            (monthly_bucket_count.get("bottom", 0) >= MIN_EVENTS_PER_MONTH_SIDE)
    monthly_spread = (monthly_bucket_mean["top"] - monthly_bucket_mean["bottom"])[valid].dropna()

    # --- Net Sharpe of a long-top/short-bottom book, priced with real
    # per-event costs, aggregated into the SAME monthly buckets ---
    monthly_pnl: dict = {}
    for _, row in buckets.iterrows():
        direction = 1.0 if row["bucket"] == "top" else -1.0
        shares = int(NOTIONAL_PER_EVENT / row["entry_price"])
        if shares == 0:
            continue
        adv_guess = NOTIONAL_PER_EVENT * 50  # conservative fallback: assume position is 2% of ADV
        entry_side = "buy" if direction > 0 else "sell"
        exit_side = "sell" if direction > 0 else "buy"
        entry_cost = _event_cost(shares, row["entry_price"], adv_guess, entry_side)
        exit_cost = _event_cost(shares, row["exit_price"], adv_guess, exit_side)
        gross_pnl = direction * shares * (row["exit_price"] - row["entry_price"])
        net_pnl = gross_pnl - entry_cost - exit_cost
        month = row["reaction_day"].to_period("M")
        acc = monthly_pnl.setdefault(month, {"gross": 0.0, "net": 0.0})
        acc["gross"] += gross_pnl
        acc["net"] += net_pnl

    months_sorted = sorted(monthly_pnl.keys())
    gross_ret = pd.Series({m: monthly_pnl[m]["gross"] / CAPITAL for m in months_sorted})
    net_ret = pd.Series({m: monthly_pnl[m]["net"] / CAPITAL for m in months_sorted})

    def _stats(returns: pd.Series) -> dict:
        if len(returns) < 2 or returns.std(ddof=1) == 0:
            return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}
        sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(12))
        curve = (1.0 + returns).cumprod()
        n_years = len(returns) / 12.0
        total = float(curve.iloc[-1])
        cagr = total ** (1.0 / n_years) - 1.0 if total > 0 and n_years > 0 else -1.0
        running_max = curve.cummax()
        mdd = float((curve / running_max - 1.0).min())
        return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": mdd}

    return {
        "n_events_total": n,
        "n_events_top": len(top),
        "n_events_bottom": len(bottom),
        "date_range": [str(events["reaction_day"].min().date()), str(events["reaction_day"].max().date())],
        "hold_days": HOLD_DAYS,
        "gross": _stats(gross_ret),
        "net": _stats(net_ret),
        "top_minus_bottom_spread_monthly_mean": float(monthly_spread.mean()) if len(monthly_spread) else 0.0,
        "top_minus_bottom_spread_t_stat": _one_sample_t_stat(monthly_spread),
        "n_months_with_both_buckets": len(monthly_spread),
    }


def _verdict(result: dict) -> str:
    reasons = []
    if result["net"]["sharpe"] <= 0:
        reasons.append(f"net Sharpe {result['net']['sharpe']:.2f} <= 0")
    if abs(result["top_minus_bottom_spread_t_stat"]) < 2.0:
        reasons.append(f"spread t-stat {result['top_minus_bottom_spread_t_stat']:.2f} "
                        f"(|t| < 2 -> no significant continuation/drift after the reaction)")
    return "NO-GO (Phase 0): " + "; ".join(reasons) if reasons else \
        "PASS Phase 0 (proceed to Phase 1 — full 20-symbol universe is not enough for promotion)"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    symbols = sorted(p.stem for p in FILINGS_DIR.glob("*.json"))
    log.info("=== 8-K item-2.02 event drift: %d symbols ===", len(symbols))
    events = build_events(symbols)
    log.info("Built %d usable events (has prior close + %d-day forward window)", len(events), HOLD_DAYS)

    result = run_phase0(events)
    verdict = _verdict(result)

    log.info("--- 8k_drift (%d events: %d top / %d bottom, %s to %s, %d-day hold) ---",
              result["n_events_total"], result["n_events_top"], result["n_events_bottom"], *result["date_range"], HOLD_DAYS)
    log.info("GROSS: Sharpe %.2f, CAGR %.1f%%, MaxDD %.1f%%",
              result["gross"]["sharpe"], result["gross"]["cagr"] * 100, result["gross"]["max_drawdown"] * 100)
    log.info("NET:   Sharpe %.2f, CAGR %.1f%%, MaxDD %.1f%%",
              result["net"]["sharpe"], result["net"]["cagr"] * 100, result["net"]["max_drawdown"] * 100)
    log.info("Top-bottom monthly drift spread: mean %.3f%%, t-stat %.2f (%d months with both buckets populated)",
              result["top_minus_bottom_spread_monthly_mean"] * 100,
              result["top_minus_bottom_spread_t_stat"], result["n_months_with_both_buckets"])
    log.info("VERDICT: %s", verdict)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "phase0_8k_drift_report.json"
    out_path.write_text(json.dumps({**result, "verdict": verdict}, indent=2, default=str), encoding="utf-8")
    log.info("Full report written to %s", out_path)


if __name__ == "__main__":
    main()
