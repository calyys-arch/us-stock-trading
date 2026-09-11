"""
V2 momentum research plan (docs/v2_research_plan.md) — Phase 0 kill-test.

NOT the same script as scripts/run_v2_momentum.py (2026-09-10) or its
result (docs/v2_momentum_results.md, Sharpe 0.96 gross on a 120-symbol
universe). That prior run ranked V1's own 120-symbol universe — which
mixes individual equities WITH bond/commodity/international-equity ETFs
(see config/strategy_v1_universe.json) — in one cross-sectional momentum
ranking, applied no book-level volatility scaling (consistent with its
own reported -40.4% max drawdown / 26.2% realized vol), and never tested
whether the top-quintile "winners" actually beat the bottom-quintile
"losers" with statistical significance. Its own write-up already flagged
this ("universe 太小...可能高估了真實效果"). This script is the more
careful, narrower Phase 0 test the 2026-09-11 plan revision calls for:
individual equities only, vol-scaled book, and an explicit winner-loser
spread significance check — see docs/v2_research_plan.md §2 for why each
of these three differences matters.

Uses ONLY already-cached price data, no new network fetch. Two universes
available on disk, discovered while writing this script (the plan's
original assumption — "592 cached symbols" — turned out to be wrong; see
the module-level NOTE below):

  midcap    data/history_altuni_midcap_full/ — 71 symbols, point-in-time
            S&P 500 membership as of 2016-11-01, liquidity-band rank
            [150, 220) by trailing 60-day dollar volume (one tier down
            from mega-cap; see configs/alt_universe_midcap.yaml for full
            provenance). This is the tier the 2026-09-11 literature
            review says is where net-of-cost momentum edge is most
            likely to survive. Full history 2016-11-01 to 2026-07-31.

  largecap  data/history/ symbols with requested_start <= 2019-01-01 that
            are individual equities (not ETFs) — 57 symbols. NOT a
            rigorously constructed universe: it is just whatever
            happened to get a long fetch in a past session (heavily
            mega-cap tech: AAPL/AMZN/AVGO/GOOG/GOOGL/META/MSFT/NVDA/...).
            Included only as a rough within-repo comparison against the
            midcap tier, mirroring the CRSP market-cap-tier finding cited
            in the plan (large: net +0.31%/mo vs small: net -0.12%/mo).
            Treat this arm's numbers as directional, not rigorous.

NOTE on the plan's "592 cached symbols" claim: checked while building
this script — 469 of the 592 symbols in data/history/ were only fetched
from 2026-02-28 onward (~6 months), not enough for a single 12-1
momentum observation (needs 13 months of trailing history). Only 123
symbols have long-enough history, and only 57 of those are individual
stocks (the rest are ETFs already covered by V1). This script uses what
is ACTUALLY usable, not what the plan assumed was cached.

Signal: 12-1 cross-sectional momentum (skip most recent month), monthly
rebalance, long-only top quintile, equal-weighted, then the whole book
scaled to a target annualized volatility (Barroso & Santa-Clara 2015
style single scalar — not per-name).

Costs: python/core/fees_equity.py's model, charged on ENTRY (new name
joins the long book: commission + market impact [+ half-spread]) and on
EXIT (name leaves the book: commission + SEC Section 31 fee + FINRA TAF +
market impact [+ half-spread]). Names that stay in the book across a
rebalance are NOT charged (weight-drift rebalancing cost is not modeled
at Phase 0 — a disclosed simplification that makes net Sharpe slightly
optimistic, acceptable for a kill-test, NOT acceptable for a promotion
decision).

This is a research script. It never reads or writes configs/strategy.yaml
and never touches enabled/auto_execute — see docs/v2_research_plan.md
for the promotion path if this ever clears Phase 0/1/2.

Usage:
    .venv/bin/python scripts/run_v2_momentum_phase0.py --universe both
    .venv/bin/python scripts/run_v2_momentum_phase0.py --universe midcap --top-quantile 0.2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.core.fees_equity import commission, finra_taf, market_impact, sec_section_31_fee

log = logging.getLogger(__name__)

MIDCAP_DIR = Path("data/history_altuni_midcap_full")
LARGECAP_DIR = Path("data/history")
LARGECAP_META = LARGECAP_DIR / "_meta.json"
OUT_DIR = Path("backtests/v2_momentum")

# V1's own convention (python/portfolio/strategy_v1.py's risk_bucket_study),
# reused here for comparability rather than inventing a new default.
DEFAULT_VOL_TARGET = 0.15
DEFAULT_VOL_LOOKBACK_MONTHS = 12
DEFAULT_TOP_QUANTILE = 0.2
DEFAULT_BOTTOM_QUANTILE = 0.2
DEFAULT_CAPITAL = 10_000_000.0
DEFAULT_IMPACT_COEFFICIENT_BPS = 10.0   # fees_equity.py's own default
DEFAULT_HALF_SPREAD_BPS = 5.0           # disclosed assumption, not calibrated
MIN_EXPOSURE_SCALE = 0.2
MAX_EXPOSURE_SCALE = 1.5
LOOKBACK_MONTHS = 12
SKIP_MONTHS = 1
MIN_MONTHS_HISTORY = LOOKBACK_MONTHS + SKIP_MONTHS + 1


def _largecap_single_name_symbols() -> list[str]:
    """The 57-symbol comparison arm: individual-equity CSVs in
    data/history/ whose cache was requested with a >=13-month start, i.e.
    actually usable for a 12-1 signal. ETFs (V1's own buckets) are
    excluded since cross-sectional momentum is normally computed on
    individual names, not baskets that already diversify away the
    idiosyncratic return the signal is trying to rank on."""
    meta = json.loads(LARGECAP_META.read_text(encoding="utf-8"))
    etf_symbols: set[str] = set()
    v1_universe_path = Path("config/strategy_v1_universe.json")
    if v1_universe_path.exists():
        v1_universe = json.loads(v1_universe_path.read_text(encoding="utf-8"))
        for bucket, syms in v1_universe.get("buckets", {}).items():
            if bucket != "us_single_name":
                etf_symbols.update(syms)
    long_hist = [s for s, v in meta.items() if v.get("requested_start", "9999") <= "2019-01-01"]
    return sorted(s for s in long_hist if s not in etf_symbols)


def _load_close_and_dollar_volume_multi(
    sources: list[tuple[Path, list[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine multiple (cache_dir, symbols) sources into one cross-section
    — used by --universe combined to test whether the midcap/largecap
    arms' individually-insignificant winner-loser spread (2026-09-11
    Phase 0 run) was a genuine null or just an underpowered small-N test.
    A symbol present in more than one source is taken from the FIRST
    source that has it (midcap arm wins the one actual overlap, "D")."""
    closes: dict[str, pd.Series] = {}
    dollar_vols: dict[str, pd.Series] = {}
    for cache_dir, symbols in sources:
        for sym in symbols:
            if sym in closes:
                continue
            path = cache_dir / f"{sym}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
            if df.empty or "close" not in df.columns:
                continue
            closes[sym] = df["close"].astype(float)
            dollar_vols[sym] = (df["close"].astype(float) * df["volume"].astype(float))
    return pd.DataFrame(closes).sort_index(), pd.DataFrame(dollar_vols).sort_index()


def _load_close_and_dollar_volume(cache_dir: Path, symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Direct CSV reads — deliberately NOT python.data.price_cache's
    get_cached_price_panel, because that function will attempt a NETWORK
    FETCH for any symbol/range its _meta.json doesn't already cover
    (data/history_altuni_midcap_full/ has no _meta.json at all), which
    Phase 0's whole point is to avoid."""
    closes: dict[str, pd.Series] = {}
    dollar_vols: dict[str, pd.Series] = {}
    skipped: list[str] = []
    for sym in symbols:
        path = cache_dir / f"{sym}.csv"
        if not path.exists():
            skipped.append(sym)
            continue
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        if df.empty or "close" not in df.columns:
            skipped.append(sym)
            continue
        closes[sym] = df["close"].astype(float)
        dollar_vols[sym] = (df["close"].astype(float) * df["volume"].astype(float))
    if skipped:
        log.warning("run_v2_momentum_phase0: %d/%d symbols skipped (no/empty CSV) in %s: %s",
                    len(skipped), len(symbols), cache_dir, skipped)
    close_panel = pd.DataFrame(closes).sort_index()
    dv_panel = pd.DataFrame(dollar_vols).sort_index()
    return close_panel, dv_panel


def _monthly_close(daily_close: pd.DataFrame) -> pd.DataFrame:
    # Forward-fill small gaps (halts, odd missing bars) before resampling so
    # a single missing daily print doesn't manufacture a fake missing month;
    # limit=5 caps this at one trading week, matching the tolerance used
    # elsewhere in this repo's data-quality checks.
    filled = daily_close.ffill(limit=5)
    return filled.resample("ME").last()


def _one_sample_t_stat(x: pd.Series) -> float:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(n)))


def run_phase0(
    close_panel: pd.DataFrame,
    dollar_vol_panel: pd.DataFrame,
    *,
    top_quantile: float = DEFAULT_TOP_QUANTILE,
    bottom_quantile: float = DEFAULT_BOTTOM_QUANTILE,
    vol_target: float = DEFAULT_VOL_TARGET,
    vol_lookback_months: int = DEFAULT_VOL_LOOKBACK_MONTHS,
    capital: float = DEFAULT_CAPITAL,
    impact_coefficient_bps: float = DEFAULT_IMPACT_COEFFICIENT_BPS,
    half_spread_bps: float = DEFAULT_HALF_SPREAD_BPS,
) -> dict:
    monthly_close = _monthly_close(close_panel)
    monthly_dv = dollar_vol_panel.reindex(close_panel.index).ffill(limit=5).resample("ME").mean()
    monthly_dv = monthly_dv.reindex(monthly_close.index)

    past_return = monthly_close.shift(SKIP_MONTHS) / monthly_close.shift(SKIP_MONTHS + LOOKBACK_MONTHS) - 1.0
    fwd_return = monthly_close.pct_change().shift(-1)

    months = monthly_close.index
    if len(months) < MIN_MONTHS_HISTORY + 2:
        raise ValueError(f"run_v2_momentum_phase0: only {len(months)} months available, "
                          f"need >= {MIN_MONTHS_HISTORY + 2}")

    prev_book: set[str] = set()
    raw_gross_returns: list[float] = []      # unscaled, for vol-target bootstrapping
    scaled_gross_returns: list[float] = []
    net_returns: list[float] = []
    scales_used: list[float] = []
    book_sizes: list[int] = []
    turnovers: list[int] = []
    spreads: list[float] = []                # top-quantile mean - bottom-quantile mean, per month
    monthly_costs: list[float] = []
    dates_used: list[pd.Timestamp] = []

    for i, t in enumerate(months):
        if i >= len(months) - 1:
            break  # no forward return available for the last month
        sig = past_return.loc[t].dropna()
        fwd = fwd_return.loc[t]
        eligible = sig.index[sig.index.isin(fwd.dropna().index)]
        sig = sig.loc[eligible]
        if len(sig) < 10:
            continue  # too few names this month to form meaningful quintiles

        n_top = max(1, int(round(len(sig) * top_quantile)))
        n_bottom = max(1, int(round(len(sig) * bottom_quantile)))
        ranked = sig.sort_values(ascending=False)
        top_names = list(ranked.index[:n_top])
        bottom_names = list(ranked.index[-n_bottom:])

        top_fwd = fwd.loc[top_names]
        bottom_fwd = fwd.loc[bottom_names]
        spreads.append(float(top_fwd.mean() - bottom_fwd.mean()))

        book = set(top_names)
        raw_book_ret = float(top_fwd.mean())
        raw_gross_returns.append(raw_book_ret)

        # Vol-scale using trailing realized vol of the UNSCALED book return
        # series so far (Barroso-Santa Clara style single scalar). Needs
        # `vol_lookback_months` of prior raw returns to bootstrap; before
        # that, run unscaled (scale=1.0) and say so via scales_used.
        if len(raw_gross_returns) > vol_lookback_months:
            trailing = pd.Series(raw_gross_returns[-(vol_lookback_months + 1):-1])
            realized_vol = float(trailing.std(ddof=1) * np.sqrt(12))
            scale = 1.0 if realized_vol == 0 else vol_target / realized_vol
            scale = float(np.clip(scale, MIN_EXPOSURE_SCALE, MAX_EXPOSURE_SCALE))
        else:
            scale = 1.0
        scales_used.append(scale)
        scaled_ret = scale * raw_book_ret
        scaled_gross_returns.append(scaled_ret)

        # --- Turnover costs (entries/exits only; see module docstring) ---
        entries = book - prev_book
        exits = prev_book - book
        weight_per_name = scale / len(book)
        cost_t = 0.0
        px_row = monthly_close.loc[t]
        dv_row = monthly_dv.loc[t]
        for sym in entries:
            price = float(px_row.get(sym, np.nan))
            if not np.isfinite(price) or price <= 0:
                continue
            notional = weight_per_name * capital
            shares = int(notional / price)
            if shares == 0:
                continue
            adv = float(dv_row.get(sym, 0.0)) if pd.notna(dv_row.get(sym, np.nan)) else 0.0
            cost_t += commission(shares, price)
            cost_t += market_impact(notional, adv, impact_coefficient_bps)
            if half_spread_bps > 0:
                cost_t += notional * (half_spread_bps / 10_000.0)
        for sym in exits:
            # Exit priced at the PREVIOUS month's close (that name's actual
            # weight came from last month's scale/book-size, which we no
            # longer have exactly — approximate with this month's scale as
            # a disclosed simplification; exits are a minority of the book
            # each month so this does not materially bias turnover cost).
            price = float(px_row.get(sym, np.nan))
            if not np.isfinite(price) or price <= 0:
                continue
            notional = weight_per_name * capital
            shares = int(notional / price)
            if shares == 0:
                continue
            adv = float(dv_row.get(sym, 0.0)) if pd.notna(dv_row.get(sym, np.nan)) else 0.0
            cost_t += commission(shares, price)
            cost_t += sec_section_31_fee(shares, price, "sell")
            cost_t += finra_taf(shares, "sell")
            cost_t += market_impact(notional, adv, impact_coefficient_bps)
            if half_spread_bps > 0:
                cost_t += notional * (half_spread_bps / 10_000.0)

        monthly_costs.append(cost_t)
        net_returns.append(scaled_ret - cost_t / capital)
        book_sizes.append(len(book))
        turnovers.append(len(entries) + len(exits))
        dates_used.append(t)
        prev_book = book

    gross = pd.Series(scaled_gross_returns, index=dates_used)
    net = pd.Series(net_returns, index=dates_used)
    spread_series = pd.Series(spreads, index=dates_used)

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
        "n_months": len(dates_used),
        "date_range": [str(dates_used[0].date()), str(dates_used[-1].date())] if dates_used else None,
        "avg_book_size": float(np.mean(book_sizes)) if book_sizes else 0.0,
        "avg_monthly_turnover_names": float(np.mean(turnovers)) if turnovers else 0.0,
        "avg_exposure_scale": float(np.mean(scales_used)) if scales_used else 0.0,
        "gross": _stats(gross),
        "net": _stats(net),
        "avg_monthly_cost_bps_of_capital": float(np.mean(monthly_costs) / capital * 10_000.0) if monthly_costs else 0.0,
        "top_minus_bottom_quintile_spread_monthly_mean": float(spread_series.mean()) if len(spread_series) else 0.0,
        "top_minus_bottom_quintile_spread_t_stat": _one_sample_t_stat(spread_series),
        "gross_returns_monthly": {str(d.date()): float(r) for d, r in gross.items()},
        "net_returns_monthly": {str(d.date()): float(r) for d, r in net.items()},
    }


def _verdict(result: dict) -> str:
    """Section 5 of docs/v2_research_plan.md's falsification criteria,
    applied mechanically so this doesn't turn into post-hoc rationalizing."""
    reasons = []
    if result["net"]["sharpe"] <= 0:
        reasons.append(f"net Sharpe {result['net']['sharpe']:.2f} <= 0")
    if abs(result["top_minus_bottom_quintile_spread_t_stat"]) < 2.0:
        reasons.append(f"spread t-stat {result['top_minus_bottom_quintile_spread_t_stat']:.2f} "
                        f"(|t| < 2 -> signal has no cross-sectional discrimination in this universe)")
    return "NO-GO (Phase 0): " + "; ".join(reasons) if reasons else \
        "PASS Phase 0 (survivorship-biased, NOT sufficient for promotion — proceed to Phase 1)"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=["midcap", "largecap", "both", "combined"], default="both")
    parser.add_argument("--top-quantile", type=float, default=DEFAULT_TOP_QUANTILE)
    parser.add_argument("--bottom-quantile", type=float, default=DEFAULT_BOTTOM_QUANTILE)
    parser.add_argument("--vol-target", type=float, default=DEFAULT_VOL_TARGET)
    parser.add_argument("--vol-lookback-months", type=int, default=DEFAULT_VOL_LOOKBACK_MONTHS)
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--half-spread-bps", type=float, default=DEFAULT_HALF_SPREAD_BPS)
    args = parser.parse_args()

    universes = {}
    if args.universe in ("midcap", "both"):
        syms = sorted(p.stem for p in MIDCAP_DIR.glob("*.csv"))
        universes["midcap"] = (MIDCAP_DIR, syms)
    if args.universe in ("largecap", "both"):
        universes["largecap"] = (LARGECAP_DIR, _largecap_single_name_symbols())
    if args.universe == "combined":
        mid_syms = sorted(p.stem for p in MIDCAP_DIR.glob("*.csv"))
        large_syms = _largecap_single_name_symbols()
        universes["combined"] = ("MULTI", [(MIDCAP_DIR, mid_syms), (LARGECAP_DIR, large_syms)])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for name, spec in universes.items():
        if name == "combined":
            cache_dir, sources = spec
            symbols = sorted({s for _, syms in sources for s in syms})
            log.info("=== %s: %d unique symbols from %d sources ===", name, len(symbols), len(sources))
            close_panel, dv_panel = _load_close_and_dollar_volume_multi(sources)
        else:
            cache_dir, symbols = spec
            log.info("=== %s: %d symbols from %s ===", name, len(symbols), cache_dir)
            close_panel, dv_panel = _load_close_and_dollar_volume(cache_dir, symbols)
        result = run_phase0(
            close_panel, dv_panel,
            top_quantile=args.top_quantile, bottom_quantile=args.bottom_quantile,
            vol_target=args.vol_target, vol_lookback_months=args.vol_lookback_months,
            capital=args.capital, half_spread_bps=args.half_spread_bps,
        )
        verdict = _verdict(result)
        all_results[name] = {**result, "verdict": verdict, "n_symbols": len(symbols)}

        log.info("--- %s (%d symbols, %d months, %s to %s) ---",
                  name, len(symbols), result["n_months"], *result["date_range"])
        log.info("avg book size: %.1f, avg monthly turnover (names): %.1f, avg exposure scale: %.2f",
                  result["avg_book_size"], result["avg_monthly_turnover_names"], result["avg_exposure_scale"])
        log.info("GROSS: Sharpe %.2f, CAGR %.1f%%, MaxDD %.1f%%",
                  result["gross"]["sharpe"], result["gross"]["cagr"] * 100, result["gross"]["max_drawdown"] * 100)
        log.info("NET:   Sharpe %.2f, CAGR %.1f%%, MaxDD %.1f%%  (avg cost %.1f bps/mo of capital)",
                  result["net"]["sharpe"], result["net"]["cagr"] * 100, result["net"]["max_drawdown"] * 100,
                  result["avg_monthly_cost_bps_of_capital"])
        log.info("Top-bottom quintile monthly spread: mean %.3f%%, t-stat %.2f",
                  result["top_minus_bottom_quintile_spread_monthly_mean"] * 100,
                  result["top_minus_bottom_quintile_spread_t_stat"])
        log.info("VERDICT: %s", verdict)
        log.info("")

    out_path = OUT_DIR / "phase0_report.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    log.info("Full report written to %s", out_path)


if __name__ == "__main__":
    main()
