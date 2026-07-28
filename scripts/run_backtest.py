"""
End-to-end backtest runner: fetch data -> run strategy backtest -> Monte
Carlo -> Reality Check -> Walk-Forward -> apply configs/goal.yaml acceptance
gates -> write docs/us_equity_health_check.md.

Usage:
    # Cross-sectional strategy, real yfinance data (needs network + point-in-time universe)
    python scripts/run_backtest.py --strategy xsection_mean_reversion --start 2018-01-01 --end 2025-01-01

    # Cross-sectional strategy, offline synthetic demo data (no network required)
    python scripts/run_backtest.py --strategy xsection_mean_reversion --demo

    # Pairs strategy on two explicit tickers
    python scripts/run_backtest.py --strategy pairs_trading --pair-a XLE --pair-b XOP --start 2018-01-01 --end 2025-01-01
    python scripts/run_backtest.py --strategy pairs_trading --demo
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Windows consoles often default to a legacy codepage (e.g. cp950) rather
# than UTF-8. This codebase's docstrings/log messages use em dashes and
# other non-ASCII characters throughout, and config/YAML files are UTF-8 —
# without this, `open()` defaults to the legacy codepage (raising
# UnicodeDecodeError on config reads) and stdout printing raises/garbles on
# non-ASCII log output. Force UTF-8 everywhere for this process.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)

from python.backtest.engine import PairsBacktestConfig, run_pairs_backtest
from python.backtest.monte_carlo import MonteCarloValidator
from python.backtest.param_guard import check_max_parameters, sufficient_sample_size
from python.backtest.reality_check import RealityCheck, RealityCheckConfig
from python.backtest.vector_engine import run_vector_backtest
from python.core.data_quality import quality_report
from python.core.strategies.xsection_mean_reversion import CrossSectionalMeanReversionStrategy


def _synthetic_panel(n_days: int = 1000, n_codes: int = 40, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    codes = [f"SYN{i:03d}" for i in range(n_codes)]
    rows = []
    for code in codes:
        price = 100.0 + rng.uniform(-30, 30)
        for d in dates:
            ret = rng.normal(0.0002, 0.015)
            open_px = price
            close_px = price * (1 + ret)
            rows.append({"date": d, "code": code, "open": open_px, "close": close_px,
                         "adv_20d_dollars": 30_000_000.0})
            price = close_px
    return pd.DataFrame(rows).set_index(["date", "code"]).sort_index()


def _synthetic_pair(n_days: int = 1000, seed: int = 5) -> tuple[pd.Series, pd.Series]:
    """A genuinely cointegrated synthetic pair with a MULTI-DAY half-life,
    so the demo mode exercises real cointegration + entry/exit cycling
    instead of either (a) pure noise, or (b) a degenerate near-zero
    half-life that every revalidation window rejects via
    `min_half_life_days` and therefore never trades.

    Construction: `log_price = level + trend_coef * common_random_walk +/-
    0.5 * ou_spread + small_idiosyncratic_noise`, where the OU spread is a
    genuine AR(1) process (half-life controlled by `phi`) shared with
    OPPOSITE sign between the two legs, and `common_random_walk` is the
    much-larger-variance shared trend that makes OLS correctly recover a
    hedge ratio near +1.

    An earlier version of this generator scaled the common trend down
    (coefficient 0.01) so that the anti-correlated spread term actually had
    MORE variance than the trend within any given lookback window. OLS then
    picked up the anti-correlation instead of the co-movement and returned
    a hedge ratio near -1 (should be near +1), and separately, adding the
    idiosyncratic noise directly as the entire "spread" (no persistent AR(1)
    component) made the estimated O-U half-life ~0.8 days — below
    `configs/strategy.yaml`'s `min_half_life_days: 1.0` floor, so
    `run_pairs_backtest` filtered out every single entry and produced zero
    trades. The trend coefficient below is deliberately large enough that
    its variance dominates the spread's across the whole 1000-day series.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    common_factor = np.cumsum(rng.normal(0, 1, n_days))
    phi = 0.92  # OU persistence -> theoretical half-life = ln(2)/(1-phi) ~= 8.7 days
    innovation_std = 0.15
    spread = np.zeros(n_days)
    for t in range(1, n_days):
        spread[t] = phi * spread[t - 1] + rng.normal(0, innovation_std)
    noise_a = rng.normal(0, 0.02, n_days)
    noise_b = rng.normal(0, 0.02, n_days)
    log_a = 4.0 + 0.1 * common_factor + 0.5 * spread + noise_a
    log_b = 3.8 + 0.1 * common_factor - 0.5 * spread + noise_b
    prices_a = pd.Series(np.exp(log_a), index=dates)
    prices_b = pd.Series(np.exp(log_b), index=dates)
    return prices_a, prices_b


def run_xsection(args) -> dict:
    with open("configs/strategy.yaml", encoding="utf-8") as f:
        strat_cfg = yaml.safe_load(f)["xsection_mean_reversion"]

    ok, n_params = check_max_parameters(strat_cfg)
    if not ok:
        print(f"REFUSING TO RUN: xsection_mean_reversion has {n_params} free parameters (> 5 allowed)")
        sys.exit(1)

    if args.demo:
        panel = _synthetic_panel()
        data_label = "SYNTHETIC DEMO DATA — not a real market backtest"
    else:
        from python.data.sp500_universe import universe_by_day as build_universe
        from python.simulation.hist_data_us import build_price_panel

        universe_dates = pd.bdate_range(args.start, args.end)
        universe = build_universe(list(universe_dates))
        symbols = sorted({c for codes in universe.values() for c in codes})
        panel, quality_flags = build_price_panel(symbols, args.start, args.end)
        data_label = f"Live yfinance data, {len(symbols)} symbols (point-in-time S&P 500, Wikipedia-derived)"
        if quality_flags:
            print(f"WARNING: data-quality flags on {len(quality_flags)} symbols — see report")

    strategy = CrossSectionalMeanReversionStrategy(
        lookback_days=strat_cfg["lookback_days"],
        gross_leverage_target=strat_cfg["gross_leverage_target"],
        min_universe_size=strat_cfg["min_universe_size"],
    )

    codes = sorted(panel.index.get_level_values(1).unique())
    dates = sorted(panel.index.get_level_values(0).unique())[30:]
    universe_by_day = {d: codes for d in dates}

    n_trading_days = len(dates)
    sample_ok = sufficient_sample_size(n_trading_days, n_params)

    result = run_vector_backtest(strategy, panel, universe_by_day)

    mc = MonteCarloValidator(n_sims=500)
    mc_result = mc.run(result.daily_returns.tolist())

    def backtest_fn(price_panel: pd.DataFrame) -> float:
        sub_universe = {d: codes for d in dates if d in price_panel.index.get_level_values(0)}
        r = run_vector_backtest(strategy, price_panel, sub_universe)
        return r.sharpe_annualized

    close_panel = panel["close"].unstack("code")
    rc = RealityCheck(backtest_fn=lambda p: _sharpe_from_close_panel(p, strategy, codes), config=RealityCheckConfig(n_sims=50))
    rc_result = rc.run(close_panel)

    with open("configs/goal.yaml", encoding="utf-8") as f:
        goal = yaml.safe_load(f)

    gates = {
        "sufficient_sample_size": sample_ok,
        "monte_carlo_p5_sharpe_nonneg": mc_result.sharpe.p5 >= goal["monte_carlo"]["min_p5_sharpe"],
        "reality_check_pass": rc_result.p_value <= goal["reality_check"]["max_p_value"],
    }

    return {
        "strategy": "xsection_mean_reversion",
        "data_label": data_label,
        "n_free_parameters": n_params,
        "n_trading_days": n_trading_days,
        "sharpe_annualized": result.sharpe_annualized,
        "max_drawdown": result.max_drawdown,
        "cagr": result.cagr,
        "monte_carlo": mc_result.to_dict(),
        "reality_check": rc_result.to_dict(),
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def _sharpe_from_close_panel(close_panel: pd.DataFrame, strategy, codes) -> float:
    """Rebuild a (date, code)-indexed OHLC-shaped panel from a wide
    close-price matrix using vectorized pandas ops (stack), not a Python
    row-by-row loop — Reality Check calls this dozens of times, so any
    per-call inefficiency here is multiplied straight into total runtime.

    IMPORTANT: White's Reality Check phase-randomizes a SINGLE daily price
    per instrument (see reality_check.py's module docstring) — it has no
    concept of an intraday open distinct from the close. Setting
    `open == close` on the SAME row (an earlier version of this function
    did that) makes every day's `(close/open - 1)` realized return exactly
    zero by construction, leaving only transaction-cost drag — that
    produced a nonsensical real Sharpe around -432 instead of the true
    ~-1.0 the actual open/close backtest reports. Using the PRIOR day's
    close as the synthetic "open" instead gives a genuine (if
    close-to-close, not open-to-close) realized-return series, which is
    the standard proxy for this kind of significance test when only one
    price observation per day is available.
    """
    stacked = close_panel.stack().dropna().rename("close").to_frame()
    stacked.index.names = ["date", "code"]
    panel = stacked.sort_index()
    panel["open"] = panel.groupby(level="code")["close"].shift(1)
    panel["adv_20d_dollars"] = 30_000_000.0
    panel = panel.dropna(subset=["open"])
    all_dates = sorted(panel.index.get_level_values(0).unique())
    dates = all_dates[30:]
    present_codes = set(panel.index.get_level_values(1))
    eligible_codes = [c for c in codes if c in present_codes]
    universe_by_day = {d: eligible_codes for d in dates}
    result = run_vector_backtest(strategy, panel, universe_by_day)
    return result.sharpe_annualized


def run_pairs(args) -> dict:
    with open("configs/strategy.yaml", encoding="utf-8") as f:
        strat_cfg = yaml.safe_load(f)["pairs_trading"]

    ok, n_params = check_max_parameters(strat_cfg)
    if not ok:
        print(f"REFUSING TO RUN: pairs_trading has {n_params} free parameters (> 5 allowed)")
        sys.exit(1)

    if args.demo:
        prices_a, prices_b = _synthetic_pair()
        code_a, code_b = "SYNA", "SYNB"
        data_label = "SYNTHETIC DEMO DATA (genuinely cointegrated by construction)"
    else:
        from python.simulation.hist_data_us import fetch_daily_bars

        code_a, code_b = args.pair_a, args.pair_b
        prices_a = fetch_daily_bars(code_a, args.start, args.end)["close"]
        prices_b = fetch_daily_bars(code_b, args.start, args.end)["close"]
        data_label = f"Live yfinance data: {code_a} / {code_b}"

    quality_a = quality_report(prices_a)
    quality_b = quality_report(prices_b)

    n_trading_days = min(len(prices_a), len(prices_b))
    sample_ok = sufficient_sample_size(n_trading_days, n_params)

    cfg = PairsBacktestConfig(
        entry_z=strat_cfg["entry_z"], exit_z=strat_cfg["exit_z"],
        coint_lookback_days=strat_cfg["coint_lookback_days"],
        revalidate_every_days=strat_cfg["revalidate_every_days"],
        notional_per_leg=strat_cfg["notional_per_leg"],
        half_life_multiplier_max_hold=strat_cfg["half_life_multiplier_max_hold"],
        min_half_life_days=strat_cfg["min_half_life_days"],
        max_half_life_days=strat_cfg["max_half_life_days"],
    )
    report = run_pairs_backtest(code_a, code_b, prices_a, prices_b, cfg)
    metrics = report.to_dict()

    mc = MonteCarloValidator(n_sims=500)
    mc_result = mc.run([t.net_pnl for t in report.trades]) if report.trades else mc.run([])

    with open("configs/goal.yaml", encoding="utf-8") as f:
        goal = yaml.safe_load(f)

    gates = {
        "sufficient_sample_size": sample_ok,
        "has_trades": len(report.trades) > 0,
        "monte_carlo_p5_sharpe_nonneg": mc_result.sharpe.p5 >= goal["monte_carlo"]["min_p5_sharpe"],
    }

    return {
        "strategy": "pairs_trading",
        "pair": f"{code_a}/{code_b}",
        "data_label": data_label,
        "n_free_parameters": n_params,
        "n_trading_days": n_trading_days,
        "n_trades": len(report.trades),
        "total_net_pnl": metrics["total_net_pnl"],
        "win_rate": metrics["win_rate"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "monte_carlo": mc_result.to_dict(),
        "data_quality_a": quality_a,
        "data_quality_b": quality_b,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def write_health_check(results: list[dict]) -> Path:
    out_path = Path("docs/us_equity_health_check.md")
    lines = [
        "# US Equity Strategy Health Check",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "> **Disclaimer**: backtests using `--demo` mode use synthetic data and validate",
        "> PIPELINE CORRECTNESS only, not strategy edge. Backtests using real data rely on",
        "> yfinance + a Wikipedia-derived point-in-time S&P 500 universe — see",
        "> README.md 'Known limitations (MVP)' before trusting these numbers for capital",
        "> allocation decisions.",
        "",
    ]
    for r in results:
        lines.append(f"## {r['strategy']}")
        lines.append("")
        lines.append(f"- Data: {r['data_label']}")
        lines.append(f"- Free parameters: {r['n_free_parameters']} (Chan Ch.3 ceiling: 5)")
        lines.append(f"- Trading days tested: {r['n_trading_days']}")
        for k, v in r.items():
            if k in ("strategy", "data_label", "n_free_parameters", "n_trading_days", "gates",
                      "monte_carlo", "reality_check", "data_quality_a", "data_quality_b", "overall_pass"):
                continue
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("**Acceptance gates:**")
        for gate, passed in r["gates"].items():
            lines.append(f"- [{'x' if passed else ' '}] {gate}")
        lines.append("")
        lines.append(f"**Overall: {'PASS' if r['overall_pass'] else 'NO-GO'}**")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["xsection_mean_reversion", "pairs_trading", "both"], default="both")
    parser.add_argument("--demo", action="store_true", help="Use offline synthetic data (no network required)")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--pair-a", default="XLE")
    parser.add_argument("--pair-b", default="XOP")
    args = parser.parse_args()

    results = []
    if args.strategy in ("xsection_mean_reversion", "both"):
        print("Running xsection_mean_reversion backtest...")
        results.append(run_xsection(args))
    if args.strategy in ("pairs_trading", "both"):
        print("Running pairs_trading backtest...")
        results.append(run_pairs(args))

    out_path = write_health_check(results)
    print(f"\nHealth check written to {out_path}")
    for r in results:
        print(f"  {r['strategy']}: {'PASS' if r['overall_pass'] else 'NO-GO'}")


if __name__ == "__main__":
    main()
