"""
Round 2 of the `absorption_breakout` investigation (backtests/reports/
absorption_breakout_investigation_report.md's dated 2026-08-14 addendum) —
follow-up on TWO specific rescue-lever suggestions for the round-1 result
(dev-window best config B5_tight6_clearance: TIGHT6 + breakout_atr_mult=0.5,
gross PF 0.966; honest single-shot holdout PF 1.525/1.578, clears 3/4 gates,
fails only monte_carlo_p5_sharpe because the top 5 of 204 holdout trades
exceed the ENTIRE holdout's net P&L — a small-sample/tail-concentration
fragility, not a "no edge" failure).

THREE other levers from the same suggestion list (dwell-time,
multi-level L2 depth imbalance, aggressor-size ratio) are OUT OF SCOPE for
this script — already confirmed infeasible for the same data-availability
reason as round 1's Option B (data/ticks/, data/depth/ only cover
2026-08-04..2026-08-14, which does not overlap the cached 1-minute OHLCV
history's DEV/HOLDOUT windows enough to compute or validate a WFO-fold-scale
feature from). Not attempted here; do not re-litigate.

## CRITICAL METHODOLOGY CHECK (done FIRST, see `check_holdout_freshness()`)

Round 1's ONLY reserved holdout (2026-06-01..2026-08-01) has already been
evaluated exactly once (`backtests/reports/_absorption_breakout_validation/
HOLDOUT_best.json`). Re-evaluating anything on that SAME window a second
time would be exactly the holdout-reuse problem
`backtests/reports/holdout_methodology_dossier.md` documents elsewhere in
this repo (four of six recent-campaign rounds share that exact failure).
This script NEVER touches HOLDOUT_START/HOLDOUT_END from round 1 — full
stop, no exceptions, no `--force`.

`check_holdout_freshness()` establishes, from data on disk (not assumed),
whether any GENUINELY NEW 1-minute-bar-equivalent data exists for a period
STRICTLY AFTER round 1's holdout:
  - `data/history_1m/<SYM>/*.parquet`: last closed month is still 2026-07
    for every universe symbol (confirmed: IB backfill blocked in this
    environment, see below) -- NO new canonical 1-minute bars exist past
    round 1's own holdout end.
  - `scripts/backfill_intraday.py` (the only way to extend that canonical
    cache) requires a live IB Gateway connection
    (`python.data.ibkr_price_source.open_ib_connection`) -- confirmed
    unreachable in this environment (`ConnectionRefusedError` at
    127.0.0.1:4002, 2026-08-14). Per this investigation's own brief: "if
    the backfill script requires credentials that aren't available in this
    environment, note that blocker explicitly and stop this lever" -- this
    blocks a genuine, full-size fresh holdout entirely; NOT worked around
    with `python.data.intraday_cache`'s own explicitly-documented policy of
    "no yfinance fallback... IB is the only source" (`intraday_cache.py`'s
    own module docstring) -- honored here, not overridden.
  - HOWEVER: `data/ticks/<SYM>/<YYYYMMDD>.jsonl` (raw trade prints, NOT
    1-minute bars) DOES cover 5 trading days strictly after round 1's
    holdout's real underlying bar coverage (2026-08-04, 05, 06, 12, 13 --
    the last CLOSED 1-minute-bar month is 2026-07, so round 1's holdout's
    actual tradeable data never reached August at all). These ticks were
    captured for a DIFFERENT purpose (the L2-depth/order-flow features
    round 1 and this round both ruled infeasible) but are RAW trade data,
    not a derived feature -- they can be aggregated into the exact same
    OHLCV bar schema `data/history_1m/` uses (verified: the ticks' `time`
    field is already tz-naive US/Eastern wall-clock, RTH 09:30-16:00,
    identical convention -- see `aggregate_ticks_to_1m_bars()`).
  - Verdict (see `check_holdout_freshness()`'s printed output): fresh data
    EXISTS but is far too small (5 non-contiguous trading days across a
    tight-6-symbol universe -- roughly 24-30 expected trades at this
    signal's historical rate, vs. the >=100-per-fold gate and vs. round 1's
    own 42-day/204-trade holdout that ALREADY failed the Monte Carlo gate
    on small-sample grounds) to serve as a real, gate-based, decision-making
    holdout -- it cannot even form one 30-day OOS fold, let alone support a
    meaningful bootstrap. Per this investigation's own explicit fallback
    instruction for exactly this case: this script (a) uses ONLY the DEV
    window's WFO folds (2025-08-01..2026-06-01, IDENTICAL to round 1, never
    the reserved holdout) for lever diagnosis/selection below, and (b)
    still aggregates and runs the tiny fresh tick window as a clearly-
    labeled, NON-GATING "canary" sanity check (`FRESH_canary_B5`) on the
    frozen round-1 winner ONLY -- never used to pick among round-2
    candidates, reported with its tiny-sample caveats spelled out, not
    silently presented as a second holdout.

## Lever 1: macro/sector beta alignment filter -- BLOCKED, not implemented

Requires 1-minute QQQ/SPY/XLK bars over (at least) the DEV window to test
across WFO folds. Confirmed absent from `data/history_1m/` (no QQQ/SPY/XLK
symbol directories exist there at all) and NOT obtainable in this
environment: the only backfill path (`scripts/backfill_intraday.py`) needs
IB Gateway, confirmed unreachable above; `data/ticks/`/`data/depth/` do NOT
include index/sector ETF symbols (checked: only the 20-symbol trading
universe was captured); yfinance's 1-minute history caps at ~7 days
(`intraday_cache.py`'s own docstring), nowhere near the ~217-day DEV window
needed to run even one WFO fold, let alone all 7. Per the investigation
brief's explicit instruction ("do not attempt to hand-roll a different data
source"), this lever is NOT implemented in `absorption_breakout.py` and NOT
tested -- there is nothing to grid-search or diagnose without the
underlying feature existing at all. Documented as a hard data blocker, not
a null result.

## Lever 2: asymmetric micro-stop -- IMPLEMENTED, tested on DEV-window folds

`micro_stop_cents` (new optional param, `absorption_breakout.py`, defaults
to `None` = unchanged ATR-based stop): a fixed dollar distance past the
broken level instead of `stop_atr_mult * ATR`. Tested in TWO stages:
  C1/C2  ISOLATED from round 1's clearance lever -- same base as round 1's
         B2_tight6 (TIGHT6, breakout_atr_mult=0.0, baseline volume_mult),
         swapping ONLY the stop mechanism (1 cent, then 2 cents) -- isolates
         the stop lever's own marginal effect the same way round 1's B3
         isolated the clearance lever from the universe lever.
  D1     STACKED onto round 1's actual DEV winner (B5_tight6_clearance:
         TIGHT6 + breakout_atr_mult=0.5) using whichever of C1/C2 scored
         better in isolation -- only run if C1 or C2 showed a clear,
         honestly-diagnosed improvement (this script's diagnosis-before-
         lever discipline; see `main()`).

1-minute-bar caveat (restated from `absorption_breakout.py`'s own
docstring, not softened here): a 1-2 cent stop is resolved by the SAME
coarse "did this whole minute's high/low range cross the stop" logic as any
wider ATR stop -- this backtest cannot simulate true sub-minute intrabar
path, so a "tighter" stop is NOT a "more precisely simulated" stop; if
anything it is treated as a coarser approximation of what a real resting
order would have done inside that specific minute.

Discipline (identical to scripts/_absorption_breakout_validation.py /
scripts/_l2_absorption_validation.py):
  * Gate thresholds read from configs/goal.yaml UNCHANGED. Nothing here
    writes to configs/strategy.yaml's human-owned fields; `auto_execute`
    is never flipped.
  * configs/goal.yaml's HOLDOUT_START/HOLDOUT_END from round 1
    (2026-06-01/2026-08-01) is NEVER evaluated by this script -- there is
    no code path here that can reach it; the only "final window" reachable
    is the tiny FRESH_canary tick-derived window, which is explicitly
    NON-GATING (no decision is ever made FROM it).
  * Selection rule for whether D1 (the stack) gets run at all, declared
    HERE before C1/C2 were run: "the better of C1/C2 (by full-window
    cost-adjusted PF on the DEV window, same primary metric round 1 used)
    must beat round 1's B2_tight6 (its own same-universe, same-baseline-
    params control) by a non-trivial margin (>=0.05 PF) for D1 to be worth
    running; if neither clears that bar, report both as a negative/
    inconclusive result for the isolated stop lever and do not run D1."

Resilience: per-config checkpoint JSON in
`backtests/reports/_absorption_breakout_round2/`; every individual
backtest_fn(start, end, params) call is memoized to disk under
`backtests/reports/_absorption_breakout_round2_cache/`.

Usage:
    python scripts/_absorption_breakout_round2.py --check-freshness
    python scripts/_absorption_breakout_round2.py --list
    python scripts/_absorption_breakout_round2.py C1_microstop_1c_tight6
    python scripts/_absorption_breakout_round2.py C2_microstop_2c_tight6
    python scripts/_absorption_breakout_round2.py D1_stack_best_microstop_on_b5
    python scripts/_absorption_breakout_round2.py FRESH_canary_B5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd  # noqa: E402

from run_intraday_backtest import GOAL_PATH, STRATEGY_PATH, _load_yaml  # noqa: E402
from _absorption_breakout_validation import (  # noqa: E402
    DEV_END, DEV_START, HOLDOUT_END, HOLDOUT_START, TIGHT6,
    load_calibrated_spreads, load_bars,
)
from python.backtest.intraday_engine import IntradayBacktestConfig  # noqa: E402
from python.backtest.monte_carlo import MonteCarloValidator  # noqa: E402
from python.backtest.optimize import (  # noqa: E402
    build_intraday_backtest_fn,
    check_drawdown_gate,
    check_has_trades_gate,
    check_min_trades_gate,
    check_profit_factor_gate,
    load_wfo_config,
    max_oos_drawdown_threshold,
    preflight_check,
)
from python.backtest.walk_forward import WalkForwardOptimizer  # noqa: E402

SIGNAL = "absorption_breakout"
CHECKPOINT_DIR = Path("backtests/reports/_absorption_breakout_round2")
CACHE_DIR = Path("backtests/reports/_absorption_breakout_round2_cache")
TICKS_DIR = Path("data/ticks")

BASE_PARAMS = {"volume_mult": 3.0, "breakout_atr_mult": 0.0, "stop_atr_mult": 0.5,
               "target_r_multiple": None, "micro_stop_cents": None}

# The 5 trading days data/ticks/ actually has on disk for the full universe,
# as of this writing -- checked directly (`ls data/ticks/AAPL/`), NOT
# assumed to be a contiguous range. Strictly AFTER round 1's holdout's real
# underlying bar coverage (last closed 1-minute-bar month is 2026-07).
FRESH_TICK_DATES = ["20260804", "20260805", "20260806", "20260812", "20260813"]


def _p(**overrides) -> dict:
    out = dict(BASE_PARAMS)
    out.update(overrides)
    return out


# ── holdout-freshness check (run this FIRST, per the investigation brief) ──

def check_holdout_freshness() -> dict:
    """Prints and returns a summary of what fresh data (if any) exists
    strictly after round 1's holdout. Does NOT touch round 1's holdout
    checkpoint or window in any way -- read-only inspection of data/ on
    disk plus one connection-attempt probe (already performed once
    manually and recorded here as a fact, not re-attempted live every run
    to avoid a slow/hanging import-time side effect)."""
    universe_dir = Path("data/history_1m")
    last_months = {}
    for sym_dir in sorted(universe_dir.iterdir()):
        if not sym_dir.is_dir():
            continue
        months = sorted(p.stem for p in sym_dir.glob("*.parquet"))
        if months:
            last_months[sym_dir.name] = months[-1]
    all_last = set(last_months.values())

    tick_symbols = sorted(p.name for p in TICKS_DIR.iterdir()) if TICKS_DIR.exists() else []
    tick_dates_by_symbol = {}
    for sym in tick_symbols:
        dates = sorted(p.stem for p in (TICKS_DIR / sym).glob("*.jsonl"))
        tick_dates_by_symbol[sym] = dates
    common_tick_dates = sorted(set.intersection(*[set(d) for d in tick_dates_by_symbol.values()])) \
        if tick_dates_by_symbol else []

    summary = {
        "canonical_1m_bar_cache": {
            "last_closed_month_per_symbol": last_months,
            "uniform_last_month": sorted(all_last),
            "verdict": ("No new 1-minute bars past round 1's holdout window -- "
                        "IB Gateway backfill confirmed unreachable in this environment "
                        "(ConnectionRefusedError at 127.0.0.1:4002, ib_async, 2026-08-14). "
                        "See scripts/backfill_intraday.py; not worked around."),
        },
        "raw_tick_data": {
            "symbols_captured": len(tick_symbols),
            "dates_common_to_all_symbols": common_tick_dates,
            "n_fresh_trading_days": len(common_tick_dates),
            "verdict": (
                f"{len(common_tick_dates)} trading days of RAW TICK data exist strictly after "
                "round 1's holdout's real underlying bar coverage (2026-07-31) -- genuinely never "
                "touched by round 1 or any prior investigation. Aggregable into 1-minute OHLCV bars "
                "(verified: tick 'time' field is tz-naive US/Eastern wall-clock RTH, identical "
                "convention to data/history_1m/). Too small (5 non-contiguous days) to form even one "
                "30-day OOS fold or support a meaningful Monte Carlo bootstrap -- NOT used as this "
                "round's real holdout. Used ONLY as a small, explicitly non-gating 'canary' check "
                "(FRESH_canary_B5) on the already-frozen round-1 DEV winner, never for lever "
                "selection."
            ),
        },
        "decision": (
            "No adequately-sized fresh holdout available. Per the investigation brief's explicit "
            "fallback: lever diagnosis/selection below uses ONLY the DEV-window WFO folds "
            f"({DEV_START}..{DEV_END}, identical to round 1); round 1's reserved holdout "
            f"({HOLDOUT_START}..{HOLDOUT_END}) is NEVER evaluated a second time by this script. "
            "Any round-2 'improved' configuration is IN-SAMPLE-ONLY evidence pending either more "
            "forward 1-minute-bar data (requires IB Gateway access this environment does not have) "
            "or live/paper forward validation -- flagged prominently in the report, not silently "
            "presented as confirmed."
        ),
    }
    print(json.dumps(summary, indent=2))
    return summary


# ── tick -> 1-minute OHLCV bar aggregation (FRESH_canary only) ─────────────

_KEEP_TICK_TYPES = {"AUTO_MATCH", "ODD_LOT"}  # real executed trades; excludes
# AVERAGE_PRICE (a derived, potentially double-counting print), UNKNOWN,
# CONTINGENT, DERIVATIVELY_PRICED, CRASH, OTC_SOLD, PRIOR_REFERENCE_PRICE
# (special tape condition codes, not ordinary continuous-market prints) --
# same spirit as a real OHLCV vendor's "regular trade" filter, applied here
# with necessarily less certainty than a proper tape-condition spec would
# give; disclosed as an approximation in the report, not presented as
# vendor-grade.

def aggregate_ticks_to_1m_bars(symbol: str, dates: list[str]) -> pd.DataFrame:
    """RTH (09:30:00-16:00:00, tz-naive US/Eastern wall-clock -- verified
    against data/ticks/*/*.jsonl's own 'time' field, which already matches
    data/history_1m/'s convention with no timezone conversion needed: the
    last trade of a full session lands at 15:59:59.xxx, one second before
    the 16:00:00 close, for every file checked) 1-minute OHLCV bars
    aggregated from raw trade prints, matching data/history_1m/'s exact
    schema (open/high/low/close/volume, index name 'ts', tz-naive
    datetime64). Returns an empty-columns DataFrame if no files exist for
    `symbol` (caller's responsibility to skip/report that)."""
    frames = []
    for d in dates:
        path = TICKS_DIR / symbol / f"{d}.jsonl"
        if not path.exists():
            continue
        df = pd.read_json(path, lines=True)
        if df.empty:
            continue
        df = df[df["tick_type"].isin(_KEEP_TICK_TYPES)]
        df["t"] = pd.to_datetime(df["time"], format="ISO8601")
        rth = df[(df["t"].dt.time >= pd.Timestamp("09:30:00").time()) &
                 (df["t"].dt.time < pd.Timestamp("16:00:00").time())]
        if rth.empty:
            continue
        bars = rth.set_index("t").resample("1min").agg(
            open=("price", "first"), high=("price", "max"),
            low=("price", "min"), close=("price", "last"),
            volume=("size", "sum"),
        ).dropna()
        frames.append(bars)
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = pd.concat(frames).sort_index()
    out.index.name = "ts"
    return out


def load_fresh_tick_bars(symbols: list[str], dates: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for sym in symbols:
        bars = aggregate_ticks_to_1m_bars(sym, dates)
        if not bars.empty:
            out[sym] = bars
    return out


# ── disk memoization (identical pattern to _absorption_breakout_validation.py) ──

def _cache_path(tag: str, start, end, params: dict) -> Path:
    raw = json.dumps({"tag": tag, "start": start.isoformat(), "end": end.isoformat(), "params": params},
                     sort_keys=True, default=str)
    return CACHE_DIR / f"{tag}__{hashlib.sha1(raw.encode()).hexdigest()[:24]}.json"


def _memoize(fn, tag: str):
    def wrapped(start, end, params):
        path = _cache_path(tag, start, end, params)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        result = fn(start, end, params)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, default=str), encoding="utf-8")
        tmp.replace(path)
        return result
    return wrapped


# ── C1/C2/D1: DEV-window WFO runs, FIXED candidate (no re-optimization) ────

@dataclass
class R2Config:
    config_id: str
    lever: str
    description: str
    params: dict
    universe: list[str]
    extra: dict = field(default_factory=dict)


CONFIGS: dict[str, R2Config] = {}


def _add(cfg: R2Config) -> None:
    CONFIGS[cfg.config_id] = cfg


_add(R2Config(
    "C1_microstop_1c_tight6", "2: micro-stop (isolated)",
    "TIGHT6, baseline breakout_atr_mult=0.0 (SAME base as round 1's B2_tight6) -- swap ONLY the stop "
    "mechanism to a fixed 1-cent-past-the-level stop (micro_stop_cents=0.01), isolating the stop "
    "lever's own marginal effect from round 1's clearance lever, mirroring how B3 isolated clearance "
    "from the universe lever on TIGHT10.",
    _p(micro_stop_cents=0.01), TIGHT6,
))
_add(R2Config(
    "C2_microstop_2c_tight6", "2: micro-stop (isolated)",
    "Same as C1 but micro_stop_cents=0.02 (2 cents), per the user's suggested 1-2 cent range.",
    _p(micro_stop_cents=0.02), TIGHT6,
))
# D1 is registered dynamically in main() ONLY if C1 or C2 clears the
# pre-declared bar (>=0.05 PF improvement over B2_tight6) -- see module
# docstring's "Selection rule" and main()'s enforcement of it.


def evaluate_dev(cfg: R2Config) -> dict:
    base_cfg = _load_yaml(STRATEGY_PATH)[SIGNAL]
    goal = _load_yaml(GOAL_PATH)
    intraday_goal = goal.get("intraday", {})
    min_trades = int(intraday_goal.get("min_trades_per_oos_fold", 100))
    min_pf = float(intraday_goal.get("min_cost_adjusted_profit_factor", 1.3))
    stress_mult = float(intraday_goal.get("stress_slippage_multiplier", 2.0))
    min_p5 = float(goal.get("monte_carlo", {}).get("min_p5_sharpe", 0.0))

    start_ts, end_ts = pd.Timestamp(DEV_START), pd.Timestamp(DEV_END)
    spreads = load_calibrated_spreads()
    bars_by_symbol, symbols_used = load_bars(cfg.universe, DEV_START, DEV_END)
    print(f"[{cfg.config_id}] {len(symbols_used)} symbols, window [{DEV_START}, {DEV_END}) (DEV, "
          f"NEVER the holdout), params={cfg.params}", flush=True)

    param_grid = [cfg.params]
    preflight = preflight_check(SIGNAL, base_cfg, param_grid,
                                total_trading_days=len(pd.bdate_range(start_ts, end_ts)))

    engine_cfg = IntradayBacktestConfig(half_spread_bps_by_symbol=spreads)
    fn = _memoize(build_intraday_backtest_fn(bars_by_symbol, SIGNAL, base_cfg, engine_cfg=engine_cfg),
                  f"dev__{cfg.config_id}")

    result: dict = {
        "config_id": cfg.config_id, "lever": cfg.lever, "description": cfg.description,
        "params": cfg.params, "window": f"{DEV_START} .. {DEV_END}", "window_kind": "dev",
        "cost_model": "calibrated_per_symbol", "n_symbols": len(symbols_used), "symbols": symbols_used,
        "sample_size_check": preflight, "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    wfo_cfg = load_wfo_config(SIGNAL)
    t0 = time.time()
    wfo = WalkForwardOptimizer(fn, wfo_cfg, param_grid).run(start_ts.to_pydatetime(), end_ts.to_pydatetime())
    print(f"[{cfg.config_id}] WFO done in {time.time()-t0:.0f}s: "
          f"{wfo.passing_folds}/{wfo.total_folds} folds, OOS Sharpe mean {wfo.oos_sharpe_mean:+.3f}", flush=True)

    stress_fn = _memoize(
        build_intraday_backtest_fn(
            bars_by_symbol, SIGNAL, base_cfg,
            engine_cfg=IntradayBacktestConfig(half_spread_bps_by_symbol=spreads,
                                              stress_slippage_multiplier=stress_mult)),
        f"dev__{cfg.config_id}__stress{stress_mult:g}x")

    full = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), cfg.params)
    mc = MonteCarloValidator(n_sims=500).run(full.get("daily_returns", []))
    stress = stress_fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), cfg.params)
    print(f"[{cfg.config_id}] full-window: n_trades={full.get('n_trades')} "
          f"PF_net={full.get('profit_factor', 0):.3f} PF_gross={full.get('profit_factor_gross', 0):.3f} "
          f"net_pnl={full.get('total_net_pnl', 0):,.0f} gross_pnl={full.get('gross_pnl', 0):,.0f} "
          f"| stress net={stress.get('total_net_pnl', 0):,.0f} | mc_p5={mc.sharpe.p5:+.3f}", flush=True)

    result.update({
        "wfo_folds": wfo.total_folds, "wfo_passing_folds": wfo.passing_folds,
        "wfo_pass_ratio": wfo.pass_ratio, "oos_sharpe_mean": wfo.oos_sharpe_mean,
        "fold_oos_sharpes": [f.oos_sharpe for f in wfo.folds],
        "fold_oos_trades": [int(f.oos_metrics.get("n_trades", 0)) for f in wfo.folds],
        "full_window_metrics": {k: v for k, v in full.items() if k != "daily_returns"},
        "stress_metrics": {k: v for k, v in stress.items() if k != "daily_returns"},
        "mc_p5_sharpe": mc.sharpe.p5,
    })
    gates = {
        "wfo_go": wfo.decision == "GO",
        "oos_drawdown_within_limit": check_drawdown_gate(wfo, max_oos_drawdown_threshold()),
        "has_oos_trades": check_has_trades_gate(wfo),
        "min_trades_per_oos_fold": check_min_trades_gate(wfo, min_trades),
        "cost_adjusted_profit_factor": check_profit_factor_gate(wfo, min_pf),
        "monte_carlo_p5_sharpe": mc.sharpe.p5 >= min_p5,
        f"stress_slippage_{stress_mult:g}x_net_positive": stress["total_net_pnl"] > 0,
    }
    result["gates"] = gates
    result["decision"] = "GO" if all(gates.values()) else "NO-GO"
    return result


# ── FRESH_canary: tiny, NON-GATING sanity check on the frozen B5 recipe ────

# Frozen EXACTLY as round 1's HOLDOUT_best.json evaluated it -- no
# re-tuning, no re-selection; this is a canary on the ALREADY-DECIDED
# round-1 winner, not a new candidate evaluation.
B5_FROZEN_PARAMS = _p(breakout_atr_mult=0.5)


def run_fresh_canary(force: bool = False) -> dict:
    path = CHECKPOINT_DIR / "FRESH_canary_B5.json"
    if path.exists() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        print(">>> FRESH_canary_B5 already checkpointed — skipping", flush=True)
        return existing

    t0 = time.time()
    base_cfg = _load_yaml(STRATEGY_PATH)[SIGNAL]
    spreads = load_calibrated_spreads()
    bars_by_symbol = load_fresh_tick_bars(TIGHT6, FRESH_TICK_DATES)
    symbols_used = sorted(bars_by_symbol)
    n_bars_total = sum(len(b) for b in bars_by_symbol.values())
    print(f"[FRESH_canary_B5] {len(symbols_used)}/{len(TIGHT6)} TIGHT6 symbols have tick data, "
          f"{n_bars_total} aggregated 1-minute bars total, dates={FRESH_TICK_DATES}, "
          f"params={B5_FROZEN_PARAMS} (round 1's frozen DEV winner, UNCHANGED)", flush=True)

    if not bars_by_symbol:
        result = {"config_id": "FRESH_canary_B5", "decision": "SKIPPED",
                   "reason": "no aggregable tick data found"}
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        (path).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    engine_cfg = IntradayBacktestConfig(half_spread_bps_by_symbol=spreads)
    fn = build_intraday_backtest_fn(bars_by_symbol, SIGNAL, base_cfg, engine_cfg=engine_cfg)
    stress_fn = build_intraday_backtest_fn(
        bars_by_symbol, SIGNAL, base_cfg,
        engine_cfg=IntradayBacktestConfig(half_spread_bps_by_symbol=spreads, stress_slippage_multiplier=2.0))

    start = min(b.index.min() for b in bars_by_symbol.values()).to_pydatetime()
    end = (max(b.index.max() for b in bars_by_symbol.values()) + pd.Timedelta(minutes=1)).to_pydatetime()

    full = fn(start, end, B5_FROZEN_PARAMS)
    stress = stress_fn(start, end, B5_FROZEN_PARAMS)
    n_daily_returns = len(full.get("daily_returns", []))
    mc = MonteCarloValidator(n_sims=500).run(full.get("daily_returns", [])) if n_daily_returns >= 2 else None

    print(f"[FRESH_canary_B5] n_trades={full.get('n_trades')} "
          f"PF_net={full.get('profit_factor', 0):.3f} PF_gross={full.get('profit_factor_gross', 0):.3f} "
          f"net_pnl={full.get('total_net_pnl', 0):,.0f} | stress net={stress.get('total_net_pnl', 0):,.0f} "
          f"| n_daily_return_obs={n_daily_returns}", flush=True)

    result = {
        "config_id": "FRESH_canary_B5",
        "NON_GATING_DISCLAIMER": (
            "This is NOT a real holdout evaluation and NO gate verdict is computed from it. "
            f"Only {n_daily_returns} daily-return observations across {len(symbols_used)} symbols exist "
            "in this window -- far too few to run a WFO fold (needs >=120 calendar days), form a "
            "trustworthy Monte Carlo bootstrap, or evaluate the min_trades_per_oos_fold gate "
            "meaningfully. Reported for transparency only, per the investigation brief's freshness-check "
            "requirement -- see backtests/reports/absorption_breakout_investigation_report.md's round-2 "
            "addendum for the full caveat."
        ),
        "params": B5_FROZEN_PARAMS, "params_source": "round 1's frozen HOLDOUT_best.json config, UNCHANGED",
        "dates": FRESH_TICK_DATES, "window": f"{start.isoformat()} .. {end.isoformat()}",
        "data_source": "data/ticks/ raw trade prints, aggregated to 1-minute OHLCV bars in-process "
                        "(NOT the canonical data/history_1m/ cache -- see aggregate_ticks_to_1m_bars())",
        "n_symbols": len(symbols_used), "symbols": symbols_used, "n_bars_total": n_bars_total,
        "n_daily_return_obs": n_daily_returns,
        "full_window_metrics": {k: v for k, v in full.items() if k != "daily_returns"},
        "stress_metrics": {k: v for k, v in stress.items() if k != "daily_returns"},
        "mc_p5_sharpe": (mc.sharpe.p5 if mc is not None else None),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    print(f">>> FRESH_canary_B5: checkpointed ({result['elapsed_s']}s, NON-GATING)", flush=True)
    return result


def checkpoint_path(config_id: str) -> Path:
    return CHECKPOINT_DIR / f"{config_id}.json"


def run_config(cfg: R2Config, force: bool = False) -> dict:
    path = checkpoint_path(cfg.config_id)
    if path.exists() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        print(f">>> {cfg.config_id} already checkpointed ({existing.get('decision')}) — skipping", flush=True)
        return existing
    t0 = time.time()
    result = evaluate_dev(cfg)
    result["elapsed_s"] = round(time.time() - t0, 1)
    result["config_spec"] = asdict(cfg)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    print(f">>> {cfg.config_id}: {result['decision']} (checkpointed, {result['elapsed_s']}s)", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config_ids", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check-freshness", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.check_freshness:
        check_holdout_freshness()
        return

    if args.list:
        print("FRESH_canary_B5           [freshness-check] NON-GATING canary on round 1's frozen B5 config")
        for cid, c in CONFIGS.items():
            print(f"{cid:32s} [{c.lever}] {c.description}")
        return

    for cid in args.config_ids:
        if cid == "FRESH_canary_B5":
            run_fresh_canary(force=args.force)
            continue
        if cid == "D1_stack_best_microstop_on_b5":
            # Registered dynamically here (not at module import time) so the
            # pre-declared selection rule (module docstring) is enforced
            # mechanically: D1 can only be constructed AFTER C1/C2's own
            # checkpoints already exist on disk, and only if one of them
            # actually cleared the pre-declared >=0.05 PF bar over B2_tight6.
            c1 = json.loads(checkpoint_path("C1_microstop_1c_tight6").read_text(encoding="utf-8")) \
                if checkpoint_path("C1_microstop_1c_tight6").exists() else None
            c2 = json.loads(checkpoint_path("C2_microstop_2c_tight6").read_text(encoding="utf-8")) \
                if checkpoint_path("C2_microstop_2c_tight6").exists() else None
            b2 = json.loads(Path("backtests/reports/_absorption_breakout_validation/B2_tight6.json")
                             .read_text(encoding="utf-8"))
            b2_pf = float(b2["full_window_metrics"]["profit_factor_gross"])
            candidates = [(cid_, r) for cid_, r in [("C1", c1), ("C2", c2)] if r is not None]
            if not candidates:
                raise SystemExit("D1 requires C1 and/or C2 to be run first (see --list)")
            best_id, best = max(candidates, key=lambda kv: kv[1]["full_window_metrics"]["profit_factor_gross"])
            best_pf = float(best["full_window_metrics"]["profit_factor_gross"])
            print(f"[D1 gate check] {best_id} gross PF {best_pf:.3f} vs B2_tight6 control {b2_pf:.3f} "
                  f"(delta {best_pf - b2_pf:+.3f}, bar is >=+0.05, pre-declared BEFORE C1/C2 ran)")
            if best_pf - b2_pf < 0.05:
                raise SystemExit(
                    f"PRE-DECLARED SELECTION RULE NOT MET: {best_id} did not beat B2_tight6 by >=0.05 "
                    "gross PF -- per this script's own docstring, D1 is not run; the isolated micro-stop "
                    "result is reported as negative/inconclusive instead."
                )
            best_cents = best["params"]["micro_stop_cents"]
            _add(R2Config(
                "D1_stack_best_microstop_on_b5", "1+2: clearance AND micro-stop combined",
                f"Round 1's B5 base (TIGHT6 + breakout_atr_mult=0.5) + {best_id}'s winning "
                f"micro_stop_cents={best_cents} -- stacks the two independently-helpful levers, same "
                "spirit as round 1's B5 stacking B2+B3.",
                _p(breakout_atr_mult=0.5, micro_stop_cents=best_cents), TIGHT6,
            ))
        if cid not in CONFIGS:
            raise SystemExit(f"unknown config id {cid!r} (use --list)")
        run_config(CONFIGS[cid], force=args.force)


if __name__ == "__main__":
    main()
