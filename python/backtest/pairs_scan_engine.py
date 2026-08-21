"""
Scanned-universe pairs trading: point-in-time pair selection + multi-pair
portfolio replay.

WHY THIS EXISTS
---------------
`python/backtest/engine.py`'s `run_pairs_backtest` trades exactly ONE
hardcoded pair. On real data (AMAT/LRCX, 1761 trading days) that produced
**8 trades in ~7 years** — far too few to separate edge from luck, which is
why the Monte Carlo bootstrap gate failed decisively (p5 Sharpe -0.903) and
the walk-forward pass ratio was 8%. See
`backtests/reports/strategy_review_summary.md` §2.1. The diagnosis there is
a rare-event / sample-size problem, not a cost problem, and the named next
experiment is: scan a universe of candidate pairs instead of hardcoding one.

This module is that experiment. It is an ADDITIVE, opt-in path — nothing in
`engine.py`, `run_backtest.py` or the live/dashboard wiring changes, and the
single-pair behavior is untouched. Entry/exit logic itself is unchanged: the
same `PairsTradingStrategy` and `PairPositionManager` decide every trade.
The only difference is WHICH pairs are eligible on any given day.

POINT-IN-TIME SELECTION — THE WHOLE BALLGAME
--------------------------------------------
Scanning a universe for cointegration over the FULL history and then
backtesting the winners is not a weak result, it is a fabricated one: the
scan's output is precisely "which relationships turned out to hold", so
trading them in the past is pure lookahead. Every safeguard below exists to
make that impossible rather than merely discouraged:

  1. `build_scan_schedule` is the ONLY place a cointegration test is run.
     For a scan dated `as_of = dates[j]` it passes `close.iloc[j - L : j]`
     to `python/stat/pair_scanner.scan` — an exclusive upper bound, so the
     as-of bar itself is not visible, let alone anything after it. This is
     the same discipline as `engine.py`'s `df.iloc[i - lookback : i]`,
     `orb_vwap_regime`'s `.shift(1)` regime label, and
     `intraday_engine.py`'s strictly-causal event loop.
  2. The scan schedule is anchored to GLOBAL positions in the full price
     index (`j = L, L + R, L + 2R, ...`), not to any backtest window's
     start. A walk-forward fold therefore cannot shift the scan cadence to
     a more convenient date, and the identical schedule is replayed by
     every fold and every parameter candidate.
  3. `run_scan_backtest` never calls `test_pair` at all. It can only read
     scan results whose `as_of` is <= the bar being traded
     (`_effective_scan_date`). A future scan is unreachable by construction.
  4. `tests/test_pairs_scan.py` pins all three of the above, including a
     test that fails if a future scan result ever becomes reachable and a
     test that corrupts all post-T prices and requires every pre-T trade to
     be bit-identical.

SELECTION RULE — MECHANICAL, IDENTICAL IN EVERY FOLD
----------------------------------------------------
At each scan date, in rank order and with no discretion anywhere:

  a. Candidate pairs = all within-bucket pairs from `configs/pairs_universe.yaml`
     (an a-priori economic bucketing, fixed before any backtest was run).
  b. Keep pairs passing `CointegrationResult.is_tradeable` (CADF stationary
     at the 5% critical value AND a positive half-life shorter than half the
     lookback) — the repo's pre-existing screen, no new threshold.
  c. Keep pairs whose half-life is inside
     [`min_half_life_days`, `max_half_life_days`] — the repo's pre-existing
     `configs/strategy.yaml` values, unchanged.
  d. Rank ascending by CADF t-statistic (most negative = strongest evidence
     of stationarity), which is exactly `pair_scanner.scan`'s own ordering.
  e. Walk the ranked list opening any pair whose |z| clears `entry_z`, until
     `max_concurrent_pairs` positions are open.

Note what is NOT here: no "pick the N best-performing pairs", no
backtest-informed shortlist, no per-fold hand-picking. The rank key is a
statistic computed from in-sample-at-the-time prices only.

FREE-PARAMETER ACCOUNTING (Chan Ch.3 ceiling = 5)
--------------------------------------------------
`configs/strategy.yaml:pairs_trading` already sits exactly AT the ceiling
with `entry_z`, `exit_z`, `half_life_multiplier_max_hold`,
`min_half_life_days`, `max_half_life_days`. This module therefore adds NO
tunable parameter and no key to that block:

  - Candidate universe / bucketing -> `configs/pairs_universe.yaml`, data
    about which instruments exist, the same status `configs/universe.yaml`
    has for `xsection_mean_reversion`.
  - Cointegration significance -> reuses `is_tradeable` (CADF 5%). No knob.
  - Ranking -> CADF t-statistic. No knob.
  - `max_concurrent_pairs` -> a CAPITAL constraint, not a signal threshold,
    and pinned a-priori at `MAX_CONCURRENT_PAIRS` below by an arithmetic
    argument that never looks at returns (see that constant's comment). It
    is never gridded and never varied between folds.
  - `half_spread_bps` -> a COST assumption, deliberately conservative;
    varied only in the mandatory adverse-cost stress re-run, never
    optimized.

EXIT-RULE ABLATIONS (2026-08-13)
--------------------------------
Round one of this study concluded NO-GO and localized the failure: 91-96% of
positions exited on the STALE TIMEOUT rather than on z-reversion, so the
half-life-derived max-hold was closing trades at an arbitrary point. Three
opt-in exit rules were added to test whether that is fixable — dynamic
half-life re-estimation, a cointegration-breakdown exit, and a z-widening
stop. See `PairsScanConfig` below for each, and `pairs_scan_report.md`
§2026-08-13 for the measured answer (short version: no, and they stay off).
All three default to OFF; a default-constructed config is round one's engine.

COSTS
-----
Unlike `engine.py` (which passes neither ADV nor a spread, so it prices
neither market impact nor the bid-ask spread), this module charges, per leg,
per round trip: IBKR commission + SEC Section 31 + FINRA TAF + short borrow
+ square-root market impact against real 20-day dollar ADV + a bid-ask
half-spread on both fills. All of it via `python/core/fees_equity.py`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..core.fees_equity import round_trip_cost
from ..core.pair_position_manager import PairPositionManager
from ..core.strategies.pairs_trading import PairsTradingStrategy
from ..core.types import CointegrationResult, QualifiedSpreadOrder, SpreadSide
from ..stat.cointegration import current_spread, spread_z_score
from ..stat.pair_scanner import scan
from .engine import PairTrade

log = logging.getLogger(__name__)

PAIRS_UNIVERSE_PATH = Path("configs/pairs_universe.yaml")

# ── A-priori structural constants (NOT tuned, NOT gridded) ──────────────────

# Concurrent-position cap. Fixed by arithmetic, not by backtest outcome:
# `notional_per_leg` is $50,000 and every pair holds two legs, so 10 pairs is
# 10 x 2 x $50,000 = $1,000,000 gross — exactly 1.0x the $1,000,000 capital
# base used everywhere else in this repo (`optimize._CAPITAL`,
# `PairsBacktestReport.to_dict`), and the same 1.0x gross-leverage target
# `configs/strategy.yaml:xsection_mean_reversion.gross_leverage_target`
# already uses. Changing this changes leverage, not the signal.
MAX_CONCURRENT_PAIRS = 10

# Bid-ask half-spread assumed for every leg, in bps of traded notional.
# `backtests/reports/slippage_calibration_report.md` calibrates real captured
# half-spreads for 20 single-name stocks (0.32bps AAPL .. 6.59bps STX); NONE
# of them is an ETF, so no calibrated value applies to this universe. Real
# quoted half-spreads on large index ETFs are typically well under 1bp (SPY,
# QQQ, the SPDR sectors) and a few bps on the thinnest names here (SIL, GDXJ,
# ILF, UNG). 3.0bps is deliberately set ABOVE the plausible blended value so
# the cost assumption errs against the strategy; the mandatory stress re-run
# doubles it to 6.0bps.
DEFAULT_HALF_SPREAD_BPS = 3.0
STRESS_HALF_SPREAD_MULTIPLIER = 2.0

# Passed to pair_scanner.scan so the scan CACHE is independent of the
# half-life config; the configured [min, max] bounds are then applied at
# replay time by `select_active_pairs`. Storing the unfiltered result means
# a half-life-bound change never silently reuses a stale cache.
_SCAN_MIN_HALF_LIFE = 0.0
_SCAN_MAX_HALF_LIFE = float("inf")


# ── Exit-rule ablations (2026-08-13) — all DEFAULT OFF ──────────────────────
#
# The first round of this study found that 91-96% of positions exit on the
# stale timeout rather than on z-reversion, i.e. the half-life-derived
# max-hold fires at an arbitrary point instead of a considered one. These
# three constants parameterize the follow-up ablations. They are switched on
# only by `scripts/run_pairs_exit_ablations.py`; every other caller, and the
# `PairsScanConfig` defaults below, keep the original behavior exactly.

# Stop level as a multiple of `entry_z`, used ONLY when the z-widening stop is
# enabled. Pinned a-priori at the value named in the experiment brief; never
# gridded, never fitted (`tests/test_pairs_exit_rules.py::
# test_param_grid_never_grids_an_exit_ablation_constant`). Enabling the stop
# at all is a departure from the documented no-stops design and requires
# human sign-off — see `pair_position_manager`'s module docstring.
STOP_Z_MULTIPLE = 1.5


@dataclass
class PairsScanConfig:
    """Trading parameters. The first eight fields mirror
    `PairsBacktestConfig` exactly (same names, same defaults, same meaning)
    so the scanned and single-pair paths stay comparable; then the two
    structural constants documented above, then the exit-rule ablation
    switches — all off, so a default-constructed config is the original
    strategy."""
    entry_z: float = 2.0
    exit_z: float = 0.5
    coint_lookback_days: int = 252
    revalidate_every_days: int = 21
    notional_per_leg: float = 50_000.0
    half_life_multiplier_max_hold: float = 3.0
    min_half_life_days: float = 1.0
    max_half_life_days: float = 60.0
    max_concurrent_pairs: int = MAX_CONCURRENT_PAIRS
    half_spread_bps: float = DEFAULT_HALF_SPREAD_BPS

    # Ablation 1: re-derive an open position's max-hold from the freshest
    # point-in-time half-life estimate instead of freezing it at open time.
    # Not a parameter — it changes how `half_life_multiplier_max_hold` is
    # applied, not what it is.
    dynamic_half_life: bool = False
    # Ablation 2: close immediately when the pair stops passing the existing
    # `is_tradeable` screen. A boolean design option, not a threshold.
    coint_breakdown_exit: bool = False
    # Ablation 3: close when |z| blows through `entry_z * stop_z_multiple`.
    # POLICY CHANGE — contradicts the documented no-price-stops design and
    # needs human sign-off. `None` disables it.
    stop_z_multiple: float | None = None
    # Set False together with ablation 3 so the stop REPLACES the timeout and
    # the free-parameter count stays at 5 rather than becoming 6.
    stale_timeout_enabled: bool = True

    @property
    def stop_z(self) -> float | None:
        """Absolute |z| stop level. Derived from `entry_z`, never configured
        independently of it."""
        if self.stop_z_multiple is None:
            return None
        return abs(self.entry_z) * float(self.stop_z_multiple)


@dataclass
class PairsScanReport:
    trades: list = field(default_factory=list)
    daily_pnl: dict = field(default_factory=dict)      # {date: pnl}
    pairs_traded: set = field(default_factory=set)     # {(code_a, code_b)}
    scan_dates_used: list = field(default_factory=list)
    eligible_counts: dict = field(default_factory=dict)  # {scan_date: n_pairs_passing}
    # Positions still open when the replay window ended. They are NOT trades
    # and contribute no P&L, exactly as before — but an exit rule that holds
    # positions longer (or forever) inflates this instead of booking losses,
    # so it is surfaced rather than left implicit.
    open_at_end: int = 0

    @property
    def total_net_pnl(self) -> float:
        return sum(t.net_pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.net_pnl > 0) / len(self.trades)

    @property
    def profit_factor(self) -> float:
        """Cost-adjusted: `net_pnl` is already after commission, fees, borrow,
        impact and half-spread."""
        wins = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        losses = abs(sum(t.net_pnl for t in self.trades if t.net_pnl < 0))
        if losses > 0:
            return wins / losses
        return float("inf") if wins > 0 else 0.0

    @property
    def total_cost(self) -> float:
        return sum(t.cost for t in self.trades)

    def daily_returns_series(self, capital: float) -> pd.Series:
        idx = pd.DatetimeIndex(sorted(self.daily_pnl.keys()))
        return pd.Series([self.daily_pnl[d] / capital for d in idx], index=idx)

    def to_dict(self, capital: float = 1_000_000.0) -> dict:
        returns = self.daily_returns_series(capital)
        sharpe = 0.0
        if len(returns) >= 2 and returns.std(ddof=1) > 0:
            sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        equity = (1 + returns).cumprod() if len(returns) else pd.Series(dtype=float)
        max_dd = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
        return {
            "total_trades": len(self.trades),
            "total_net_pnl": self.total_net_pnl,
            "total_cost": self.total_cost,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "n_distinct_pairs_traded": len(self.pairs_traded),
            "exit_reasons": self.exit_reason_counts,
            "open_at_end": self.open_at_end,
        }

    @property
    def exit_reason_counts(self) -> dict:
        counts: dict[str, int] = {}
        for t in self.trades:
            counts[t.exit_reason] = counts.get(t.exit_reason, 0) + 1
        return dict(sorted(counts.items()))


# ── Candidate universe ──────────────────────────────────────────────────────

def load_pairs_universe(path: str | Path = PAIRS_UNIVERSE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {})["pairs_universe"]


def pairs_buckets(universe: dict) -> dict[str, list[str]]:
    """Return {bucket: [tickers]} from either the wrapped YAML
    (`computed_at` / `selection_rule` / `buckets`) or a legacy flat map."""
    raw = universe.get("buckets") if isinstance(universe.get("buckets"), dict) else universe
    return {k: list(v) for k, v in raw.items() if isinstance(v, (list, tuple))}


def candidate_pairs_from_buckets(buckets: dict[str, list[str]]) -> list[tuple[str, str]]:
    """All WITHIN-bucket pairs (a, b) with a < b; never across buckets.

    Reuses `pair_scanner.candidate_pairs_by_sector`, which already implements
    exactly this rule (it takes a {code: group} map) — the bucket file is
    just the inverted representation. Chan p.129: restricting candidates to
    economically related instruments is the primary defense against the
    multiple-comparison trap, since it is what keeps the number of tests
    small enough for a 5% critical value to mean anything.
    """
    from ..stat.pair_scanner import candidate_pairs_by_sector

    sector_map = {code: bucket for bucket, codes in buckets.items() for code in codes}
    return sorted(candidate_pairs_by_sector(sector_map))


# ── Point-in-time scan schedule ─────────────────────────────────────────────

def scan_positions(n_rows: int, lookback_days: int, revalidate_every_days: int) -> list[int]:
    """Row positions in the FULL price index at which a rescan happens.

    Anchored to the global index (`lookback, lookback + R, ...`), never to a
    backtest window's start — so every walk-forward fold and every parameter
    candidate replays the identical scan cadence, and no window can shift the
    cadence onto more convenient dates."""
    if revalidate_every_days <= 0:
        raise ValueError("revalidate_every_days must be positive")
    return list(range(lookback_days, n_rows, revalidate_every_days))


# Bump whenever the MEANING of a stored scan result changes (a different
# spread definition, a different eligibility screen, a different estimator).
# `build_scan_schedule` refuses to resume a checkpoint written under a
# different fingerprint rather than silently mixing incompatible results —
# the spread-mean/intercept fix in python/stat/cointegration.py is exactly the
# kind of change that would otherwise have been invisible on resume.
SCAN_SCHEMA_VERSION = 2


def _scan_fingerprint(candidate_pairs: list[tuple[str, str]],
                      lookback_days: int, revalidate_every_days: int,
                      index_signature: str) -> dict:
    return {
        "schema_version": SCAN_SCHEMA_VERSION,
        "lookback_days": int(lookback_days),
        "revalidate_every_days": int(revalidate_every_days),
        "n_candidate_pairs": len(candidate_pairs),
        "candidate_pairs_digest": hashlib.sha256(
            "|".join(f"{a},{b}" for a, b in candidate_pairs).encode()).hexdigest()[:16],
        "price_index": index_signature,
    }


def _result_to_json(r: CointegrationResult) -> dict:
    d = dict(r.__dict__)
    d["computed_at"] = r.computed_at.isoformat()
    return d


def _result_from_json(d: dict) -> CointegrationResult:
    d = dict(d)
    d["computed_at"] = datetime.fromisoformat(d["computed_at"])
    return CointegrationResult(**d)


def build_scan_schedule(
    close_panel: pd.DataFrame,
    candidate_pairs: list[tuple[str, str]],
    lookback_days: int = 252,
    revalidate_every_days: int = 21,
    checkpoint_path: str | Path | None = None,
    progress_every: int = 1,
) -> dict[pd.Timestamp, list[CointegrationResult]]:
    """Run the cointegration scan at every scheduled date, using ONLY prices
    strictly before that date.

    `close_panel` is a wide (date x symbol) adjusted-close frame covering the
    FULL history. For a scan dated `dates[j]` the window handed to
    `pair_scanner.scan` is `close_panel.iloc[j - lookback_days : j]` — an
    exclusive upper bound, so `dates[j]` itself is invisible to the test that
    decides what may be traded on `dates[j]`.

    `checkpoint_path` (JSONL, one line per scan date) makes the scan durable:
    a run interrupted after k scan dates resumes at k+1 and loses nothing.
    This is the expensive half of the whole pipeline (~one CADF fit per
    candidate pair per scan date), so it is computed ONCE per
    (universe, lookback, cadence) and then replayed by every fold and every
    parameter candidate.
    """
    dates = close_panel.index
    positions = scan_positions(len(dates), lookback_days, revalidate_every_days)

    schedule: dict[pd.Timestamp, list[CointegrationResult]] = {}
    done: set[str] = set()
    ckpt = Path(checkpoint_path) if checkpoint_path else None
    fingerprint = _scan_fingerprint(
        candidate_pairs, lookback_days, revalidate_every_days,
        f"{dates[0].date()}..{dates[-1].date()}:{len(dates)}")
    meta_path = ckpt.with_suffix(".meta.json") if ckpt is not None else None

    if ckpt is not None and ckpt.exists():
        on_disk = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        if on_disk != fingerprint:
            raise ValueError(
                f"{ckpt} was written under a different scan definition and cannot be "
                f"resumed (on disk: {on_disk}; requested: {fingerprint}). Delete the "
                "checkpoint and rescan — reusing it would mix incompatible results."
            )
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            as_of = pd.Timestamp(rec["as_of"])
            schedule[as_of] = [_result_from_json(r) for r in rec["results"]]
            done.add(rec["as_of"])
        log.info("build_scan_schedule: resumed %d/%d scan dates from %s",
                 len(done), len(positions), ckpt)

    if ckpt is not None:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")

    for n, j in enumerate(positions, start=1):
        as_of = dates[j]
        key = as_of.isoformat()
        if key in done:
            continue
        # Exclusive upper bound `j` — the as-of bar is NOT in the window.
        window = close_panel.iloc[j - lookback_days: j]
        results = scan(
            candidate_pairs, window,
            lookback_days=lookback_days,
            as_of=as_of.to_pydatetime(),
            min_half_life_days=_SCAN_MIN_HALF_LIFE,
            max_half_life_days=_SCAN_MAX_HALF_LIFE,
        )
        schedule[as_of] = results
        if ckpt is not None:
            with open(ckpt, "a", encoding="utf-8") as f:
                f.write(json.dumps({"as_of": key,
                                    "results": [_result_to_json(r) for r in results]}) + "\n")
        if progress_every and n % progress_every == 0:
            log.info("build_scan_schedule: %d/%d scan dates (as_of=%s, %d pairs passed)",
                     n, len(positions), as_of.date(), len(results))

    return schedule


def select_active_pairs(
    results: list[CointegrationResult],
    cfg: PairsScanConfig,
) -> list[CointegrationResult]:
    """Apply the configured half-life eligibility band and return the survivors
    ranked by CADF t-statistic (most negative first).

    `is_tradeable` was already enforced inside `pair_scanner.scan`; the
    half-life band is applied here so the on-disk scan cache stays valid
    across half-life-bound changes."""
    eligible = [
        r for r in results
        if cfg.min_half_life_days <= r.half_life_days <= cfg.max_half_life_days
    ]
    return sorted(eligible, key=lambda r: r.cadf_tstat)


# ── Replay ──────────────────────────────────────────────────────────────────

def _effective_scan_date(scan_dates: pd.DatetimeIndex, today: pd.Timestamp) -> pd.Timestamp | None:
    """Most recent scan date <= today. Guarantees a bar can only ever see a
    scan that had already happened when that bar traded — a scan dated after
    `today` is unreachable, which is what makes the no-lookahead property
    structural rather than a convention the replay loop must remember."""
    pos = int(scan_dates.searchsorted(today, side="right")) - 1
    return scan_dates[pos] if pos >= 0 else None


def run_scan_backtest(
    close_panel: pd.DataFrame,
    adv_panel: pd.DataFrame,
    scan_schedule: dict[pd.Timestamp, list[CointegrationResult]],
    config: PairsScanConfig | None = None,
    entry_gate: pd.Series | None = None,
) -> PairsScanReport:
    """Replay the daily close-to-close pairs portfolio over `close_panel`.

    Deliberately does NOT run any cointegration test: every estimate comes
    from `scan_schedule`, and only from a scan date that is <= the bar being
    traded. That is what makes the lookahead property structural rather than
    a convention this loop has to remember to honor.

    `adv_panel` is a wide (date x symbol) frame of 20-day dollar ADV, used
    for the square-root market-impact term. Entry/exit fill at that day's
    close, matching `engine.run_pairs_backtest`'s documented simplification.

    `entry_gate` (optional, default None = unchanged behavior for every
    existing caller): a boolean Series indexed by date. On a day where
    `entry_gate.get(today)` is False, NO NEW positions are opened (the
    entries loop is skipped for that day only) — existing open positions are
    still marked-to-market and still exit on their own configured rule
    exactly as if no gate were present. This mirrors the existing
    `max_concurrent_pairs` cap's own documented behavior ("blocks NEW
    entries, never force-liquidates") rather than inventing a new semantics.
    A day missing from `entry_gate` (e.g. before the gate has enough history
    to decide) is treated as gate-closed (`get(today, False)`), the same
    fail-closed default `python/analytics/trend_efficiency_gate.
    shifted_entry_gate` already uses for its own undecided rows.
    """
    cfg = config or PairsScanConfig()
    strategy = PairsTradingStrategy(entry_z=cfg.entry_z, exit_z=cfg.exit_z)
    pm = PairPositionManager(half_life_multiplier_max_hold=cfg.half_life_multiplier_max_hold)
    report = PairsScanReport()

    dates = close_panel.index
    scan_dates = pd.DatetimeIndex(sorted(d for d in scan_schedule if d <= dates[-1]))
    if not len(scan_dates):
        return report

    active: list[CointegrationResult] = []
    latest_by_pair: dict[tuple[str, str], CointegrationResult] = {}
    current_scan: pd.Timestamp | None = None
    entry_adv: dict[tuple[str, str], tuple[float, float]] = {}
    # Pairs the CURRENT point-in-time scan still certifies as tradeable. Read
    # from the raw scan results (the `is_tradeable` screen `pair_scanner.scan`
    # already applied) rather than from `active`, because `active` additionally
    # applies the configured half-life band — an entry-eligibility parameter,
    # which should not double as an exit trigger.
    tradeable_now: set[tuple[str, str]] = set()

    close_by_date = {d: r for d, r in zip(close_panel.index, close_panel.to_dict("records"))}
    adv_by_date = {d: r for d, r in zip(adv_panel.index, adv_panel.to_dict("records"))}

    for today in dates:
        effective = _effective_scan_date(scan_dates, today)
        if effective is None:
            continue                      # before the first point-in-time scan
        if effective != current_scan:
            current_scan = effective
            results = scan_schedule[effective]
            active = select_active_pairs(results, cfg)
            for r in active:
                latest_by_pair[(r.code_a, r.code_b)] = r
            tradeable_now = {(r.code_a, r.code_b) for r in results}
            report.scan_dates_used.append(effective)
            report.eligible_counts[effective] = len(active)

            # Ablation 1. Every estimate here comes from `scan_schedule` at a
            # scan date <= today, so re-estimation inherits the replay loop's
            # point-in-time property rather than needing its own guard.
            if cfg.dynamic_half_life:
                fresh = {(r.code_a, r.code_b): r.half_life_days for r in results}
                for pos in pm.open_positions:
                    hl = fresh.get((pos.code_a, pos.code_b))
                    if hl is not None:
                        pm.reestimate_half_life(pos.code_a, pos.code_b, hl)

        report.daily_pnl.setdefault(today, 0.0)
        # Plain dict, not the pandas row: this inner loop runs once per
        # candidate pair per day for the whole study, and Series.get is
        # roughly an order of magnitude slower than a dict lookup.
        row = close_by_date[today]

        # ── Exits (positions are held to their own exit even if the pair has
        # since dropped out of the eligible list — the cap below only blocks
        # NEW entries, it never force-liquidates). ───────────────────────────
        z_by_pair: dict[tuple[str, str], float] = {}
        for pos in pm.open_positions:
            key = (pos.code_a, pos.code_b)
            price_a, price_b = row.get(pos.code_a), row.get(pos.code_b)
            if price_a is None or price_b is None or pd.isna(price_a) or pd.isna(price_b):
                continue
            est = latest_by_pair.get(key)
            if est is not None and est.spread_std > 0:
                spread = current_spread(float(price_a), float(price_b), pos.hedge_ratio)
                z_by_pair[key] = spread_z_score(spread, est.spread_mean, est.spread_std)
            else:
                z_by_pair[key] = pos.entry_z      # no usable estimate — hold

        broken = ({key for key in (
            (p.code_a, p.code_b) for p in pm.open_positions) if key not in tradeable_now}
            if cfg.coint_breakdown_exit else None)

        for closed_pos, reason in pm.check_exits(
            z_by_pair, today, cfg.exit_z,
            stop_z=cfg.stop_z,
            broken_pairs=broken,
            stale_timeout_enabled=cfg.stale_timeout_enabled,
        ):
            key = (closed_pos.code_a, closed_pos.code_b)
            price_a, price_b = row.get(closed_pos.code_a), row.get(closed_pos.code_b)
            if price_a is None or price_b is None or pd.isna(price_a) or pd.isna(price_b):
                continue
            closed = pm.close_position(*key)
            holding_days = closed.holding_days(today)
            is_short_leg_a = closed.side == SpreadSide.SHORT_SPREAD
            adv_a, adv_b = entry_adv.pop(key, (0.0, 0.0))
            cost_a = round_trip_cost(
                closed.qty_a, closed.entry_price_a, float(price_a),
                is_short=is_short_leg_a, holding_days=holding_days,
                adv_dollars=adv_a, half_spread_bps=cfg.half_spread_bps).total
            cost_b = round_trip_cost(
                closed.qty_b, closed.entry_price_b, float(price_b),
                is_short=not is_short_leg_a, holding_days=holding_days,
                adv_dollars=adv_b, half_spread_bps=cfg.half_spread_bps).total
            gross = closed.unrealized_pnl(float(price_a), float(price_b))
            net = gross - cost_a - cost_b
            report.trades.append(PairTrade(
                code_a=closed.code_a, code_b=closed.code_b,
                entry_date=closed.entry_time, exit_date=today,
                side=closed.side.value, qty_a=closed.qty_a, qty_b=closed.qty_b,
                entry_price_a=closed.entry_price_a, entry_price_b=closed.entry_price_b,
                exit_price_a=float(price_a), exit_price_b=float(price_b),
                gross_pnl=gross, cost=cost_a + cost_b, net_pnl=net,
                exit_reason=reason,
            ))
            report.daily_pnl[today] += net

        # ── Entries, in CADF-rank order, until the concurrent cap. ───────────
        if entry_gate is not None and not bool(entry_gate.get(today, False)):
            continue
        adv_row = adv_by_date.get(today)
        n_open = len(pm.open_positions)
        for coint in active:
            if n_open >= cfg.max_concurrent_pairs:
                break
            key = (coint.code_a, coint.code_b)
            if pm.is_open(*key):
                continue
            price_a, price_b = row.get(coint.code_a), row.get(coint.code_b)
            if price_a is None or price_b is None or pd.isna(price_a) or pd.isna(price_b):
                continue
            price_a, price_b = float(price_a), float(price_b)
            if price_a <= 0 or price_b <= 0:
                continue
            signal = strategy.evaluate(coint, [], price_a, price_b, today)
            if signal is None:
                continue
            qty_a = int(cfg.notional_per_leg / price_a)
            qty_b = int((cfg.notional_per_leg * abs(coint.hedge_ratio)) / price_b)
            if qty_a <= 0 or qty_b <= 0:
                continue
            order = QualifiedSpreadOrder(
                raw=signal, qty_a=qty_a, qty_b=qty_b,
                gross_notional=qty_a * price_a + qty_b * price_b,
                estimated_cost=0.0, kelly_fraction_used=0.0, approved=True,
            )
            pm.open_position(order, price_a, price_b, today)
            n_open += 1
            report.pairs_traded.add(key)
            adv_a = adv_row.get(coint.code_a) if adv_row else None
            adv_b = adv_row.get(coint.code_b) if adv_row else None
            entry_adv[key] = (
                0.0 if adv_a is None or pd.isna(adv_a) else float(adv_a),
                0.0 if adv_b is None or pd.isna(adv_b) else float(adv_b),
            )

    report.open_at_end = len(pm.open_positions)
    return report
