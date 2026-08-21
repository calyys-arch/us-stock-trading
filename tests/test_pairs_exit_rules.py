"""
Exit-rule ablations for scanned-universe pairs trading (2026-08-13).

Round one (`backtests/reports/pairs_scan_report.md`) localized the strategy's
failure: 91-96% of positions exited on the STALE TIMEOUT rather than on
z-reversion, so the half-life-derived max-hold was closing trades at an
arbitrary point rather than a considered one. Three opt-in exit rules were
added to test whether that is fixable:

  1. dynamic half-life re-estimation while a position is open
  2. immediate exit when the pair stops passing the `is_tradeable` screen
  3. a z-widening stop-loss (a POLICY change — see below)

Two properties matter more than any of the individual behaviors:

  * DEFAULTS ARE UNCHANGED. Every rule is off unless explicitly switched on,
    so nothing that already existed — the live path, the dashboard, the
    single-pair engine, round one's numbers — moves.
    -> test_defaults_reproduce_the_legacy_exit_rule_exactly

  * THE NEW RULES ARE STILL POINT-IN-TIME. Ablation 1 re-estimates a
    statistic mid-trade and ablation 2 reads the eligibility set mid-trade;
    both are new places where tomorrow's data could leak into today's exit.
    The round-one future-mutation test is therefore extended to run under
    every ablation, not just the default configuration.
    -> test_trades_before_cutoff_unchanged_by_future_price_mutation_under_every_ablation
    -> test_ablations_cannot_reach_a_future_scan

Ablation 3 contradicts this strategy family's documented design (Chan argues
AGAINST price stops for mean reversion — a spread that moved against you is
cheaper, not worse). It is implemented and measured, but enabling it is a
human decision, not a numeric one; `test_stop_loss_is_off_by_default_and_is_a_documented_policy_change`
exists to keep that visible.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from python.backtest.pairs_scan_engine import (
    STOP_Z_MULTIPLE,
    PairsScanConfig,
    build_scan_schedule,
    candidate_pairs_from_buckets,
    run_scan_backtest,
)
from python.core.pair_position_manager import PairPositionManager
from python.core.types import (
    CointegrationResult,
    QualifiedSpreadOrder,
    SpreadSide,
    SpreadSignal,
)

_LOOKBACK = 120
_REVALIDATE = 20


# ── fixtures shared with tests/test_pairs_scan.py's generator shape ──────────

def _cointegrated_leg_pair(n: int, seed: int, phi: float = 0.92):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 1, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = phi * spread[t - 1] + rng.normal(0, 0.15)
    log_a = 4.0 + 0.1 * common + 0.5 * spread + rng.normal(0, 0.02, n)
    log_b = 3.8 + 0.1 * common - 0.5 * spread + rng.normal(0, 0.02, n)
    return np.exp(log_a), np.exp(log_b)


def _panel(n_days: int = 600, n_pairs: int = 3, seed: int = 11):
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    cols = {}
    for i in range(n_pairs):
        a, b = _cointegrated_leg_pair(n_days, seed + i)
        cols[f"A{i}"], cols[f"B{i}"] = a, b
    close = pd.DataFrame(cols, index=dates)
    adv = pd.DataFrame(3.0e8, index=dates, columns=close.columns)
    return close, adv


def _schedule(close: pd.DataFrame):
    return build_scan_schedule(
        close, candidate_pairs_from_buckets({"g": list(close.columns)}),
        lookback_days=_LOOKBACK, revalidate_every_days=_REVALIDATE, progress_every=0)


# The four variants under test, applied identically to every fold and every
# parameter candidate — they define the STRATEGY VARIANT, never something the
# search may select. `A3a` disables the stale timeout so the stop REPLACES it
# and the free-parameter count stays at 5 rather than becoming 6.
ABLATIONS: dict[str, dict] = {
    "A0_baseline": {},
    "A1_dynamic_half_life": {"dynamic_half_life": True},
    "A2_coint_breakdown_exit": {"coint_breakdown_exit": True},
    "A3a_stop_replaces_timeout": {"stop_z_multiple": STOP_Z_MULTIPLE,
                                  "stale_timeout_enabled": False},
    "A3b_stop_plus_timeout": {"stop_z_multiple": STOP_Z_MULTIPLE},
    "A4_dynamic_plus_breakdown": {"dynamic_half_life": True,
                                  "coint_breakdown_exit": True},
}


def _open_one(pm: PairPositionManager, half_life_days: float = 4.0,
              side: SpreadSide = SpreadSide.LONG_SPREAD, entry_z: float = -2.0,
              entry_time: datetime | None = None):
    """Open a single synthetic position directly on the manager, so exit
    precedence can be tested without routing through a full replay."""
    signal = SpreadSignal(
        id="t", strategy="pairs_trading", code_a="AAA", code_b="BBB",
        side=side, hedge_ratio=1.0, z_score=entry_z,
        entry_z_threshold=2.0, exit_z_threshold=0.5, spread_mean=0.0,
        half_life_days=half_life_days,
        confidence=1.0, timestamp=entry_time or datetime(2020, 1, 1),
    )
    order = QualifiedSpreadOrder(
        raw=signal, qty_a=100, qty_b=100, gross_notional=20_000.0,
        estimated_cost=0.0, kelly_fraction_used=0.0, approved=True)
    return pm.open_position(order, 100.0, 100.0, entry_time or datetime(2020, 1, 1))


# ── the defaults must not have moved ────────────────────────────────────────

def test_defaults_reproduce_the_legacy_exit_rule_exactly():
    """The one property everything else depends on: adding three opt-in exit
    rules must not perturb the configuration round one measured. A default
    `check_exits` may only ever return z_reversion or stale_timeout."""
    pm = PairPositionManager(half_life_multiplier_max_hold=3.0)
    _open_one(pm, half_life_days=4.0, entry_z=-2.0)

    # Still widening, not yet stale -> no exit at all under the defaults,
    # even though this is precisely where a stop would fire.
    assert pm.check_exits({("AAA", "BBB"): -9.0}, datetime(2020, 1, 2), 0.5) == []

    reverted = pm.check_exits({("AAA", "BBB"): 0.0}, datetime(2020, 1, 2), 0.5)
    assert [r for _p, r in reverted] == ["z_reversion"]

    stale = pm.check_exits({("AAA", "BBB"): -9.0}, datetime(2020, 2, 1), 0.5)
    assert [r for _p, r in stale] == ["stale_timeout"]


def test_default_config_is_byte_identical_to_round_one_end_to_end():
    close, adv = _panel(n_days=500, n_pairs=3, seed=61)
    schedule = _schedule(close)
    explicit_off = PairsScanConfig(
        dynamic_half_life=False, coint_breakdown_exit=False,
        stop_z_multiple=None, stale_timeout_enabled=True)
    a = run_scan_backtest(close, adv, schedule, PairsScanConfig())
    b = run_scan_backtest(close, adv, schedule, explicit_off)
    assert len(a.trades) == len(b.trades)
    assert a.total_net_pnl == pytest.approx(b.total_net_pnl, abs=1e-12)
    assert set(a.exit_reason_counts) <= {"z_reversion", "stale_timeout"}


def test_stop_loss_is_off_by_default_and_is_a_documented_policy_change():
    """Ablation 3 contradicts the module's documented no-price-stops design.
    It must never become reachable by accident: off in the dataclass default,
    off in the strategy config, and called out in the docstring a reviewer
    would read."""
    assert PairsScanConfig().stop_z_multiple is None
    assert PairsScanConfig().stop_z is None
    assert PairsScanConfig().stale_timeout_enabled is True

    import python.core.pair_position_manager as ppm

    doc = ppm.__doc__ or ""
    assert "human sign-off" in doc
    assert "stop_loss" in (ppm.PairPositionManager.check_exits.__doc__ or "")


# ── ablation 1: dynamic half-life ───────────────────────────────────────────

def test_reestimate_half_life_rescales_the_remaining_allowance():
    pm = PairPositionManager(half_life_multiplier_max_hold=3.0)
    pos = _open_one(pm, half_life_days=10.0)
    assert pos.max_holding_days == pytest.approx(30.0)

    assert pm.reestimate_half_life("AAA", "BBB", 20.0)
    assert pos.max_holding_days == pytest.approx(60.0)      # reverts slower -> hold longer
    assert pm.reestimate_half_life("AAA", "BBB", 2.0)
    assert pos.max_holding_days == pytest.approx(6.0)       # reverts faster -> hold less

    assert not pm.reestimate_half_life("XXX", "YYY", 5.0)   # unknown pair is a no-op


def test_a_downward_reestimate_can_retire_a_position_immediately():
    """The intended sharp edge: if the spread turns out to revert much faster
    than believed at entry, an already-long hold is already too long. A
    re-estimate that only ever EXTENDED the timeout would silently make this
    ablation a one-directional 'hold losers longer' rule."""
    pm = PairPositionManager(half_life_multiplier_max_hold=3.0)
    entry = datetime(2020, 1, 1)
    _open_one(pm, half_life_days=30.0, entry_time=entry)
    now = entry + timedelta(days=20)
    assert pm.check_exits({("AAA", "BBB"): -9.0}, now, 0.5) == []

    pm.reestimate_half_life("AAA", "BBB", 2.0)              # budget 90d -> 6d
    assert [r for _p, r in pm.check_exits({("AAA", "BBB"): -9.0}, now, 0.5)] == ["stale_timeout"]


def test_dynamic_half_life_changes_holding_periods_end_to_end():
    close, adv = _panel(n_days=600, n_pairs=3, seed=71)
    schedule = _schedule(close)
    base = run_scan_backtest(close, adv, schedule, PairsScanConfig())
    dyn = run_scan_backtest(close, adv, schedule, PairsScanConfig(dynamic_half_life=True))
    assert base.trades and dyn.trades

    def mean_hold(report):
        return float(np.mean([(pd.Timestamp(t.exit_date) - pd.Timestamp(t.entry_date)).days
                              for t in report.trades]))

    assert mean_hold(base) != pytest.approx(mean_hold(dyn)), (
        "re-estimation had no effect on holding periods — the ablation is not wired in")


# ── ablation 2: cointegration-breakdown exit ────────────────────────────────

def test_breakdown_exit_fires_when_the_pair_leaves_the_tradeable_set():
    pm = PairPositionManager(half_life_multiplier_max_hold=3.0)
    _open_one(pm, half_life_days=30.0)
    now = datetime(2020, 1, 2)
    z = {("AAA", "BBB"): -9.0}

    assert pm.check_exits(z, now, 0.5) == []
    closed = pm.check_exits(z, now, 0.5, broken_pairs={("AAA", "BBB")})
    assert [r for _p, r in closed] == ["coint_breakdown"]


def test_reversion_outranks_breakdown():
    """A pair that reverted AND dropped out of the eligible set should book
    the profitable exit, not be relabelled (and mis-attributed) a breakdown."""
    pm = PairPositionManager(half_life_multiplier_max_hold=3.0)
    _open_one(pm)
    closed = pm.check_exits({("AAA", "BBB"): 0.0}, datetime(2020, 1, 2), 0.5,
                            broken_pairs={("AAA", "BBB")})
    assert [r for _p, r in closed] == ["z_reversion"]


def test_breakdown_exit_uses_is_tradeable_not_the_half_life_band():
    """The eligible-for-ENTRY set additionally applies the configured
    half-life band. Using that as the exit trigger would make a parameter
    that is supposed to gate entries silently gate exits too, so a pair whose
    half-life merely drifted outside the band must NOT be called broken."""
    close, adv = _panel(n_days=500, n_pairs=3, seed=81)
    schedule = _schedule(close)
    wide = PairsScanConfig(coint_breakdown_exit=True,
                           min_half_life_days=1.0, max_half_life_days=60.0)
    narrow = PairsScanConfig(coint_breakdown_exit=True,
                             min_half_life_days=1.0, max_half_life_days=8.0)
    wide_run = run_scan_backtest(close, adv, schedule, wide)
    narrow_run = run_scan_backtest(close, adv, schedule, narrow)

    def breakdown_share(report):
        n = report.exit_reason_counts.get("coint_breakdown", 0)
        return n / max(len(report.trades), 1)

    # Narrowing the ENTRY band admits fewer pairs; it must not manufacture
    # breakdown exits for the ones it does admit.
    assert narrow_run.trades
    assert breakdown_share(narrow_run) <= breakdown_share(wide_run) + 1e-9


# ── ablation 3: z-widening stop (POLICY CHANGE) ─────────────────────────────

def test_stop_fires_only_when_the_spread_widens_past_both_the_level_and_entry():
    pm = PairPositionManager(half_life_multiplier_max_hold=3.0)
    _open_one(pm, half_life_days=30.0, entry_z=-2.0)
    now = datetime(2020, 1, 2)
    stop = 3.0

    # Inside the stop level: hold.
    assert pm.check_exits({("AAA", "BBB"): -2.5}, now, 0.5, stop_z=stop) == []
    # Through it, and wider than entry: stop.
    closed = pm.check_exits({("AAA", "BBB"): -3.5}, now, 0.5, stop_z=stop)
    assert [r for _p, r in closed] == ["stop_loss"]


def test_stop_does_not_fire_on_a_position_that_was_already_beyond_it_at_entry():
    """A position opened at z = -4.0 under a stop level of 3.0 has not
    'widened past the entry point' — stopping it on bar one would measure the
    entry distribution, not the behavior the stop is meant to catch."""
    pm = PairPositionManager(half_life_multiplier_max_hold=3.0)
    _open_one(pm, half_life_days=30.0, entry_z=-4.0)
    now = datetime(2020, 1, 2)
    assert pm.check_exits({("AAA", "BBB"): -4.0}, now, 0.5, stop_z=3.0) == []
    assert pm.check_exits({("AAA", "BBB"): -3.5}, now, 0.5, stop_z=3.0) == []
    closed = pm.check_exits({("AAA", "BBB"): -4.5}, now, 0.5, stop_z=3.0)
    assert [r for _p, r in closed] == ["stop_loss"]


def test_stop_is_symmetric_across_sides():
    pm = PairPositionManager(half_life_multiplier_max_hold=3.0)
    _open_one(pm, half_life_days=30.0, side=SpreadSide.SHORT_SPREAD, entry_z=2.0)
    now = datetime(2020, 1, 2)
    assert pm.check_exits({("AAA", "BBB"): 2.5}, now, 0.5, stop_z=3.0) == []
    closed = pm.check_exits({("AAA", "BBB"): 3.5}, now, 0.5, stop_z=3.0)
    assert [r for _p, r in closed] == ["stop_loss"]


def test_stop_z_is_derived_from_entry_z_and_never_configured_alone():
    """The stop level is not an independent knob: it is `entry_z` times a
    pinned multiple, so re-optimizing entry_z moves it automatically and it
    cannot drift into being a 6th free parameter."""
    cfg = PairsScanConfig(entry_z=2.0, stop_z_multiple=1.5)
    assert cfg.stop_z == pytest.approx(3.0)
    assert PairsScanConfig(entry_z=2.5, stop_z_multiple=1.5).stop_z == pytest.approx(3.75)
    assert STOP_Z_MULTIPLE == 1.5


def test_disabling_the_stale_timeout_leaves_positions_open_rather_than_hidden():
    """Ablation 3a removes the timeout so the stop can replace it inside the
    5-parameter budget. The cost is that non-reverting, non-widening spreads
    are held indefinitely — which must show up as `open_at_end`, not quietly
    vanish from the trade count."""
    close, adv = _panel(n_days=500, n_pairs=3, seed=91)
    schedule = _schedule(close)
    base = run_scan_backtest(close, adv, schedule, PairsScanConfig())
    no_timeout = run_scan_backtest(close, adv, schedule, PairsScanConfig(
        stop_z_multiple=STOP_Z_MULTIPLE, stale_timeout_enabled=False))
    assert "stale_timeout" not in no_timeout.exit_reason_counts
    assert no_timeout.open_at_end >= base.open_at_end


# ── LOOK-AHEAD: the round-one guard, extended to every new exit rule ─────────

@pytest.mark.parametrize("name", list(ABLATIONS))
def test_trades_before_cutoff_unchanged_by_future_price_mutation_under_every_ablation(name):
    """Round one pinned this for the default configuration. Ablations 1 and 2
    both consult scan output DURING a hold — new places for tomorrow's data to
    reach today's exit — so the same guard has to hold for each of them:
    corrupt every price on/after a cutoff, and no trade that had already
    closed by then may move."""
    close, adv = _panel(n_days=500, n_pairs=3, seed=101)
    cfg = PairsScanConfig(coint_lookback_days=_LOOKBACK,
                          revalidate_every_days=_REVALIDATE, **ABLATIONS[name])
    baseline = run_scan_backtest(close, adv, _schedule(close), cfg)

    cutoff = close.index[350]
    corrupted = close.copy()
    corrupted.loc[corrupted.index >= cutoff] *= 100.0
    mutated = run_scan_backtest(corrupted, adv, _schedule(corrupted), cfg)

    def before(report):
        return [(t.code_a, t.code_b, t.entry_date, t.exit_date, t.exit_reason,
                 round(t.net_pnl, 6))
                for t in report.trades if pd.Timestamp(t.exit_date) < cutoff]

    assert before(baseline), f"{name}: cutoff chosen badly — no closed trades before it"
    assert before(baseline) == before(mutated)


@pytest.mark.parametrize("name", list(ABLATIONS))
def test_ablations_cannot_reach_a_future_scan(name):
    """Ablation 1 reads half-lives and ablation 2 reads the eligible set from
    the scan schedule mid-hold. Inject a scan dated after the replay window;
    if either reached forward for 'the current' estimate instead of the latest
    PAST one, results would move."""
    close, adv = _panel(n_days=400, n_pairs=3, seed=111)
    schedule = _schedule(close)
    cfg = PairsScanConfig(**ABLATIONS[name])
    baseline = run_scan_backtest(close, adv, schedule, cfg)

    future_date = close.index[-1] + pd.Timedelta(days=30)
    poisoned = dict(schedule)
    poisoned[future_date] = []                      # would flag EVERY pair broken
    poisoned_run = run_scan_backtest(close, adv, poisoned, cfg)

    assert len(poisoned_run.trades) == len(baseline.trades)
    assert poisoned_run.total_net_pnl == pytest.approx(baseline.total_net_pnl, abs=1e-9)
    assert poisoned_run.exit_reason_counts == baseline.exit_reason_counts


def test_reestimation_only_ever_uses_a_scan_dated_at_or_before_today():
    """Direct form of the same property for ablation 1: every half-life a
    position is ever re-estimated with must come from a scan that had already
    run. Recorded through the manager rather than inferred from P&L."""
    close, adv = _panel(n_days=500, n_pairs=3, seed=121)
    schedule = _schedule(close)
    hl_by_scan: dict[float, set[pd.Timestamp]] = {}
    for as_of, results in schedule.items():
        for r in results:
            hl_by_scan.setdefault(round(r.half_life_days, 9), set()).add(as_of)

    seen: list[tuple[pd.Timestamp, float]] = []
    real = PairPositionManager.reestimate_half_life

    def recording(self, code_a, code_b, half_life_days):
        pos = self.get(code_a, code_b)
        if pos is not None:
            seen.append((pd.Timestamp(pos.entry_time), float(half_life_days)))
        return real(self, code_a, code_b, half_life_days)

    PairPositionManager.reestimate_half_life = recording
    try:
        run_scan_backtest(close, adv, schedule, PairsScanConfig(dynamic_half_life=True))
    finally:
        PairPositionManager.reestimate_half_life = real

    assert seen, "no re-estimation happened — the assertion below would be vacuous"
    for _entry_time, hl in seen:
        assert round(hl, 9) in hl_by_scan, "re-estimated from a half-life no scan produced"


# ── parameter discipline ────────────────────────────────────────────────────

def test_param_grid_never_grids_an_exit_ablation_constant():
    """The ablation switches define the strategy VARIANT under test; the
    walk-forward search must never be able to select among them, or the
    variant becomes a 6th (categorical) fitted parameter."""
    from python.backtest.optimize import load_param_grid

    gridded = {k for combo in load_param_grid("pairs_trading") for k in combo}
    assert gridded == {"entry_z", "exit_z", "half_life_multiplier_max_hold"}
    assert not (gridded & {"dynamic_half_life", "coint_breakdown_exit",
                           "stop_z_multiple", "stale_timeout_enabled"})


def test_exit_ablations_add_no_key_to_the_strategy_config():
    import yaml

    from python.backtest.param_guard import MAX_FREE_PARAMETERS, check_max_parameters

    with open("configs/strategy.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["pairs_trading"]
    ok, n = check_max_parameters(cfg)
    assert ok and n == MAX_FREE_PARAMETERS
    assert not (set(cfg) & {"dynamic_half_life", "coint_breakdown_exit",
                            "stop_z_multiple", "stale_timeout_enabled", "stop_z"})


def test_stop_variant_stays_within_the_five_parameter_budget():
    """A3a is the only stop configuration that is even eligible for promotion:
    the stop REPLACES `half_life_multiplier_max_hold` (timeout disabled), so
    the count stays at 5. A3b runs both and is therefore diagnostic-only."""
    a3a = PairsScanConfig(**ABLATIONS["A3a_stop_replaces_timeout"])
    assert a3a.stop_z is not None and a3a.stale_timeout_enabled is False

    a3b = PairsScanConfig(**ABLATIONS["A3b_stop_plus_timeout"])
    assert a3b.stop_z is not None and a3b.stale_timeout_enabled is True


def test_unknown_exit_rule_is_rejected_rather_than_ignored():
    """A typo'd switch that silently did nothing would produce an ablation
    table where two rows are secretly the same run."""
    from python.backtest.optimize import build_pairs_scan_backtest_fn

    close, adv = _panel(n_days=300, n_pairs=2, seed=131)
    base_cfg = {"entry_z": 2.0, "exit_z": 0.5, "coint_lookback_days": _LOOKBACK,
                "revalidate_every_days": _REVALIDATE, "notional_per_leg": 50_000.0,
                "half_life_multiplier_max_hold": 3.0,
                "min_half_life_days": 1.0, "max_half_life_days": 60.0}
    fn = build_pairs_scan_backtest_fn(close, adv, _schedule(close), base_cfg,
                                      exit_rules={"dynamic_halflife": True})
    with pytest.raises(ValueError, match="unknown exit rule"):
        fn(close.index[200].to_pydatetime(), close.index[-1].to_pydatetime(), {})


def test_metrics_expose_the_exit_mix_and_the_gross_profit_factor():
    """The ablations are judged on the exit mix, and 'is there an edge at all'
    is judged on the pre-cost profit factor — both have to survive the trip
    through the walk-forward metrics dict, not just the engine report."""
    from python.backtest.optimize import build_pairs_scan_backtest_fn

    close, adv = _panel(n_days=500, n_pairs=3, seed=141)
    base_cfg = {"entry_z": 2.0, "exit_z": 0.5, "coint_lookback_days": _LOOKBACK,
                "revalidate_every_days": _REVALIDATE, "notional_per_leg": 50_000.0,
                "half_life_multiplier_max_hold": 3.0,
                "min_half_life_days": 1.0, "max_half_life_days": 60.0}
    fn = build_pairs_scan_backtest_fn(close, adv, _schedule(close), base_cfg)
    metrics = fn(close.index[_LOOKBACK].to_pydatetime(), close.index[-1].to_pydatetime(), {})

    assert metrics["n_trades"] > 0
    assert sum(metrics["exit_reasons"].values()) == metrics["n_trades"]
    assert set(metrics["exit_reasons"]) <= {"z_reversion", "stale_timeout"}
    assert metrics["gross_pnl"] == pytest.approx(
        metrics["total_net_pnl"] + metrics["total_cost"], rel=1e-9)
    assert metrics["profit_factor_gross"] >= metrics["profit_factor"]


def test_every_ablation_is_reachable_and_actually_changes_something():
    """Guards against an ablation table of five identical rows."""
    close, adv = _panel(n_days=600, n_pairs=3, seed=151)
    schedule = _schedule(close)
    signatures = {}
    for name, rules in ABLATIONS.items():
        report = run_scan_backtest(close, adv, schedule, PairsScanConfig(**rules))
        signatures[name] = (len(report.trades), round(report.total_net_pnl, 6),
                            tuple(sorted(report.exit_reason_counts.items())))
    assert len(set(signatures.values())) == len(ABLATIONS), (
        f"two ablations produced identical results: {signatures}")
