"""
Guards on the UHAI parameter suggester's contract with this repo.

The suggester is additive — the fixed grid always runs — so the risks worth
testing are not "does it find good parameters" but "can it corrupt a run":
fitting on synthetic Sharpes, proposing a fractional day count, escaping the
grid's sanctioned range, or POSTing to whatever else is on a local port.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from python.uhai.client import (
    DEFAULT_SIDECAR_URL,
    assess_degradation,
    bounds_from_grid_spec,
    coerce_to_grid_types,
)
from python.uhai.history import load_history, summarize, to_sidecar_payload
from python.uhai.regime import UNKNOWN, current_regime, dominant_regime

PARAM_GRIDS_PATH = Path("configs/param_grids.yaml")


def _write_history(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _record(
    *,
    strategy: str = "pairs_trading",
    params: dict | None = None,
    sharpe: float = 1.0,
    ts: str = "2026-01-01T00:00:00+00:00",
    data_source: str = "polygon",
    decision: str = "REJECTED",
    baseline_params: dict | None = None,
    baseline_sharpe: float | None = 0.0,
) -> dict:
    entry = {
        "timestamp": ts,
        "strategy": strategy,
        "decision": decision,
        "candidate_params": params if params is not None else {"entry_z": 2.0},
        "candidate_oos_sharpe": sharpe,
        "data_source": data_source,
    }
    # Every real promotion.py record always has these two — they are what is
    # ACTUALLY deployed, re-measured on this run's window, independent of
    # whatever candidate the search happened to try. Settable to None to
    # simulate a malformed/legacy record missing them.
    if baseline_sharpe is not None:
        entry["baseline_oos_sharpe"] = baseline_sharpe
        entry["baseline_params"] = baseline_params if baseline_params is not None else {"entry_z": 4.0}
    return entry


# ── history: what may and may not become training data ───────────────────────

def test_synthetic_runs_are_excluded_from_real_history(tmp_path):
    """A --demo run records data_source=synthetic. Fitting the suggester that
    proposes real-money parameters on synthetic Sharpes would be worse than
    having no memory at all, so it must be excluded by default."""
    path = tmp_path / "history.jsonl"
    _write_history(path, [
        _record(sharpe=9.9, data_source="synthetic"),
        _record(sharpe=0.4, data_source="polygon"),
    ])

    real = load_history("pairs_trading", path=path)
    assert [r.oos_sharpe for r in real] == [0.4]

    both = load_history("pairs_trading", path=path, include_synthetic=True)
    assert len(both) == 2


def test_history_is_scoped_to_one_strategy(tmp_path):
    path = tmp_path / "history.jsonl"
    _write_history(path, [
        _record(strategy="pairs_trading", params={"entry_z": 2.0}),
        _record(strategy="xsection_mean_reversion", params={"lookback_days": 3}),
    ])

    assert len(load_history("pairs_trading", path=path)) == 1
    assert len(load_history("xsection_mean_reversion", path=path)) == 1


def test_records_whose_parameter_shape_changed_are_dropped(tmp_path):
    """A record tested before an axis existed cannot be placed in today's
    search space; keeping it would misattribute its Sharpe to the remaining
    dimensions."""
    path = tmp_path / "history.jsonl"
    _write_history(path, [
        _record(params={"entry_z": 2.0, "exit_z": 0.5}),
        _record(params={"entry_z": 2.0}),
    ])

    kept = load_history("pairs_trading", path=path, param_keys={"entry_z", "exit_z"})
    assert [set(r.params) for r in kept] == [{"entry_z", "exit_z"}]


def test_malformed_lines_do_not_break_the_read(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text(
        "not json at all\n"
        "\n"
        + json.dumps({"strategy": "pairs_trading"}) + "\n"           # no params/sharpe
        + json.dumps(_record(sharpe=1.25)) + "\n",
        encoding="utf-8",
    )

    assert [r.oos_sharpe for r in load_history("pairs_trading", path=path)] == [1.25]


@pytest.mark.parametrize("bad_sharpe", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_sharpes_are_dropped(tmp_path, bad_sharpe):
    """A run with no trades or zero return variance can record a non-finite
    Sharpe. It says nothing about the parameter space and would poison a GP
    fit, so it must not become an observation."""
    path = tmp_path / "history.jsonl"
    # json.dumps emits Infinity/NaN, which json.loads accepts back as floats.
    _write_history(path, [_record(sharpe=bad_sharpe), _record(sharpe=1.25)])

    assert [r.oos_sharpe for r in load_history("pairs_trading", path=path)] == [1.25]


def test_missing_history_file_is_a_cold_start_not_an_error(tmp_path):
    assert load_history("pairs_trading", path=tmp_path / "absent.jsonl") == []


def test_history_is_returned_oldest_first(tmp_path):
    path = tmp_path / "history.jsonl"
    _write_history(path, [
        _record(ts="2026-03-01T00:00:00+00:00", sharpe=3.0),
        _record(ts="2026-01-01T00:00:00+00:00", sharpe=1.0),
        _record(ts="2026-02-01T00:00:00+00:00", sharpe=2.0),
    ])

    assert [r.oos_sharpe for r in load_history("pairs_trading", path=path)] == [1.0, 2.0, 3.0]


def test_rejected_candidates_are_kept_as_observations(tmp_path):
    """A REJECTED candidate still tells the model that this corner of the
    space scores about this much. Filtering to PROMOTED-only would leave it
    blind to every region known to be bad."""
    path = tmp_path / "history.jsonl"
    _write_history(path, [
        _record(decision="REJECTED", sharpe=-0.8),
        _record(decision="PROMOTED", sharpe=1.4),
    ])

    records = load_history("pairs_trading", path=path)
    assert len(records) == 2
    assert summarize(records)["promoted"] == 1


def test_baseline_fields_are_read_back_distinct_from_candidate_fields(tmp_path):
    path = tmp_path / "history.jsonl"
    _write_history(path, [_record(
        params={"entry_z": 1.5}, sharpe=2.5,
        baseline_params={"entry_z": 4.0}, baseline_sharpe=0.3,
    )])

    record = load_history("pairs_trading", path=path)[0]
    assert record.params == {"entry_z": 1.5}
    assert record.oos_sharpe == 2.5
    assert record.baseline_params == {"entry_z": 4.0}
    assert record.baseline_oos_sharpe == 0.3


def test_missing_baseline_sharpe_is_none_not_zero(tmp_path):
    path = tmp_path / "history.jsonl"
    _write_history(path, [_record(baseline_sharpe=None)])

    record = load_history("pairs_trading", path=path)[0]
    assert record.baseline_oos_sharpe is None


def test_market_regime_tag_is_read_back_when_present(tmp_path):
    """self_improve_loop.py records the window's regime under extra. Recording
    it is only worth doing if it can be read back."""
    path = tmp_path / "history.jsonl"
    tagged = _record(sharpe=1.0)
    tagged["extra"] = {"market_regime": "Bull"}
    _write_history(path, [tagged, _record(sharpe=0.5)])

    records = load_history("pairs_trading", path=path)
    assert records[0].market_regime == "Bull"
    # A record written before the tag existed must still load.
    assert records[1].market_regime == "unknown"


def test_sidecar_payload_carries_only_what_the_optimizer_needs(tmp_path):
    path = tmp_path / "history.jsonl"
    _write_history(path, [_record(params={"entry_z": 2.0}, sharpe=0.7)])

    payload = to_sidecar_payload(load_history("pairs_trading", path=path))
    assert list(payload[0]) == ["params", "oos_sharpe", "timestamp"]
    assert payload[0]["params"] == {"entry_z": 2.0}


# ── bounds: what the suggester is allowed to search over ─────────────────────

def test_boolean_axes_are_excluded_from_the_search_space():
    """A Real-valued GP over a boolean axis would emit values like 0.37, which
    is not a setting the strategy can take. Those axes stay grid-only."""
    bounds = bounds_from_grid_spec({"or_minutes": [5, 15, 30], "vwap_side_filter": [True, False]})
    assert set(bounds) == {"or_minutes"}


def test_single_valued_axes_are_excluded():
    """low == high is a zero-width dimension, which skopt rejects outright."""
    assert bounds_from_grid_spec({"fixed": [2.0], "varying": [1.0, 3.0]}) == {"varying": [1.0, 3.0]}


def test_bounds_are_the_grids_own_min_and_max():
    """Suggestions are confined to the range the grid's author already
    sanctioned — the suggester widens resolution, never scope."""
    assert bounds_from_grid_spec({"entry_z": [1.5, 2.0, 2.5]}) == {"entry_z": [1.5, 2.5]}


def test_every_self_improve_strategy_has_a_searchable_axis():
    """Both strategies the loop drives must expose at least one continuous
    axis, or enabling UHAI silently does nothing for them."""
    grids = yaml.safe_load(PARAM_GRIDS_PATH.read_text(encoding="utf-8"))
    for strategy in ("pairs_trading", "xsection_mean_reversion"):
        assert bounds_from_grid_spec(grids[strategy]), f"{strategy} has no continuous axis"


# ── coercion: suggestions must land on a value the strategy can take ─────────

def test_integer_axes_are_rounded_to_integers():
    """xsection_mean_reversion's lookback_days is a count of days; 2.7 days is
    not a parameter the strategy can take."""
    grid = {"lookback_days": [1, 3, 5], "gross_leverage_target": [0.5, 1.0]}
    coerced = coerce_to_grid_types({"lookback_days": 2.7, "gross_leverage_target": 0.618034}, grid)

    assert coerced["lookback_days"] == 3
    assert isinstance(coerced["lookback_days"], int)
    assert coerced["gross_leverage_target"] == 0.618


def test_continuous_axes_keep_only_supportable_precision():
    grid = {"entry_z": [1.5, 2.0, 2.5]}
    assert coerce_to_grid_types({"entry_z": 2.17391304347826}, grid)["entry_z"] == 2.174


def test_a_float_valued_grid_does_not_make_an_integer_axis():
    """2.0 in the grid means the axis is continuous even though it is a round
    number — only an all-int grid signals an integer axis."""
    grid = {"half_life_multiplier_max_hold": [2.0, 3.0, 4.0]}
    coerced = coerce_to_grid_types({"half_life_multiplier_max_hold": 2.5}, grid)
    assert coerced["half_life_multiplier_max_hold"] == 2.5


def test_coerced_integer_suggestion_stays_inside_grid_bounds():
    """Rounding must not push a value past the range bounds_from_grid_spec
    handed the optimizer."""
    grid = {"lookback_days": [1, 3, 5]}
    low, high = bounds_from_grid_spec(grid)["lookback_days"]
    for raw in (1.0, 1.4, 3.3, 4.9, 5.0):
        value = coerce_to_grid_types({"lookback_days": raw}, grid)["lookback_days"]
        assert low <= value <= high


# ── degradation: the staleness signal ────────────────────────────────────────

def test_degradation_needs_both_a_recent_and_an_earlier_sample(tmp_path, monkeypatch):
    """Without the monkeypatch below this reads the REAL
    backtests/logs/promotion_history.jsonl — harmless while that file has
    too little real history to reach the threshold, but a latent isolation
    bug that a real run elsewhere (e.g. scripts/self_improve_loop.py against
    real market data) can silently turn into a failure here. Pin the
    isolation explicitly rather than relying on the repo's real history
    happening to stay thin."""
    path = tmp_path / "history.jsonl"
    _write_history(path, [_record(baseline_sharpe=1.0) for _ in range(4)])
    monkeypatch.setattr("python.uhai.client.load_history",
                        lambda *a, **k: load_history("pairs_trading", path=path))

    verdict = assess_degradation("pairs_trading", recent_runs=3, min_earlier_runs=3)
    assert not verdict.is_degraded
    assert "insufficient history" in verdict.reason


def test_degradation_is_detected_when_recent_runs_score_worse(tmp_path, monkeypatch):
    """Recent runs' LIVE (baseline) parameters are re-measured worse than
    earlier — the strategy actually deployed looks like it's decaying."""
    path = tmp_path / "history.jsonl"
    _write_history(path, [
        _record(ts=f"2026-01-0{i}T00:00:00+00:00", baseline_sharpe=2.0) for i in range(1, 5)
    ] + [
        _record(ts=f"2026-02-0{i}T00:00:00+00:00", baseline_sharpe=0.5) for i in range(1, 4)
    ])
    monkeypatch.setattr("python.uhai.client.load_history",
                        lambda *a, **k: load_history("pairs_trading", path=path))

    verdict = assess_degradation("pairs_trading", recent_runs=3, min_earlier_runs=3)
    assert verdict.is_degraded
    assert verdict.recent_mean_sharpe < verdict.earlier_mean_sharpe


def test_stable_history_is_not_flagged_as_degraded(tmp_path, monkeypatch):
    path = tmp_path / "history.jsonl"
    _write_history(path, [
        _record(ts=f"2026-01-0{i}T00:00:00+00:00", baseline_sharpe=1.0) for i in range(1, 8)
    ])
    monkeypatch.setattr("python.uhai.client.load_history",
                        lambda *a, **k: load_history("pairs_trading", path=path))

    assert not assess_degradation("pairs_trading", recent_runs=3, min_earlier_runs=3).is_degraded


def test_degradation_tracks_the_live_baseline_not_the_search_candidates(tmp_path, monkeypatch):
    """Regression for the actual bug: a search that keeps finding
    (rejected) candidates with a RISING Sharpe says nothing about whether
    the LIVE strategy is decaying — only the re-measured baseline does.
    Candidate Sharpe rising + baseline Sharpe falling must still flag
    degraded."""
    path = tmp_path / "history.jsonl"
    records = [
        _record(ts=f"2026-01-0{i}T00:00:00+00:00", sharpe=0.5 + 0.1 * i, baseline_sharpe=2.0)
        for i in range(1, 5)
    ] + [
        _record(ts=f"2026-02-0{i}T00:00:00+00:00", sharpe=1.5 + 0.1 * i, baseline_sharpe=0.4)
        for i in range(1, 4)
    ]
    _write_history(path, records)
    monkeypatch.setattr("python.uhai.client.load_history",
                        lambda *a, **k: load_history("pairs_trading", path=path))

    verdict = assess_degradation("pairs_trading", recent_runs=3, min_earlier_runs=3)

    # The candidate-Sharpe trend (rising) must NOT be what drove this.
    recent_candidate = [1.6, 1.7, 1.8]
    earlier_candidate = [0.6, 0.7, 0.8, 0.9]
    assert sum(recent_candidate) / 3 > sum(earlier_candidate) / 4, "test fixture sanity"
    assert verdict.is_degraded
    assert verdict.recent_mean_sharpe == pytest.approx(0.4)
    assert verdict.earlier_mean_sharpe == pytest.approx(2.0)


def test_records_with_no_baseline_sharpe_are_excluded_from_degradation(tmp_path, monkeypatch):
    """A malformed/legacy record with no baseline_oos_sharpe must be dropped
    from the trend, not treated as a zero (which would look like a crash to
    zero performance that never happened)."""
    path = tmp_path / "history.jsonl"
    records = (
        [_record(ts=f"2026-01-0{i}T00:00:00+00:00", baseline_sharpe=1.0) for i in range(1, 4)]
        + [_record(ts="2026-01-05T00:00:00+00:00", baseline_sharpe=None)]  # malformed, excluded
        + [_record(ts=f"2026-02-0{i}T00:00:00+00:00", baseline_sharpe=1.0) for i in range(1, 4)]
    )
    _write_history(path, records)
    monkeypatch.setattr("python.uhai.client.load_history",
                        lambda *a, **k: load_history("pairs_trading", path=path))

    verdict = assess_degradation("pairs_trading", recent_runs=3, min_earlier_runs=3)
    assert not verdict.is_degraded
    assert verdict.n_recent == 3
    assert verdict.n_earlier == 3


# ── regime tagging ───────────────────────────────────────────────────────────

def test_regime_labels_a_rising_series_bull_and_a_falling_one_bear():
    dates = pd.bdate_range("2024-01-01", periods=120)
    rising = pd.Series([100 * (1.004 ** i) for i in range(120)], index=dates)
    falling = pd.Series([100 * (0.996 ** i) for i in range(120)], index=dates)

    assert current_regime(rising) == "Bull"
    assert current_regime(falling) == "Bear"
    assert dominant_regime(rising) == "Bull"


def test_regime_is_unknown_without_enough_history_rather_than_raising():
    dates = pd.bdate_range("2024-01-01", periods=5)
    assert current_regime(pd.Series([100.0] * 5, index=dates), window=20) == UNKNOWN


# ── service identity ─────────────────────────────────────────────────────────

def test_sidecar_default_port_avoids_this_repos_dashboard():
    """8082 is this repo's dashboard (scripts/start_dashboard.py, README.md),
    whose API includes /api/engine/start and /api/positions/flatten_all. The
    suggester's default must not point at it."""
    assert ":8082" not in DEFAULT_SIDECAR_URL


def test_unidentified_local_service_is_not_treated_as_the_sidecar(monkeypatch):
    """A 200 from something on a local port is not proof of identity."""
    import httpx

    from python.uhai import client as uhai_client

    class _DashboardLikeResponse:
        def raise_for_status(self): return None
        def json(self): return {"engine": "running", "positions": []}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _DashboardLikeResponse())
    assert not uhai_client.is_uhai_sidecar("http://127.0.0.1:8082")


def test_sidecar_health_shape_is_accepted(monkeypatch):
    import httpx

    from python.uhai import client as uhai_client

    class _SidecarResponse:
        def raise_for_status(self): return None
        def json(self): return {"status": "ok", "features": {"embedding": True}}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _SidecarResponse())
    assert uhai_client.is_uhai_sidecar("http://127.0.0.1:8092")


def test_offline_sidecar_reports_absent_rather_than_raising(monkeypatch):
    import httpx

    from python.uhai import client as uhai_client

    def _refuse(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _refuse)
    assert not uhai_client.is_uhai_sidecar("http://127.0.0.1:8092")
