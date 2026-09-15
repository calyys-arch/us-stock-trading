"""
scripts/check_uhai_recommendations.py — the strategy list it monitors must
be derived from configs/strategy.yaml + configs/param_grids.yaml at call
time, not hardcoded. This is a regression test for a real staleness bug:
the script used to hardcode STRATEGIES = ["pairs_trading",
"xsection_mean_reversion"] — both RETIRED 2026-09-11 — so it kept reporting
on two strategies self_improve_loop.py can no longer promote, while staying
silent about every currently-enabled one (sweep_reclaim, orb_vwap, ...).

Runs inside a temp cwd (monkeypatch.chdir) + importlib.reload, same pattern
as test_sync_uhai.py, since STRATEGY_CONFIG_PATH/PARAM_GRIDS_PATH are
Path(...) literals evaluated at import time.
"""
from __future__ import annotations

import importlib
import json

import pytest
import yaml


@pytest.fixture()
def check_mod(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import scripts.check_uhai_recommendations as mod

    importlib.reload(mod)
    (tmp_path / "configs").mkdir()
    (tmp_path / "backtests" / "logs").mkdir(parents=True)
    return mod


def _write_configs(tmp_path, strategy_blocks: dict, grid_strategies: list[str]) -> None:
    (tmp_path / "configs" / "strategy.yaml").write_text(
        yaml.safe_dump(strategy_blocks), encoding="utf-8")
    (tmp_path / "configs" / "param_grids.yaml").write_text(
        yaml.safe_dump({name: {"x": [1, 2, 3]} for name in grid_strategies}), encoding="utf-8")


def test_retired_strategies_are_excluded_even_if_still_gridded(tmp_path, check_mod):
    """A strategy can stay in param_grids.yaml (harmless) after being set to
    enabled: false — it must not show up as something worth monitoring."""
    _write_configs(
        tmp_path,
        strategy_blocks={
            "pairs_trading": {"enabled": False, "auto_execute": False},
            "xsection_mean_reversion": {"enabled": False, "auto_execute": False},
            "sweep_reclaim": {"enabled": True, "auto_execute": False},
        },
        grid_strategies=["pairs_trading", "xsection_mean_reversion", "sweep_reclaim"],
    )
    assert check_mod._default_strategies() == ["sweep_reclaim"]


def test_enabled_strategy_without_a_grid_is_excluded(tmp_path, check_mod):
    """Nothing to re-search without a grid, so it is not a self-improve-loop
    candidate even if enabled."""
    _write_configs(
        tmp_path,
        strategy_blocks={"sweep_reclaim": {"enabled": True}, "no_grid_strategy": {"enabled": True}},
        grid_strategies=["sweep_reclaim"],
    )
    assert check_mod._default_strategies() == ["sweep_reclaim"]


def test_missing_config_files_yield_empty_list_not_a_crash(tmp_path, check_mod):
    assert check_mod._default_strategies() == []


def test_main_reports_nothing_to_monitor_gracefully(tmp_path, check_mod, capsys, monkeypatch):
    # No configs/*.yaml written at all -> _default_strategies() == [].
    monkeypatch.setattr("sys.argv", ["check_uhai_recommendations.py"])
    exit_code = check_mod.main()
    assert exit_code == 0
    assert "No enabled strategy" in capsys.readouterr().out


def _write_history(path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _degraded_record(ts: str, baseline_sharpe: float, decision: str = "REJECTED") -> dict:
    return {
        "timestamp": ts,
        "strategy": "sweep_reclaim",
        "decision": decision,
        "reason": "gates failed: monte_carlo_p5_sharpe",
        "candidate_params": {"x": 2},
        "candidate_oos_sharpe": baseline_sharpe - 0.1,
        "baseline_oos_sharpe": baseline_sharpe,
        "gates": {"wfo_go": True, "monte_carlo_p5_sharpe": False},
        "data_source": "polygon",
    }


def test_main_explain_flag_prints_template_fallback_for_a_degraded_strategy(
    tmp_path, check_mod, capsys, monkeypatch,
):
    """A strategy whose baseline Sharpe has genuinely declined must be
    flagged AND, with --explain, must print a non-empty explanation of the
    most recent record — falling back to the local template since no
    sidecar is running in the test environment."""
    monkeypatch.setenv("UHAI_EXPLAIN_DISABLE", "1")
    _write_configs(
        tmp_path,
        strategy_blocks={"sweep_reclaim": {"enabled": True}},
        grid_strategies=["sweep_reclaim"],
    )
    _write_history(
        tmp_path / "backtests" / "logs" / "promotion_history.jsonl",
        [
            _degraded_record("2026-01-01T00:00:00+00:00", 1.0),
            _degraded_record("2026-02-01T00:00:00+00:00", 1.0),
            _degraded_record("2026-03-01T00:00:00+00:00", 1.0),
            _degraded_record("2026-04-01T00:00:00+00:00", 0.3),
            _degraded_record("2026-05-01T00:00:00+00:00", 0.3),
            _degraded_record("2026-06-01T00:00:00+00:00", 0.3),
        ],
    )
    monkeypatch.setattr(
        "sys.argv", ["check_uhai_recommendations.py", "--explain"])

    exit_code = check_mod.main()
    out = capsys.readouterr().out

    assert exit_code == 2
    assert "sweep_reclaim" in out
    assert "Worth re-running" in out
    assert "monte_carlo_p5_sharpe" in out  # from the fallback template's "failed gate(s)"
