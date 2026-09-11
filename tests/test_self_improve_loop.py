"""
scripts/self_improve_loop.py — its testable helper functions.

This script is THE integration point for Reality Check, UHAI suggestions,
regime tagging, and Bonferroni correction across this whole effort, but had
never had a single unit test protecting it — only ad-hoc manual `--demo`
CLI runs. These cover the three helpers that carry real decision logic:
`_load_grid_spec`, `_append_uhai_candidates`, `_window_regime`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.self_improve_loop as loop


def _args(**overrides) -> SimpleNamespace:
    defaults = dict(disable_uhai=False, demo=False, uhai_suggestions=5)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── _load_grid_spec ──────────────────────────────────────────────────────────

def test_load_grid_spec_returns_the_named_strategys_block():
    spec = loop._load_grid_spec("pairs_trading")
    assert set(spec) == {"entry_z", "exit_z", "half_life_multiplier_max_hold"}


def test_load_grid_spec_is_empty_for_an_unknown_strategy():
    assert loop._load_grid_spec("not_a_real_strategy") == {}


# ── _append_uhai_candidates ──────────────────────────────────────────────────

BASE_CFG = {"entry_z": 4.0, "exit_z": 0.5, "half_life_multiplier_max_hold": 3.0,
            "coint_lookback_days": 252, "min_half_life_days": 1.0}


def test_disable_uhai_flag_adds_nothing_without_calling_the_suggester(monkeypatch):
    called = []
    monkeypatch.setattr(loop.uhai_client, "suggest_params",
                        lambda *a, **k: called.append(1) or [])

    grid = [{"entry_z": 1.5}]
    added = loop._append_uhai_candidates("pairs_trading", _args(disable_uhai=True), BASE_CFG, grid)

    assert added == 0
    assert grid == [{"entry_z": 1.5}]
    assert called == [], "the suggester must not even be invoked when disabled"


def test_suggester_exception_adds_nothing_and_does_not_propagate(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("sidecar exploded")

    monkeypatch.setattr(loop.uhai_client, "suggest_params", _boom)

    grid = [{"entry_z": 1.5}]
    added = loop._append_uhai_candidates("pairs_trading", _args(), BASE_CFG, grid)

    assert added == 0
    assert grid == [{"entry_z": 1.5}]


def test_valid_suggestion_is_appended_to_the_grid(monkeypatch):
    monkeypatch.setattr(loop.uhai_client, "suggest_params",
                        lambda *a, **k: [{"entry_z": 2.28, "exit_z": 0.4}])

    grid = [{"entry_z": 1.5, "exit_z": 0.0}]
    added = loop._append_uhai_candidates("pairs_trading", _args(), BASE_CFG, grid)

    assert added == 1
    assert {"entry_z": 2.28, "exit_z": 0.4} in grid


def test_suggestion_with_unknown_key_is_dropped(monkeypatch):
    """A key absent from strategy.yaml cannot become a candidate — there is
    nothing in the live config for it to override."""
    monkeypatch.setattr(loop.uhai_client, "suggest_params",
                        lambda *a, **k: [{"entry_z": 2.0, "not_a_real_param": 99}])

    grid: list[dict] = []
    added = loop._append_uhai_candidates("pairs_trading", _args(), BASE_CFG, grid)

    assert added == 0
    assert grid == []


def test_ceiling_check_still_refuses_when_the_incumbent_itself_is_already_over(monkeypatch):
    """The unknown-key filter guarantees a suggestion's keys are a subset of
    base_cfg's own keys, so a merged config's free-parameter COUNT can never
    exceed base_cfg's own count — under normal operation this makes the
    ceiling check inside _append_uhai_candidates unreachable dead code,
    since preflight_check already refuses to run at all if the incumbent
    itself is over the ceiling.

    It stays in as defense-in-depth rather than being removed, so this test
    exercises that belt-and-suspenders path directly (a contrived
    already-over-ceiling base_cfg — 6 free params — standing in for
    "whatever future change might let one slip past preflight_check"),
    rather than pretending a normal suggestion can trigger it."""
    over_ceiling_base_cfg = {
        "entry_z": 4.0, "exit_z": 0.5, "half_life_multiplier_max_hold": 3.0,
        "min_half_life_days": 1.0, "max_half_life_days": 60.0, "revalidate_every_days_free": 21,
    }
    assert len(over_ceiling_base_cfg) > loop.MAX_FREE_PARAMETERS
    monkeypatch.setattr(loop.uhai_client, "suggest_params",
                        lambda *a, **k: [{"entry_z": 2.0}])

    grid: list[dict] = []
    added = loop._append_uhai_candidates("pairs_trading", _args(), over_ceiling_base_cfg, grid)

    assert added == 0
    assert grid == []


def test_duplicate_suggestion_already_in_the_grid_is_not_added_twice(monkeypatch):
    monkeypatch.setattr(loop.uhai_client, "suggest_params",
                        lambda *a, **k: [{"entry_z": 1.5, "exit_z": 0.0}])

    grid = [{"entry_z": 1.5, "exit_z": 0.0}]  # already present
    added = loop._append_uhai_candidates("pairs_trading", _args(), BASE_CFG, grid)

    assert added == 0
    assert grid == [{"entry_z": 1.5, "exit_z": 0.0}]


def test_demo_flag_is_forwarded_as_include_synthetic_history(monkeypatch):
    """A --demo run has only synthetic history; suggest_params must be told
    so explicitly rather than defaulting to excluding it (the default the
    suggester itself uses for a real run)."""
    captured = {}

    def _capture(strategy, current_params, grid_spec, *, n_suggestions, include_synthetic_history):
        captured["include_synthetic_history"] = include_synthetic_history
        return []

    monkeypatch.setattr(loop.uhai_client, "suggest_params", _capture)

    loop._append_uhai_candidates("pairs_trading", _args(demo=True), BASE_CFG, [])
    assert captured["include_synthetic_history"] is True

    loop._append_uhai_candidates("pairs_trading", _args(demo=False), BASE_CFG, [])
    assert captured["include_synthetic_history"] is False


# ── _window_regime ───────────────────────────────────────────────────────────

def test_window_regime_is_unknown_for_none_panel():
    assert loop._window_regime(None) == "unknown"


def test_window_regime_is_unknown_for_empty_panel():
    assert loop._window_regime(pd.DataFrame()) == "unknown"


def test_window_regime_labels_a_rising_multi_column_panel_bull():
    dates = pd.bdate_range("2020-01-01", periods=200)
    panel = pd.DataFrame({
        "a": [100 * 1.004 ** i for i in range(200)],
        "b": [50 * 1.005 ** i for i in range(200)],
    }, index=dates)

    assert loop._window_regime(panel) == "Bull"


def test_window_regime_labels_a_falling_multi_column_panel_bear():
    dates = pd.bdate_range("2020-01-01", periods=200)
    panel = pd.DataFrame({
        "a": [100 * 0.996 ** i for i in range(200)],
        "b": [50 * 0.995 ** i for i in range(200)],
    }, index=dates)

    assert loop._window_regime(panel) == "Bear"


def test_window_regime_averages_columns_rather_than_reading_one_symbol():
    """One wildly-behaving column must not dominate the tag — the panel is
    equal-weight averaged first, describing the WINDOW, not a single
    instrument."""
    dates = pd.bdate_range("2020-01-01", periods=200)
    panel = pd.DataFrame({
        "flat": [100.0] * 200,
        "also_flat": [50.0] * 200,
    }, index=dates)

    assert loop._window_regime(panel) == "Sideways"


def test_window_regime_swallows_exceptions_and_returns_unknown(monkeypatch):
    """A labeling failure must never cost a run its result."""
    def _boom(series):
        raise RuntimeError("regime blew up")

    monkeypatch.setattr(loop.uhai_regime, "dominant_regime", _boom)

    dates = pd.bdate_range("2020-01-01", periods=200)
    panel = pd.DataFrame({"a": [100.0] * 200}, index=dates)
    assert loop._window_regime(panel) == "unknown"
