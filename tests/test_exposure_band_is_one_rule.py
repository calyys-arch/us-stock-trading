"""The rebalance band had drifted into three different rules.

The backtest anchored the band on the last applied target, the order sheet on
whatever the operator last remembered to `--commit`, and the paper ledger on
`invested / equity` -- the drifted actual exposure. All three called it "the
10% rebalance band". Over the same panel the backtest and the drifted-anchor
path agreed to 0.015 on average but differed by up to 0.294, with 4.9% of days
apart by more than 0.05.

Nothing caught it, and nothing could have: the sheet's path depended on how
often a human happened to run it, so there was no sequence to compare. These
tests exist to make the three paths comparable and then compare them.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import scripts.paper_track_v1 as pt
from python.portfolio.strategy_v1 import (
    band_exposure, bucket_weighted_gross, target_weights,
)
from scripts.run_vol_target_study import (
    MAX_EXPOSURE, ONE_WAY_COST_BPS, REBALANCE_BAND, VOL_LOOKBACK,
    exposure_path,
)

UNIVERSE = {
    "USA": "us_single_name", "USB": "us_sector_etf", "USC": "us_broad_etf",
    "INT": "intl_equity_etf", "BND": "bond_etf", "CMD": "commodity_etf",
}


def _panel(days: int = 500, seed: int = 5) -> pd.DataFrame:
    """A panel with a volatility regime change, so the band actually fires."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=days)
    vol = np.where((np.arange(days) > 200) & (np.arange(days) < 300), 0.035, 0.007)
    return pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0.0002, 1.0, days) * vol))
         for s in UNIVERSE}, index=idx)


def _gross(panel: pd.DataFrame) -> pd.Series:
    return bucket_weighted_gross(panel, UNIVERSE, "equal", VOL_LOOKBACK,
                                 REBALANCE_BAND, ONE_WAY_COST_BPS)[0]


def test_band_holds_until_the_target_leaves_it_then_snaps_to_the_target():
    held, moved = band_exposure(0.905, 1.0, 0.10)
    assert (held, moved) == (1.0, False), "inside the band nothing moves"
    held, moved = band_exposure(0.88, 1.0, 0.10)
    assert (held, moved) == (0.88, True), "outside it, adopt the raw target"


def test_the_band_anchors_on_the_applied_target_not_the_drifted_actual():
    """The anchor choice, stated as a test.

    Anchoring on a drifted actual makes today's decision depend on every price
    move since the last trade, so two accounts following the same rule from the
    same date diverge permanently. Anchoring on the applied target keeps the
    path a function of prices alone.
    """
    # The account has drifted to 0.95 while the applied target is still 1.00.
    assert band_exposure(0.92, 1.00, 0.10)[1] is False, "8% from applied: hold"
    assert band_exposure(0.92, 0.95, 0.10)[1] is False
    # A target 12% from the applied one must trade regardless of the drift.
    assert band_exposure(0.88, 1.00, 0.10)[1] is True


def test_a_missing_anchor_enters_rather_than_skipping_the_band():
    exposure, moved = band_exposure(0.8, None, 0.10)
    assert (exposure, moved) == (0.8, True)
    assert band_exposure(0.8, float("nan"), 0.10) == (0.8, True)


def test_an_unusable_target_never_becomes_a_position():
    exposure, moved = band_exposure(float("nan"), 0.9, 0.10)
    assert np.isnan(exposure) and moved is False


def test_the_ledger_reproduces_the_backtest_exposure_path(tmp_path, monkeypatch):
    """The regression this whole file exists for.

    Advance the ledger over a panel with a volatility spike and compare its
    applied exposure, day by day, against `exposure_path` -- the function every
    published number was measured through.
    """
    panel = _panel()
    monkeypatch.setattr(pt, "JOURNAL", tmp_path / "journal.jsonl")
    monkeypatch.setattr(pt, "load_universe", lambda: (list(UNIVERSE), UNIVERSE))

    warm = VOL_LOOKBACK + 5
    monkeypatch.setattr(pt, "load_prices", lambda s: panel.iloc[:warm])
    args = type("Args", (), {"init": True, "capital": 1_000_000.0,
                             "target_vol": 0.15, "report": False,
                             "no_update": True})()
    pt._run(args)
    monkeypatch.setattr(pt, "load_prices", lambda s: panel)
    pt._run(type("Args", (), {"init": False, "capital": 1_000_000.0,
                              "target_vol": 0.15, "report": False,
                              "no_update": True})())

    rows = [json.loads(line) for line in pt.JOURNAL.read_text().splitlines()
            if line.strip()]
    ledger = {r["date"]: r["exposure_applied"] for r in rows
              if r["action"] != "init" and r["exposure_applied"] is not None}
    assert len(ledger) > 300, "not enough days to be a real comparison"

    reference = exposure_path(_gross(panel), 0.15)
    # The ledger sizes at day D's close off vol through D, and holds that into
    # D+1; exposure_path assigns vol-through-D to row D+1. Same rule, so the
    # reference for a ledger row is the next row of the path.
    positions = {str(d.date()): i for i, d in enumerate(reference.index)}
    compared = worst = 0
    for date, applied in ledger.items():
        nxt = positions[date] + 1
        if nxt >= len(reference):
            continue
        expected = reference.iloc[nxt]
        if not np.isfinite(expected):
            continue
        compared += 1
        worst = max(worst, abs(applied - expected))

    assert compared > 300, f"only compared {compared} days"
    assert worst < 5e-4, (
        f"ledger exposure diverges from the measured path by {worst:.4f}; "
        f"before the anchors were unified this reached 0.294")


def test_the_drifted_anchor_really_did_diverge():
    """Guard the premise. If a drifted anchor tracked the applied one closely,
    unifying them would be pedantry rather than a fix -- so measure the gap the
    old ledger actually carried."""
    panel = _panel()
    gross = _gross(panel)
    reference = exposure_path(gross, 0.15)

    # Replay the old rule: anchor on exposure after it drifts with returns.
    drifted, held = [], np.nan
    realized = gross.rolling(VOL_LOOKBACK).std(ddof=1) * np.sqrt(252)
    raw = (0.15 / realized.shift(1)).clip(upper=MAX_EXPOSURE)
    for day, want in raw.items():
        if not np.isfinite(want):
            drifted.append(np.nan)
            continue
        if not np.isfinite(held) or abs(want - held) / max(held, 1e-9) > REBALANCE_BAND:
            held = want
        drifted.append(held)
        # Price drift: the invested sleeve moves, cash does not.
        r = gross.get(day, 0.0)
        if np.isfinite(r):
            held = held * (1 + r) / (1 + held * r) if (1 + held * r) else held

    gap = (pd.Series(drifted, index=raw.index) - reference).abs().dropna()
    assert gap.max() > 0.05, (
        "the two anchors are meant to differ materially; if this fails the "
        "premise of unifying them needs rechecking")
