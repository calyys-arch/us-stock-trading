"""
fixed_universe tests: ranking correctness (top-N by trailing dollar volume,
strictly before as_of), YAML round-trip, missing-file error message, and
fingerprint stability.
"""
from __future__ import annotations

import pandas as pd
import pytest

from python.data.fixed_universe import (
    load_universe_config,
    save_universe_config,
    select_fixed_top_n,
    universe_fingerprint,
)


def _panel(volumes: dict[str, float], n_days: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rows = []
    for code, volume in volumes.items():
        for d in dates:
            rows.append({"date": d, "code": code, "close": 100.0, "volume": volume})
    return pd.DataFrame(rows).set_index(["date", "code"]).sort_index()


def test_select_top_n_ranks_by_dollar_volume():
    panel = _panel({"LOW": 1e5, "MID": 1e6, "HIGH": 1e7, "TOP": 1e8})
    as_of = pd.Timestamp("2024-04-01")
    selected = select_fixed_top_n(["LOW", "MID", "HIGH", "TOP"], panel, as_of,
                                  top_n=2, lookback_days=60)
    assert selected == sorted(["TOP", "HIGH"])


def test_select_ignores_candidates_not_in_pool():
    panel = _panel({"AAA": 1e8, "BBB": 1e6, "CCC": 1e7})
    selected = select_fixed_top_n(["BBB", "CCC"], panel, pd.Timestamp("2024-04-01"),
                                  top_n=2, lookback_days=60)
    assert "AAA" not in selected
    assert selected == sorted(["BBB", "CCC"])


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "universe.yaml"
    save_universe_config(["msft", "AAPL"], pd.Timestamp("2026-07-28"), 60,
                         source_pool_label="test_pool", path=path)
    cfg = load_universe_config(path)
    assert cfg["symbols"] == ["AAPL", "MSFT"]   # upper-cased and sorted
    assert cfg["top_n"] == 2
    assert cfg["ranking_metric"] == "trailing_60d_avg_dollar_volume"
    assert cfg["computed_at"] == "2026-07-28"
    assert cfg["source_pool"] == "test_pool"


def test_load_missing_file_names_the_refresh_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="refresh_universe"):
        load_universe_config(tmp_path / "nope.yaml")


def test_fingerprint_changes_with_symbols_and_date(tmp_path):
    base = {"symbols": ["AAPL", "MSFT"], "computed_at": "2026-07-28"}
    same = {"symbols": ["MSFT", "AAPL"], "computed_at": "2026-07-28"}  # order-insensitive
    other_syms = {"symbols": ["AAPL", "NVDA"], "computed_at": "2026-07-28"}
    other_date = {"symbols": ["AAPL", "MSFT"], "computed_at": "2026-01-01"}
    assert universe_fingerprint(base) == universe_fingerprint(same)
    assert universe_fingerprint(base) != universe_fingerprint(other_syms)
    assert universe_fingerprint(base) != universe_fingerprint(other_date)
