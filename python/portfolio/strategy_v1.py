"""Universe, bucket map and weighting for Strategy V1.

Both `scripts/run_risk_bucket_study.py` (which measures the weighting) and
`scripts/strategy_v1_positions.py` (which turns it into orders) import from
here. If they each kept their own copy, the sheet could drift away from the
backtest that justified it and nothing would flag the divergence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_FILE = ROOT / "config/strategy_v1_universe.json"
HISTORY = ROOT / "data/history"

# The three US equity groups collapse into one bucket because they are one
# beta: SPY holds NVDA, XLK holds NVDA, SMH holds NVDA, and NVDA is also held
# on its own. Counting them separately is what let a ticker-count allocation
# put 69% of the book in US equity by accident.
COLLAPSE = {
    "us_single_name": "us_equity",
    "us_sector_etf": "us_equity",
    "us_broad_etf": "us_equity",
    "intl_equity_etf": "intl_equity",
    "bond_etf": "bonds",
    "commodity_etf": "commodities",
}
BUCKETS = tuple(sorted(set(COLLAPSE.values())))


def load_universe() -> tuple[list[str], dict[str, str]]:
    """Frozen instrument list and its fine-grained bucket label per symbol."""
    spec = json.loads(UNIVERSE_FILE.read_text())
    label: dict[str, str] = {}
    for bucket, syms in spec["buckets"].items():
        for s in syms:
            label[s] = bucket
    return sorted(label), label


def load_prices(symbols: list[str]) -> pd.DataFrame:
    """Wide close-price frame for whichever of `symbols` are cached."""
    cols = {}
    for s in symbols:
        path = HISTORY / f"{s}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
            cols[s] = df["close"].astype(float)
    return pd.DataFrame(cols).sort_index()


def collapsed_of(bucket_of: dict[str, str]) -> dict[str, str]:
    return {s: COLLAPSE.get(b, "") for s, b in bucket_of.items()}


def bucket_members(symbols: list[str], bucket_of: dict[str, str]
                   ) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    coll = collapsed_of(bucket_of)
    for s in symbols:
        bucket = coll.get(s, "")
        if bucket:
            out.setdefault(bucket, []).append(s)
    return {k: sorted(v) for k, v in out.items()}


def target_weights(symbols: list[str], bucket_of: dict[str, str],
                   scheme: str) -> dict[str, float]:
    """Per-symbol fraction of the invested book, before the volatility brake.

    "naive" is one share per instrument, which is the ticker-count artifact.
    "bucket" splits equally across the four collapsed buckets and then equally
    within each, so the equity tilt is a decision rather than a side effect of
    how many tickers happen to be cached.
    """
    if scheme == "naive":
        return {s: 1.0 / len(symbols) for s in symbols}
    members = bucket_members(symbols, bucket_of)
    per_bucket = 1.0 / len(members)
    return {s: per_bucket / len(syms)
            for syms in members.values() for s in syms}


def bucket_returns(panel: pd.DataFrame, bucket_of: dict[str, str]
                   ) -> pd.DataFrame:
    """Equal-weight daily return of each collapsed bucket.

    Weights come from the prior close via `shift(1)`, so a name entering the
    panel is not credited with the return of the day it appeared.
    """
    rets = panel.pct_change(fill_method=None)
    hold = panel.notna().astype(float).shift(1).fillna(0.0)
    coll = collapsed_of(bucket_of)
    out = {}
    for bucket in BUCKETS:
        cols = [c for c in panel.columns if coll.get(c, "") == bucket]
        if not cols:
            continue
        h = hold[cols]
        n = h.sum(axis=1)
        w = h.div(n.where(n > 0), axis=0).fillna(0.0)
        out[bucket] = (w * rets[cols]).sum(axis=1)
    return pd.DataFrame(out)


def banded(weights: pd.DataFrame, band: float) -> pd.DataFrame:
    """Hold bucket weights until any one drifts more than `band` relative."""
    rows, current = [], None
    for _, row in weights.iterrows():
        if row.isna().any():
            rows.append([np.nan] * len(row))
            continue
        if current is None:
            current = row.to_numpy()
        else:
            drift = np.abs(row.to_numpy() - current) / np.maximum(current, 1e-9)
            if drift.max() > band:
                current = row.to_numpy()
        rows.append(list(current))
    return pd.DataFrame(rows, index=weights.index, columns=weights.columns)


def bucket_weighted_gross(panel: pd.DataFrame, bucket_of: dict[str, str],
                          scheme: str, vol_lookback: int, band: float,
                          one_way_cost_bps: float
                          ) -> tuple[pd.Series, pd.DataFrame]:
    """Bucket-level allocation, equal weight inside each bucket.

    `scheme` is "rp" for inverse trailing volatility or "equal" for a flat
    share per bucket. Volatility is measured over days strictly BEFORE the day
    it weights (`.shift(1)` after the rolling window), so no day's allocation
    is informed by its own return. Bucket turnover pays the same one-way cost
    the volatility brake pays; the equal scheme has none by construction.
    """
    buckets = bucket_returns(panel, bucket_of)
    if scheme == "equal":
        weights = pd.DataFrame(1.0 / buckets.shape[1], index=buckets.index,
                               columns=buckets.columns)
    else:
        vol = buckets.rolling(vol_lookback).std(ddof=1).shift(1)
        inv = 1.0 / vol.where(vol > 0)
        weights = banded(inv.div(inv.sum(axis=1), axis=0), band)

    combined = (weights * buckets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    net = combined - turnover * (one_way_cost_bps / 10_000.0)
    valid = weights.notna().all(axis=1)
    return net[valid].dropna(), weights[valid]
