"""
G-Mamba-lite Phase 0 kill-test — same falsification-first discipline as
scripts/run_v2_momentum_phase0.py and scripts/run_v2_8k_drift_phase0.py,
applied to the graph-conditioned selective-state-space architecture the
user's other repo (calyys-arch/Universal-Hybrid-AI, private, cloned
2026-09-15) just added to its `trading-signals` pack
(industry-packs/trading-signals/forecasting/gmamba_forecast.gcl +
python-svc/gmamba_signal.py, commit 9ae2f54 "add G-Mamba graph-conditioned
forecasting").

Question this answers: on the SAME 127-name mid+large-cap combined
universe where 12-1 cross-sectional momentum found nothing (spread
t-stat 0.16, docs/v2_research_plan.md \u00a76) and 8-K filing drift found
nothing significant (spread t-stat 0.84, docs/v2_research_plan.md \u00a77),
does a much higher-capacity, graph-aware sequence model find any genuine
out-of-sample cross-sectional signal, or does it just confirm the same
null with extra steps?

Why this is NOT wired into configs/strategy.yaml / param_guard.py /
self_improve_loop.py, and never will be without a separate decision:
python/backtest/param_guard.py caps every live/paper strategy at
MAX_FREE_PARAMETERS=5 (Chan discipline). The model built here
(python/research/gmamba_lite.py) has on the order of several thousand
trainable weights -- this script prints the exact count at startup so
that number is never hand-waved. A model with three orders of magnitude
more capacity than anything else in this repo, fit on ~100 non-overlapping
cross-sections of ~127 noisy names, is at HIGHER overfitting risk than
12-1 momentum's single ranking rule, not lower -- so this is held to the
SAME bar (a pre-declared, mechanical falsification check on genuinely
held-out data), not a looser one just because it is "smarter."

Universe and data: identical to run_v2_momentum_phase0.py's --universe
combined (midcap: data/history_altuni_midcap_full/, 71 symbols, full
history 2016-11-01 to 2026-07-31; largecap: data/history/ symbols with
>=13-month history, 57 symbols; 127 total after the 1 known overlap).
Imported directly from that script rather than re-implemented, so both
Phase 0 scripts read the exact same on-disk CSVs the exact same way.

Features (per stock, per trading day, OHLCV-derived only -- no Wyckoff/VSA
labels exist as a historical archive for this universe, unlike the
Universal-Hybrid-AI pack's `signal_outcome_jsonl.gcl`-fed data; see
docs/gmamba_phase0_results.md \u00a71 for why substituting these 7 technical
features is the honest option here, not a shortcut):
    ret_1d, ret_5d, ret_21d        -- trailing returns (ret_21d is the
                                       same lookback 12-1 momentum used,
                                       minus the skip-month; the model sees
                                       strictly less information than that
                                       already-tested signal, not more)
    vol_20d                        -- 20-day realized volatility
    dollar_vol_z                   -- 20-day dollar-volume z-score
    dist_from_high_20d             -- (close / 20d rolling max) - 1
    dist_from_low_20d              -- (close / 20d rolling min) - 1
Each is cross-sectionally rank-normalized to [0, 1] per day (pandas
`.rank(pct=True)`) before windowing -- bounded, comparable across names
and regimes, the same spirit as the original pack's composite_score/100.

Graph: static_adj is an all-zero (V, V) placeholder, NOT a fabricated
prior. This universe has no verified sector/ETF-co-holding data on file
(configs/*.yaml has sector ETF *tickers* for pairs_trading, not a
per-symbol GICS map for these 127 names) -- inventing one would be
building a result, not testing one. mix_adjacency's learned alpha can
still put weight on the model's own DynamicAdjacency (eq. 16-17), so this
tests "does a graph-conditioned architecture with a LEARNED graph help",
not "does this specific static graph help". A real static prior is future
work IF this Phase 0 clears its bar -- it will not, by itself, be a reason
to lower it.

Windowing: as-of dates on a fixed 21-trading-day stride (~monthly, matching
momentum's own rebalance frequency for comparability), each sample =
whole (L=20, V=127, C=7) cross-section ending at that date, target =
forward 21-trading-day return per stock (masked where unavailable: name
not yet listed, or too close to the end of history for a forward return).

Split: chronological train/val with a 21-trading-day embargo gap between
them (train's labels are outcomes of price action UP TO 21 days after
train's last as-of date; without the gap, val's early samples would share
label information with train's last few -- the same purge/embargo logic
Universal-Hybrid-AI's own train_gmamba_signal.py documents for the same
reason). No early stopping / no best-epoch checkpoint selection on val:
a FIXED epoch count is trained and val is scored ONLY at that final epoch.
Picking the best-val-IC epoch (which the upstream trainer does) is itself
a mild form of snooping on the only measurement that decides this
verdict; a Phase 0 kill-test cannot afford that even if it is common
practice elsewhere.

Falsification criteria (declared here, BEFORE running, same mechanical
_verdict() pattern as the other two Phase 0 scripts):
    NO-GO if mean per-date IC <= 0, OR |t-stat of per-date IC| < 2.0
    "PASS Phase 0" (not a promotion, not production-ready) otherwise.

Usage:
    .venv/bin/python scripts/run_gmamba_phase0.py
    .venv/bin/python scripts/run_gmamba_phase0.py --epochs 80 --seed 7
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.research.gmamba_lite import GMambaConfig, GMambaSignal, prediction_loss
from scripts.run_v2_momentum_phase0 import (
    LARGECAP_DIR,
    MIDCAP_DIR,
    _largecap_single_name_symbols,
    _load_close_and_dollar_volume_multi,
    _one_sample_t_stat,
)

log = logging.getLogger(__name__)

OUT_DIR = Path("backtests/gmamba_phase0")
LOOKBACK_DAYS = 20          # L -- same as the original pack's lookbackDays()
FORWARD_DAYS = 21           # target horizon, same as momentum/8-K Phase 0
STRIDE_DAYS = 21            # ~monthly as-of spacing between samples
EMBARGO_DAYS = FORWARD_DAYS  # gap between train's last as-of date and val's first
TRAIN_FRACTION = 0.7
MIN_VALID_NAMES_PER_DATE = 10   # same cross-section floor as the momentum script
DEFAULT_EPOCHS = 60
DEFAULT_SEED = 42

FEATURE_NAMES = [
    "ret_1d", "ret_5d", "ret_21d", "vol_20d",
    "dollar_vol_z", "dist_from_high_20d", "dist_from_low_20d",
]


def _load_universe() -> tuple[pd.DataFrame, pd.DataFrame]:
    mid_syms = sorted(p.stem for p in MIDCAP_DIR.glob("*.csv"))
    large_syms = _largecap_single_name_symbols()
    return _load_close_and_dollar_volume_multi(
        [(MIDCAP_DIR, mid_syms), (LARGECAP_DIR, large_syms)])


def build_raw_features(close: pd.DataFrame, dollar_vol: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ret = close.pct_change()
    roll_max = close.rolling(20).max()
    roll_min = close.rolling(20).min()
    dv_mean = dollar_vol.rolling(20).mean()
    dv_std = dollar_vol.rolling(20).std()

    return {
        "ret_1d": ret,
        "ret_5d": close.pct_change(5),
        "ret_21d": close.pct_change(21),
        "vol_20d": ret.rolling(20).std(),
        "dollar_vol_z": (dollar_vol - dv_mean) / dv_std.replace(0.0, np.nan),
        "dist_from_high_20d": close / roll_max - 1.0,
        "dist_from_low_20d": close / roll_min - 1.0,
    }


def rank_normalize(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Cross-sectional percentile rank per day, per feature -> [0, 1].
    NaN stays NaN (no cross-sectional data that day for that name) --
    filled with 0 only at tensor-build time, matching the original pack's
    zero-pad-for-missing-history convention, and always paired with the
    validity mask so a zero-fill is never mistaken for a real observation.
    """
    return {name: df.rank(axis=1, pct=True) for name, df in raw.items()}


def build_dataset(
    features: dict[str, pd.DataFrame],
    fwd_return: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[pd.Timestamp], list[str]]:
    """Returns (X, y, mask, as_of_dates, symbols).
    X: (N, L, V, C).  y: (N, V).  mask: (N, V) bool -- True where this
    (sample, symbol) has both a complete-enough feature window (its as-of
    day's row is non-NaN across all C features) AND a known forward return.
    """
    symbols = list(next(iter(features.values())).columns)
    calendar = next(iter(features.values())).index
    n_days = len(calendar)

    warmup = 21  # longest single rolling window used above (ret_21d / 20d rollings)
    first_idx = warmup + LOOKBACK_DAYS
    last_idx = n_days - FORWARD_DAYS - 1

    feature_arrays = {name: df.to_numpy(dtype=np.float32) for name, df in features.items()}
    fwd_arr = fwd_return.to_numpy(dtype=np.float32)

    xs, ys, masks, as_of_dates = [], [], [], []
    for idx in range(first_idx, last_idx + 1, STRIDE_DAYS):
        window = np.stack(
            [np.nan_to_num(feature_arrays[name][idx - LOOKBACK_DAYS + 1: idx + 1], nan=0.0)
             for name in FEATURE_NAMES],
            axis=-1,
        )  # (L, V, C)
        if window.shape[0] != LOOKBACK_DAYS:
            continue

        target = fwd_arr[idx]  # (V,)
        as_of_valid = ~np.isnan(
            np.stack([feature_arrays[name][idx] for name in FEATURE_NAMES], axis=-1)
        ).any(axis=-1)
        valid = as_of_valid & ~np.isnan(target)

        if valid.sum() < MIN_VALID_NAMES_PER_DATE:
            continue

        xs.append(window)
        ys.append(np.nan_to_num(target, nan=0.0))
        masks.append(valid)
        as_of_dates.append(calendar[idx])

    X = torch.tensor(np.stack(xs), dtype=torch.float32)
    y = torch.tensor(np.stack(ys), dtype=torch.float32)
    mask = torch.tensor(np.stack(masks), dtype=torch.bool)
    return X, y, mask, as_of_dates, symbols


def masked_l1_loss(y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """y_hat: (B, 1, V). y/mask: (B, V)."""
    diff = (y_hat.squeeze(1) - y).abs()
    denom = mask.sum().clamp_min(1)
    return (diff * mask.float()).sum() / denom


def per_date_ic(y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> list[float]:
    """Pearson correlation between predicted score and actual forward
    return, computed SEPARATELY per as-of date (Fama-MacBeth style) rather
    than pooled across dates -- pooling would let market-wide moves on a
    handful of dates dominate the correlation and understate how clustered
    (non-independent) the observations are, the same reason the momentum/
    8-K scripts run a one-sample t-test on a per-period spread series
    rather than one giant pooled regression."""
    ics = []
    yh = y_hat.squeeze(1).detach().numpy()
    yt = y.detach().numpy()
    m = mask.numpy()
    for i in range(yh.shape[0]):
        idx = m[i]
        if idx.sum() < MIN_VALID_NAMES_PER_DATE:
            continue
        a, b = yh[i, idx], yt[i, idx]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        ics.append(float(np.corrcoef(a, b)[0, 1]))
    return ics


def tercile_spread(y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> list[float]:
    """Top-tercile-predicted minus bottom-tercile-predicted mean ACTUAL
    forward return, per as-of date -- the same long/short spread construct
    run_v2_momentum_phase0.py uses, so the two verdicts are read the same
    way."""
    spreads = []
    yh = y_hat.squeeze(1).detach().numpy()
    yt = y.detach().numpy()
    m = mask.numpy()
    for i in range(yh.shape[0]):
        idx = np.where(m[i])[0]
        if len(idx) < MIN_VALID_NAMES_PER_DATE:
            continue
        order = idx[np.argsort(-yh[i, idx])]
        n_tercile = max(1, len(order) // 3)
        top, bottom = order[:n_tercile], order[-n_tercile:]
        spreads.append(float(yt[i, top].mean() - yt[i, bottom].mean()))
    return spreads


def run_phase0(epochs: int, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    close, dollar_vol = _load_universe()
    raw = build_raw_features(close, dollar_vol)
    normalized = rank_normalize(raw)
    fwd_return = close.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)

    X, y, mask, as_of_dates, symbols = build_dataset(normalized, fwd_return)
    n_samples, _, n_symbols, n_features = X.shape
    log.info("Built %d cross-sectional samples over %d symbols, %s .. %s",
              n_samples, n_symbols, as_of_dates[0].date(), as_of_dates[-1].date())

    n_train = int(n_samples * TRAIN_FRACTION)
    n_embargo = max(1, EMBARGO_DAYS // STRIDE_DAYS)
    val_start = n_train + n_embargo
    if val_start >= n_samples - 5:
        raise ValueError("not enough samples for a train/embargo/val split at this stride")

    X_train, y_train, mask_train = X[:n_train], y[:n_train], mask[:n_train]
    X_val, y_val, mask_val = X[val_start:], y[val_start:], mask[val_start:]
    val_dates = as_of_dates[val_start:]
    log.info("train: %d samples (%s .. %s) | embargo: %d samples dropped | val: %d samples (%s .. %s)",
              n_train, as_of_dates[0].date(), as_of_dates[n_train - 1].date(),
              n_embargo, len(val_dates), val_dates[0].date(), val_dates[-1].date())

    static_adj = torch.zeros(n_symbols, n_symbols)  # disclosed placeholder, see module docstring

    cfg = GMambaConfig(n_features=n_features, d_model=16, n_layers=1, d_state=8,
                        expand=2, d_attn=8, dropout=0.1, horizon=1)
    model = GMambaSignal(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("GMambaSignal-lite: %d trainable parameters (%.0fx over param_guard.py's "
              "MAX_FREE_PARAMETERS=5 -- see module docstring; this is a research script, "
              "not something param_guard.py or self_improve_loop.py can ever touch)",
              n_params, n_params / 5)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_hat_train, _ = model(X_train, static_adj)
        loss = masked_l1_loss(y_hat_train, y_train, mask_train)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            y_hat_val, _ = model(X_val, static_adj)
            val_loss = masked_l1_loss(y_hat_val, y_val, mask_val)
        val_losses.append(float(val_loss.item()))

    # Scored ONLY at the final, pre-declared epoch -- see module docstring
    # on why best-epoch selection on val is not used here.
    model.eval()
    with torch.no_grad():
        y_hat_val, _ = model(X_val, static_adj)

    ics = per_date_ic(y_hat_val, y_val, mask_val)
    spreads = tercile_spread(y_hat_val, y_val, mask_val)
    spread_series = pd.Series(spreads, index=val_dates[:len(spreads)])

    mean_ic = float(np.mean(ics)) if ics else 0.0
    ic_t_stat = _one_sample_t_stat(pd.Series(ics)) if ics else 0.0
    spread_sharpe = (
        float(spread_series.mean() / spread_series.std(ddof=1) * np.sqrt(12))
        if len(spread_series) > 1 and spread_series.std(ddof=1) > 0 else 0.0
    )

    return {
        "n_params": n_params,
        "n_train_samples": n_train,
        "n_val_samples": len(val_dates),
        "n_symbols": n_symbols,
        "n_features": n_features,
        "epochs": epochs,
        "seed": seed,
        "train_loss_final": train_losses[-1],
        "val_loss_final": val_losses[-1],
        "val_date_range": [str(val_dates[0].date()), str(val_dates[-1].date())],
        "n_val_dates_with_ic": len(ics),
        "mean_daily_ic": mean_ic,
        "ic_t_stat": ic_t_stat,
        "tercile_spread_mean_monthly": float(spread_series.mean()) if len(spread_series) else 0.0,
        "tercile_spread_sharpe_annualized_gross": spread_sharpe,
    }


def _verdict(result: dict) -> str:
    reasons = []
    if result["mean_daily_ic"] <= 0:
        reasons.append(f"mean daily IC {result['mean_daily_ic']:.4f} <= 0")
    if abs(result["ic_t_stat"]) < 2.0:
        reasons.append(f"IC t-stat {result['ic_t_stat']:.2f} (|t| < 2 -> no reliable "
                        f"cross-sectional discrimination on held-out data)")
    return "NO-GO (Phase 0): " + "; ".join(reasons) if reasons else \
        "PASS Phase 0 (research-only signal; NOT eligible for param_guard.py/" \
        "self_improve_loop.py without a dedicated ML validation track -- see module docstring)"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    result = run_phase0(args.epochs, args.seed)
    verdict = _verdict(result)
    result["verdict"] = verdict

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"phase0_seed{args.seed}_epochs{args.epochs}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    log.info("")
    for key in ("n_params", "n_train_samples", "n_val_samples", "val_date_range",
                "n_val_dates_with_ic", "mean_daily_ic", "ic_t_stat",
                "tercile_spread_mean_monthly", "tercile_spread_sharpe_annualized_gross"):
        log.info("%-38s %s", key, result[key])
    log.info("")
    log.info("VERDICT: %s", verdict)
    log.info("Result written to %s", out_path)


if __name__ == "__main__":
    main()
