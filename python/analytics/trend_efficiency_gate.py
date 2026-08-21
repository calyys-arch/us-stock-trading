"""
Phase 2 regime-GATING classifier (backtests/reports/regime_gate_report.md).

THIS IS A SEPARATE MODULE FROM `python/analytics/regime.py` — that module's
Bull/Bear/Sideways Markov classifier, its parameters, and its behavior are
UNCHANGED by this file (per this task's explicit constraint: do not modify
`regime.py`'s existing behavior used by the retired `orb_vwap_regime`).

WHY A DIFFERENT FEATURE, NOT JUST A DIFFERENT USE OF THE SAME ONE
------------------------------------------------------------------
`orb_vwap_regime` (RETIRED, `python/microstructure/signals/orb_vwap_regime.py`)
gated a TREND-FOLLOWING signal to fire only on days `regime.py` labeled
Bull/Bear (not Sideways) via a 20-day/2% rolling-return threshold. That
regime filter made the gated strategy's pass ratio and mean OOS Sharpe WORSE
than the ungated baseline (`strategy_review_summary.md` §3.4) — a plausible-
sounding proxy ("is today trending") that did not actually predict what it
was assumed to predict ("will today whipsaw").

This module gates the OPPOSITE strategy family (mean-reversion, not
trend-following) with the OPPOSITE polarity (ON when the market is NOT in a
persistent trend) using a DIFFERENT feature: trend efficiency (net drift /
total path length over a trailing window), not a rolling-return
Bull/Bear/Sideways label. The feature choice is deliberate, not incidental:
`backtests/reports/regime_generalization_report.md` §1b already found,
BEFORE this classifier was designed and using nothing computed by this
module, that realized volatility does NOT cleanly separate the
mean-reversion-favorable windows (2020, 2022-for-pairs) from the
mean-reversion-hostile ones (2022-for-xsection, 2024-2026) — single-name
realized vol was high in every window tested (26-44% annualized) — but TREND
PERSISTENCE does: 2024-2026 (+448%, one direction), 2022 (-42%, one
direction) and 2018-2019 are all persistent trends; 2020 is a crash-then-full
-recovery (genuine two-way movement despite also being high-vol). That
already-published finding is why this module measures trend efficiency, not
realized-vol-vs-its-own-distribution (the task's other candidate) — a design
decision made from a pre-existing, independently-produced report's economic
finding, not by looking at how any candidate classifier would label the
known periods (see the honesty contract below for what that distinction
does NOT permit).

FEATURE: Kaufman-style Efficiency Ratio (net drift / path length)
-------------------------------------------------------------------
    ER(t) = |P(t) - P(t-N)| / sum_{i=t-N+1..t} |P(i) - P(i-1)|

ER in [0, 1]. ER near 1: nearly all movement over the window was in one
direction (a clean persistent trend). ER near 0: the price round-tripped a
lot relative to its net displacement (mean-reverting / choppy). This is a
well-known technical measure (Kaufman, "New Trading Systems and Methods",
the basis of the Kaufman Adaptive Moving Average) computed here from
scratch against this repo's own data, exactly as `regime.py`'s own
provenance note handles its external source.

GATE RULE: relative to the feature's OWN trailing distribution, not an
absolute magic number — "is today's trend efficiency lower than usual for
this instrument recently" rather than "is ER < 0.3" (an arbitrary constant
that would not obviously generalize across instruments or eras). The gate is
ON (mean-reversion strategies may trade) when the current N-day ER is at or
below its own trailing M-day median; OFF when ER is above its own trailing
median (an unusually persistent trend relative to this instrument's recent
past).

Free parameters (2): `window` (N, the ER lookback, trading days), `
reference_window` (M, the trailing window the current ER is compared
against). BOTH are fixed at conventional, round-number values chosen BEFORE
this module was ever run against any specific historical window and before
any gated backtest was computed: `window=20` (one trading month — the same
timescale `regime.py`'s own `window` default uses, chosen here independently
for the same "about a month of price action" convention, not copied logic),
`reference_window=252` (one trading year — the same timescale `regime.py`'s
own `min_train` uses for "how much history before a signal is trusted").
Neither value was tuned, gridded, or adjusted after seeing how it labels
2008, 2018, 2020, 2022, 2024-2026, or any other period — see
`backtests/reports/regime_gate_report.md` §2 for the falsifiable,
pre-registered validation this module's actual gating benefit is judged by
(full-history WFO gated-vs-ungated comparison), NOT retroactive agreement
with the 3-4 windows already known to be favorable/unfavorable.

NO-LOOKAHEAD DISCIPLINE (same standard as `regime.py`'s own honesty
contract and `orb_vwap_regime`'s `.shift(1)` pattern)
-------------------------------------------------------------------
`compute_gate_labels` returns a label for day t computed ONLY from prices up
to and including day t's close. Callers that want a point-in-time,
no-lookahead trading signal for "may I open a new position on day t" MUST
shift this by one day (`labels.shift(1)`) before using it, exactly as
`orb_vwap_regime`'s module docstring already documents for its own label —
this module does not shift internally so that `tests/test_trend_efficiency_
gate.py` can pin the RAW (unshifted) label's causality boundary precisely
(label(t) uses close[0..t], nothing after), the same convention
`regime.label_regimes` uses for its own raw (unshifted) output.

This module is REPORT/GATE-ONLY diagnostic infrastructure of the same
status as `regime.py`: it is not itself a trading strategy, has no
`auto_execute` wiring, and its use as an actual entry filter (Phase 2 of
`regime_gate_report.md`) went through the same WFO/Monte Carlo gating
discipline as every other strategy component in this repo before any GO/
NO-GO verdict was drawn from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 20
DEFAULT_REFERENCE_WINDOW = 252


def efficiency_ratio(close: pd.Series, window: int = DEFAULT_WINDOW) -> pd.Series:
    """Kaufman-style trend efficiency ratio, causal (uses only close[i-window..i]
    for each output row i). NaN for the first `window` rows (insufficient
    history), matching `regime.label_regimes`'s convention of not
    default-labeling rows that don't have a full trailing window yet."""
    if window <= 0:
        raise ValueError("window must be positive")
    net_drift = (close - close.shift(window)).abs()
    path_length = close.diff().abs().rolling(window, min_periods=window).sum()
    er = net_drift / path_length
    # Zero path length (a flat, unchanged price for `window` days) makes the
    # ratio undefined, not "maximally trending" or "maximally choppy" —
    # leave it NaN rather than injecting a fabricated 0/0 -> 1.0 or 0.0.
    er = er.where(path_length > 0)
    return er.clip(lower=0.0, upper=1.0)


def compute_gate_labels(
    close: pd.Series,
    window: int = DEFAULT_WINDOW,
    reference_window: int = DEFAULT_REFERENCE_WINDOW,
) -> pd.Series:
    """Boolean series, True = "mean-reversion strategies may trade" (today's
    trend efficiency is at or below its own trailing `reference_window`-day
    median), False = "persistent trend relative to this instrument's recent
    past, mean-reversion strategies should sit out". RAW label for day t
    uses close[0..t] only — see module docstring for the caller-side
    `.shift(1)` requirement before using this as an entry filter."""
    er = efficiency_ratio(close, window=window)
    # `min_periods=reference_window` so the reference distribution itself is
    # never shorter than declared — an early partial-window "median" would
    # silently be a different, looser rule than the one being tested.
    trailing_median = er.rolling(reference_window, min_periods=reference_window).median()
    gate = er <= trailing_median
    # Both `er` and `trailing_median` are NaN for the same leading rows;
    # comparisons against NaN are already False, but express the "not yet
    # decidable" rows explicitly as NaN rather than a silent False (which
    # would look like a real "trend detected, gate off" decision).
    undecided = er.isna() | trailing_median.isna()
    return gate.astype("boolean").mask(undecided, pd.NA)


def shifted_entry_gate(
    close: pd.Series,
    window: int = DEFAULT_WINDOW,
    reference_window: int = DEFAULT_REFERENCE_WINDOW,
) -> pd.Series:
    """The actual point-in-time signal a backtest engine may use to decide
    whether NEW positions may be opened on day t: `compute_gate_labels`
    shifted by one day (today's entry decision may only depend on yesterday's
    close), matching `orb_vwap_regime`'s documented `.shift(1)` convention.
    Undecided rows (insufficient history, or the first day with no prior
    label at all) default to False — "not enough history to say it's safe to
    trade" behaves as a closed gate, not an open one."""
    raw = compute_gate_labels(close, window=window, reference_window=reference_window)
    shifted = raw.shift(1)
    return shifted.fillna(False).astype(bool)


def live_entry_allowed(
    close: pd.Series,
    as_of=None,
    window: int = DEFAULT_WINDOW,
    reference_window: int = DEFAULT_REFERENCE_WINDOW,
) -> bool:
    """Point-in-time answer to "may pairs open a NEW position as of `as_of`?"

    When `as_of` is already in `close` (a completed daily bar, the backtest
    case), this is exactly `shifted_entry_gate[as_of]` — today's decision
    uses only yesterday's close.

    When `as_of` is after the last close in `close` (live RTH: today's bar
    is not closed yet), the last raw label — computed from the last
    completed close, i.e. yesterday — is used. That is the same information
    `shifted_entry_gate` would expose on the next session's row. Missing or
    undecided history fails CLOSED (False).
    """
    raw = compute_gate_labels(close, window=window, reference_window=reference_window)
    if as_of is not None:
        ts = pd.Timestamp(as_of).normalize()
        shifted = raw.shift(1)
        if ts in shifted.index and pd.notna(shifted.loc[ts]):
            return bool(shifted.loc[ts])
    decided = raw.dropna()
    if decided.empty:
        return False
    return bool(decided.iloc[-1])


def load_regime_proxy_close(
    symbol: str = "SPY",
    lookback_days: int = 800,
    cache_path=None,
) -> pd.Series | None:
    """Daily close for the live tape classifier. Cache-first, then Futu
    daily K-line — never IBKR/yfinance (those hang or 401 in this
    environment and would block absorption's gate policy). Insufficient
    history returns None so the caller stays fail-closed."""
    from pathlib import Path

    cache = Path(cache_path) if cache_path is not None else Path("data/history") / f"{symbol.upper()}.csv"
    end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=max(int(lookback_days), 1))
    min_rows = DEFAULT_WINDOW + DEFAULT_REFERENCE_WINDOW

    if cache.exists():
        raw = pd.read_csv(cache, index_col=0, parse_dates=True)
        col = "close" if "close" in raw.columns else raw.columns[0]
        cached = raw[col].astype(float)
        idx = pd.to_datetime(cached.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        cached.index = idx.normalize()
        cached = cached.sort_index()
        cached = cached.loc[cached.index >= start]
        if len(cached.dropna()) >= min_rows:
            return cached.dropna()

    try:
        from futu import KLType

        from python.data.futu_price_source import (
            fetch_history_kline_range,
            open_futu_quote_context,
        )
    except Exception:
        return None

    ctx = None
    try:
        ctx = open_futu_quote_context()
        df = fetch_history_kline_range(ctx, symbol, start, end, ktype=KLType.K_DAY)
    except Exception:
        return None
    finally:
        if ctx is not None:
            try:
                ctx.close()
            except Exception:
                pass

    if df is None or df.empty or "close" not in df.columns:
        return None
    series = df["close"].astype(float)
    idx = pd.to_datetime(series.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    series.index = idx.normalize()
    series = series[~series.index.duplicated(keep="last")].sort_index().dropna()
    if len(series) < min_rows:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    series.rename("close").to_csv(cache, header=True)
    return series
