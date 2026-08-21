"""
Macro/sector beta alignment gate — Lever 1 of `absorption_breakout`'s round-3
investigation (`backtests/reports/absorption_breakout_investigation_report.md`'s
round-3 section). Rationale: round 1's holdout edge was concentrated in a
handful of large winning trades (Monte Carlo p5 fragility) — this gate tests
whether requiring the broad market/sector tape to already be moving WITH a
breakout's direction (rather than against it) filters out the "fighting the
tape" attempts that plausibly contribute the many small losers, while
preserving the genuine trend-aligned winners. It does NOT assume this works;
see the investigation report for the actual diagnosed effect.

Design choice — a SINGLE composite feature, not "all three must individually
agree": QQQ/SPY/XLK momentum is combined via a simple equal-weighted average
of each symbol's own return at every timestamp, rather than requiring each
of the three ETFs to individually satisfy the sign condition. This is a
deliberate, disclosed interpretation of the brief's "QQQ/SPY/XLK ... momentum
is non-negative" as a statement about the broad market/tech-sector tape
taken together (a composite), not three independent unanimous votes — three
highly-correlated ETFs (QQQ, SPY, and XLK, itself QQQ's own top holdings'
supersector) requiring unanimous agreement would mostly reduce to whichever
one is noisiest at the margin, adding sensitivity to an arbitrary tie-break
rule without a clear economic reason to prefer it over just averaging. No
extra tunable threshold is introduced either way (the gate is "non-negative",
literally zero vs. non-zero, not a fitted cutoff) — this keeps the lever
completely free of new free parameters under
`python/backtest/param_guard.py`'s counting, exactly like
`orb_vwap_regime.py`'s own hardcoded (not gridded) regime threshold.

No-lookahead: `mom_1m`/`mom_5m` at row t use ONLY close[t-1..t] / close[t-5..t]
— the same bar whose close the gated signal itself fires on, not looked
ahead of. `macro_gate_ok` reads the last closed composite row at or
before the signal's OWN `signal_time` (as-of, capped at one 5-minute
bin). No `.shift()` is needed (unlike
`trend_efficiency_gate.shifted_entry_gate`'s day-ahead label). Exact
index membership is not required: live `signal_time` can carry
sub-second residue while index bars are start-labeled on the minute.

Data-quality note: `compute_macro_momentum` is built directly on
`data/history_1m/<QQQ|SPY|XLK>/*.parquet`, sourced from Futu OpenD
(`python/data/futu_price_source.py`) rather than IBKR — see that module's own
docstring for the empirically-verified bar-timestamp convention and the
known last-bar-of-day caveat, both of which apply here unchanged.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_INDEX_SYMBOLS: tuple[str, ...] = ("QQQ", "SPY", "XLK")
# Structural as-of lag: use the last closed index minute at or before the
# signal, but never a bar older than one 5-minute decision bin. Not a
# Chan free parameter. Last night's rejects were exact-timestamp misses
# (signal 12:00:00.196 vs start-labeled 12:00, or refresh still on 11:58).
_MAX_ASOF_LAG = pd.Timedelta(minutes=5)


def _momentum(close: pd.Series, bars: int) -> pd.Series:
    """Causal `bars`-bar simple return WITHIN each trading session: close[t]
    / close[t-bars] - 1, using ONLY close[t-bars..t]. NaN for the first
    `bars` rows of `close` (insufficient trailing history) — no lookahead.

    SESSION-SCOPED, not a raw row-index shift: `close` here is normally the
    FULL multi-day cached panel (see `compute_macro_momentum`'s own
    docstring — callers pass the whole window, not a pre-sliced single
    session, unlike every other cumulative helper in
    `python/microstructure/context.py`, whose callers always pre-slice to
    one session before calling in). A plain `close.shift(bars)` would
    therefore, at the first `bars` bars of every session AFTER the first,
    silently reach across the overnight gap into the PREVIOUS session's
    closing bars — not lookahead (it is still strictly-historical data),
    but a mislabeling bug: the first few bars of every session would report
    an "N-minute momentum" that is actually an overnight-gap return, which
    could differ by an order of magnitude from a genuine same-session
    N-minute move. Grouping by calendar day resets the shift at each
    session boundary instead, matching every other bar in this repo's own
    day-scoping discipline. (Found during the 2026-08-15 audit,
    `backtests/reports/backtest_engine_audit_round2.md` — verified to have
    had ZERO effect on `absorption_breakout`'s round-3 numbers, since that
    signal's own `min_bars` floor (22) exceeds this gate's longest lookback
    (5), so no bar close enough to a session boundary to be affected by the
    old bug could ever have gated a real signal — fixed anyway since this
    module is meant to be reusable for future, possibly open-firing
    signals.)"""
    if bars <= 0:
        raise ValueError("bars must be positive")
    session = close.index.normalize()
    return close / close.groupby(session).shift(bars) - 1.0


def compute_macro_momentum(
    index_bars: dict[str, pd.DataFrame],
    momentum_bars: tuple[int, int] = (1, 5),
) -> pd.DataFrame:
    """`index_bars`: {"QQQ": df, "SPY": df, "XLK": df, ...}, each a 1-minute
    OHLCV frame indexed by `ts` (start-labeled — see
    python/data/futu_price_source.py). Returns a DataFrame indexed by the
    UNION of every symbol's timestamps with columns `mom_1m`/`mom_5m`: the
    equal-weighted AVERAGE `bars`-bar return across whichever of
    `index_bars` have a row at that exact timestamp (this repo's per-symbol
    caches are independently-fetched RTH-only 1-minute bars, so a timestamp
    missing for one symbol on a given minute is a genuine data gap between
    two independently-sourced series, not resampled/forward-filled away —
    see module docstring's data-quality note). A timestamp present for only
    ONE of the symbols still produces a (single-symbol) average rather than
    NaN — deliberately lenient, since occasional single-bar gaps between
    independently-fetched series are expected and should not blank out an
    otherwise-computable composite.

    Momentum is computed PER TRADING SESSION (see `_momentum`) — the first
    `bars` bars of every session get NaN rather than reaching across the
    overnight gap into the previous session's closes, even though callers
    here typically pass the full multi-day cached panel, not a single
    pre-sliced session.

    Raises ValueError if `index_bars` is empty."""
    if not index_bars:
        raise ValueError("compute_macro_momentum: index_bars must not be empty")
    b1, b5 = momentum_bars
    frames = []
    for df in index_bars.values():
        close = df["close"]
        frames.append(pd.DataFrame({"mom_1m": _momentum(close, b1), "mom_5m": _momentum(close, b5)}))
    stacked = pd.concat(frames)
    combined = stacked.groupby(level=0).mean()
    combined.index.name = "ts"
    return combined.sort_index()


def _normalize_ts(ts, index: pd.DatetimeIndex) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    idx_tz = getattr(index, "tz", None)
    if idx_tz is not None:
        if ts.tzinfo is None:
            ts = ts.tz_localize(idx_tz)
        else:
            ts = ts.tz_convert(idx_tz)
    elif ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def _asof_row(macro_momentum: pd.DataFrame, at_time) -> pd.Series | None:
    """Last closed composite row at or before `at_time`, or None.

    Exact index membership is not required — live signal_time often has
    sub-second residue (2026-08-18 GOOGL 12:00:00.196) while Futu 1m
    bars are start-labeled on the minute. A row older than
    `_MAX_ASOF_LAG` is treated as missing (fail-closed), so yesterday's
    last print cannot authorize today's trade."""
    if macro_momentum is None or macro_momentum.empty:
        return None
    idx = macro_momentum.index
    ts = _normalize_ts(at_time, idx)
    loc = idx.searchsorted(ts, side="right") - 1
    if loc < 0:
        return None
    bar_ts = idx[loc]
    if (ts - bar_ts) > _MAX_ASOF_LAG:
        return None
    row = macro_momentum.iloc[loc]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def macro_gate_ok(macro_momentum: pd.DataFrame, direction: str, at_time) -> bool:
    """True iff the composite 1-minute AND 5-minute macro momentum are both
    non-negative (for `direction == "long"`) or both non-positive (for
    `direction == "short"`) as of the last closed index minute at or
    before `at_time` (the gated signal's own `signal_time`). No `.shift()`
    — see module docstring. Fails CLOSED when there is no row within
    `_MAX_ASOF_LAG`, or either momentum value is NaN."""
    if direction not in ("long", "short"):
        raise ValueError(f"macro_gate_ok: unknown direction {direction!r}")
    row = _asof_row(macro_momentum, at_time)
    if row is None:
        return False
    mom_1m, mom_5m = row["mom_1m"], row["mom_5m"]
    if pd.isna(mom_1m) or pd.isna(mom_5m):
        return False
    if direction == "long":
        return bool(mom_1m >= 0 and mom_5m >= 0)
    return bool(mom_1m <= 0 and mom_5m <= 0)


def load_index_1m_bars(
    symbols: tuple[str, ...] = DEFAULT_INDEX_SYMBOLS,
    lookback_days: int = 5,
    cache_dir=None,
    fetch_live: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load QQQ/SPY/XLK 1-minute bars for the live macro gate.

    Cache-first (`data/history_1m/<SYMBOL>/`). Optionally tries a Futu
    OpenD same-day fetch when `fetch_live=True` — a failure there is
    swallowed (logged) rather than raised, because the gate's contract is
    fail-CLOSED: missing today's bar at the signal timestamp blocks the
    trade; it must never silently drop the gate and trade ungated.

    Returns a (possibly empty) `{symbol: ohlcv}` dict. An empty dict means
    "no index bars available" and MUST be treated as gate-closed by the
    caller.
    """
    from python.data.intraday_cache import CACHE_DIR, get_cached_intraday_panel

    cache_dir = cache_dir or CACHE_DIR
    end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=max(int(lookback_days), 1) + 2)
    out: dict[str, pd.DataFrame] = {}
    try:
        panel = get_cached_intraday_panel(list(symbols), start, end, cache_dir=cache_dir)
    except Exception:
        panel = None
    if panel is not None and not panel.empty:
        for symbol in symbols:
            try:
                df = panel.xs(symbol, level="code").sort_index()
            except (KeyError, ValueError):
                continue
            if not df.empty:
                out[symbol] = _ohlcv_cols(df)

    if fetch_live:
        _merge_live_futu_index_bars(out, symbols)

    return out


def _ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[cols]


def _merge_live_futu_index_bars(out: dict[str, pd.DataFrame], symbols: tuple[str, ...]) -> None:
    """Best-effort same-day Futu 1m append. Failures are logged, never raised."""
    import logging

    log = logging.getLogger(__name__)
    try:
        from python.data.futu_price_source import (
            FutuHistoricalUnavailable,
            fetch_history_kline_range,
            load_connection_settings,
            open_futu_quote_context,
        )
    except Exception:
        log.warning("macro_beta_gate: futu_price_source unavailable — live index fetch skipped")
        return

    today = pd.Timestamp.now().normalize()
    start = today.strftime("%Y-%m-%d")
    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        ctx = open_futu_quote_context()
    except FutuHistoricalUnavailable as exc:
        log.warning("macro_beta_gate: OpenD unreachable for live index bars (%s) — gate stays fail-closed on missing timestamps", exc)
        return
    except Exception as exc:
        log.warning("macro_beta_gate: live index fetch setup failed (%s)", exc)
        return

    settings = load_connection_settings()
    try:
        for symbol in symbols:
            try:
                fresh = fetch_history_kline_range(
                    ctx, symbol, start, end,
                    market_prefix=settings.get("market_prefix", "US"),
                )
            except Exception:
                log.exception("macro_beta_gate: live Futu 1m fetch failed for %s", symbol)
                continue
            if fresh is None or fresh.empty:
                continue
            fresh = _ohlcv_cols(fresh).sort_index()
            existing = out.get(symbol)
            if existing is None or existing.empty:
                out[symbol] = fresh
            else:
                combined = pd.concat([existing, fresh])
                out[symbol] = combined[~combined.index.duplicated(keep="last")].sort_index()
    finally:
        try:
            ctx.close()
        except Exception:
            pass


class LiveMacroGate:
    """Fail-closed live wrapper around `macro_gate_ok`.

    If index bars cannot be loaded, or no closed index minute sits within
    `_MAX_ASOF_LAG` of the signal, or either momentum value is NaN,
    `ok()` returns False. The gate is never silently dropped.

    `live_fetch=True` (EngineRuntime only) re-pulls QQQ/SPY/XLK from
    OpenD at evaluation time, once per decision minute, so the as-of
    row is tonight's tape rather than the bars from Start. Tests leave
    this False so an empty injected gate stays closed.
    """

    def __init__(
        self,
        index_bars: dict[str, pd.DataFrame] | None = None,
        live_fetch: bool = False,
    ) -> None:
        self._momentum: pd.DataFrame | None = None
        self._live_fetch = bool(live_fetch)
        self._last_fetch_minute: pd.Timestamp | None = None
        if index_bars:
            self.set_index_bars(index_bars)

    def set_index_bars(self, index_bars: dict[str, pd.DataFrame]) -> None:
        if not index_bars:
            self._momentum = None
            return
        try:
            self._momentum = compute_macro_momentum(index_bars)
        except ValueError:
            self._momentum = None

    def refresh_from_cache(self, fetch_live: bool = False) -> None:
        self.set_index_bars(load_index_1m_bars(fetch_live=fetch_live))

    def refresh_for(self, at_time) -> None:
        """Pull live index 1m once per decision minute, then leave as-of
        to `ok()`. Failures keep the last seed — never ungate."""
        if not self._live_fetch:
            return
        minute = pd.Timestamp(at_time).floor("min")
        if minute.tzinfo is not None:
            minute = minute.tz_localize(None)
        if self._last_fetch_minute is not None and self._last_fetch_minute == minute:
            return
        try:
            self.refresh_from_cache(fetch_live=True)
            self._last_fetch_minute = minute
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "LiveMacroGate.refresh_for failed — gate stays on last seed"
            )

    def ok(self, direction: str, at_time) -> bool:
        if self._momentum is None or self._momentum.empty:
            return False
        return macro_gate_ok(self._momentum, direction, at_time)
