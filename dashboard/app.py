"""
FastAPI backend for the React/Vite dashboard.

Endpoints:
  GET  /api/state              — full DashboardState snapshot (polled by the UI)
  POST /api/engine/start       — start the paper-trading engine (observe mode only)
  POST /api/engine/stop        — stop it (also disarms auto trading)
  POST /api/engine/auto/start  — arm real order submission for this session ("Start Auto Trading")
  POST /api/engine/auto/stop   — disarm it, back to observe mode
  POST /api/positions/flatten_all — emergency: close every open position now
  GET  /api/positions          — real per-symbol broker positions
  POST /api/positions/{code}/flatten — close a single symbol's position now
  POST /api/backtest/run       — run a quick demo cross-sectional backtest and
                                  store the summary in DashboardState
  GET  /api/backtest/latest    — latest stored backtest summary
  GET  /api/chart/{symbol}     — on-demand price chart for any ticker.
                                  ?interval=1d (default, daily bars, any
                                  history) or 1m/5m/15m (intraday bars,
                                  CACHE ONLY — data/history_1m/, built by
                                  scripts/backfill_intraday.py; no live IB
                                  fetch from this endpoint. 5m/15m are
                                  resampled on the fly from the 1-minute
                                  cache, see intraday_cache.resample_ohlcv).
  GET  /api/chart/{symbol}/context — microstructure context (VWAP + bands,
                                  liquidity levels, volume profile, opening
                                  range) for one cached 1-minute session —
                                  python/microstructure/context.py, for
                                  overlaying on the 1m chart.
  GET  /api/regime/{symbol}    — report-only Markov regime diagnostic
                                  (Bull/Bear/Sideways transition matrix +
                                  stationary distribution) on daily bars —
                                  python/analytics/regime.py. NOT wired to
                                  any strategy/order; see that module's
                                  docstring for the honesty contract.

Serves the built frontend (frontend/dist) as static files when present, so
`uvicorn dashboard.app:app` alone is enough for a production-style demo;
during development, run the Vite dev server separately (`npm run dev` in
frontend/) and it proxies /api to this backend (see frontend/vite.config.ts).
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .engine_bridge import EngineRuntime
from .state import state

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="US Equity Quant Trading Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_live_universe_symbols() -> list[str] | None:
    """configs/universe.yaml's fixed_universe.symbols (the SAME 20-symbol,
    top-by-liquidity universe scripts/run_intraday_backtest.py and the
    self-improve WFO loop use) — wired in here so the live dashboard
    runtime actually streams/trades the universe operators believe it
    does, instead of silently falling back to EngineRuntime's 5-symbol
    hardcoded placeholder (see EngineRuntime.__init__ and
    DashboardState's module docstring, which already flags this exact
    mismatch). Returns None (-> EngineRuntime's own hardcoded fallback)
    rather than crashing FastAPI startup if the config is missing or
    malformed, matching dashboard/engine_bridge.py's own
    _load_broker_config fail-safe style."""
    from python.data.fixed_universe import load_universe_config

    try:
        return list(load_universe_config()["symbols"])
    except Exception:
        log.exception(
            "dashboard.app: failed to load configs/universe.yaml — EngineRuntime will fall back "
            "to its hardcoded 5-symbol placeholder universe"
        )
        return None


runtime = EngineRuntime(state, symbols=_load_live_universe_symbols())


@app.get("/api/state")
async def get_state() -> dict:
    return state.snapshot()


@app.post("/api/engine/start")
async def start_engine() -> dict:
    await runtime.start()
    return {"ok": True, "running": state.running}


@app.post("/api/engine/stop")
async def stop_engine() -> dict:
    await runtime.stop()
    return {"ok": True, "running": state.running}


@app.post("/api/engine/auto/start")
async def start_auto_trading() -> dict:
    """Arms real order submission for the running session ("Start Auto
    Trading"). See EngineRuntime.enable_auto_trading — this flips the
    gateway's two-key AND gate in memory only, never touches
    configs/strategy.yaml, and always reverts to observe on the next
    Stop."""
    if not state.running:
        raise HTTPException(status_code=400, detail="Start the engine (paper) before arming auto trading")
    strategies = await runtime.enable_auto_trading()
    return {"ok": True, "mode": state.mode, "armed_strategies": sorted(strategies)}


@app.post("/api/engine/auto/stop")
async def stop_auto_trading() -> dict:
    await runtime.disable_auto_trading()
    return {"ok": True, "mode": state.mode}


@app.post("/api/positions/flatten_all")
async def flatten_all_positions() -> dict:
    """"Exit All Positions" emergency button — closes every open position
    at the current broker immediately, regardless of engine/auto state."""
    results = await runtime.emergency_flatten_all()
    return {"ok": True, "closed": results}


@app.get("/api/positions")
async def list_positions() -> dict:
    """Real per-symbol positions at the current broker (SimBroker or
    IbkrBroker) — the system may hold several symbols at once across both
    strategies, so this is independent of Strategy A/B's own open_pairs /
    portfolio_weights bookkeeping (which reflect strategy INTENT, not the
    broker's actual fills). Backs the dashboard's Positions panel, which
    has a per-symbol Exit button next to each row."""
    loop = asyncio.get_event_loop()
    positions = await loop.run_in_executor(None, runtime.broker.get_positions)
    return {
        "positions": [
            {"code": code, "qty": qty, "side": "long" if qty > 0 else "short"}
            for code, qty in positions.items() if qty != 0
        ],
    }


@app.post("/api/positions/{code}/flatten")
async def flatten_one_position(code: str) -> dict:
    """Exit button next to a single symbol's row in the Positions panel."""
    result = await runtime.flatten_position(code.upper().strip())
    if result is None:
        raise HTTPException(status_code=404, detail=f"No open position for {code}")
    return {"ok": True, "closed": result}


@app.post("/api/backtest/run")
async def run_backtest_demo() -> dict:
    """Runs a small synthetic-data demo of the cross-sectional strategy so
    the dashboard has something concrete to show without requiring a live
    yfinance connection. For a real research backtest, use
    scripts/run_backtest.py directly — this endpoint is a UI convenience,
    not the system of record for validation results."""
    import numpy as np
    import pandas as pd

    from python.backtest.vector_engine import run_vector_backtest
    from python.core.strategies.xsection_mean_reversion import CrossSectionalMeanReversionStrategy

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=120)
    codes = [f"DEMO{i:02d}" for i in range(30)]
    rows = []
    for code in codes:
        price = 100.0 + rng.uniform(-20, 20)
        for d in dates:
            ret = rng.normal(0, 0.015)
            open_px = price
            close_px = price * (1 + ret)
            rows.append({"date": d, "code": code, "open": open_px, "close": close_px, "adv_20d_dollars": 20_000_000.0})
            price = close_px
    panel = pd.DataFrame(rows).set_index(["date", "code"]).sort_index()

    strategy = CrossSectionalMeanReversionStrategy(min_universe_size=10)
    universe_by_day = {d: codes for d in dates[20:]}
    result = run_vector_backtest(strategy, panel, universe_by_day)

    # Normalize to start at 100 so the equity-curve chart reads as a % growth
    # index rather than raw dollar notional (the absolute capital amount is
    # an arbitrary demo constant, the shape of the curve is what matters).
    equity = result.equity_curve
    equity_curve = (equity / equity.iloc[0] * 100.0).round(4).tolist() if len(equity) else []

    summary = {
        "label": "SYNTHETIC DEMO DATA — not a real market backtest",
        "sharpe_annualized": result.sharpe_annualized,
        "max_drawdown": result.max_drawdown,
        "cagr": result.cagr,
        "n_days": int(len(result.daily_returns)),
        "equity_curve": equity_curve,
    }
    state.set_backtest_summary(summary)
    return summary


@app.get("/api/backtest/latest")
async def latest_backtest() -> dict:
    return state.latest_backtest_summary or {}


@app.post("/api/pairs/demo_scan")
async def run_pairs_demo_scan() -> dict:
    """Runs the real CADF cointegration test (python/stat/cointegration.py)
    against synthetic-but-genuinely-mean-reverting price pairs, so the
    dashboard has concrete spread z-score history to chart without requiring
    a live yfinance connection. For real pair discovery use
    python/stat/pair_scanner.py against real price history — this endpoint
    is a UI convenience only, mirroring /api/backtest/run's synthetic-demo
    pattern."""
    import numpy as np
    import pandas as pd

    from python.stat.cointegration import current_spread, spread_z_score, test_pair

    rng = np.random.default_rng(1)
    n = 180
    candidates = []
    for name_a, name_b, ar_coef in [("PAIRA", "PAIRB", 0.8), ("PAIRC", "PAIRD", 0.85)]:
        common_walk = np.cumsum(rng.normal(0, 0.01, n))
        idio_noise_a = np.cumsum(rng.normal(0, 0.003, n))

        # Stationary AR(1) spread — guarantees a genuine, finite O-U half-life
        # rather than hoping a random walk happens to look mean-reverting.
        stat_spread = np.zeros(n)
        for i in range(1, n):
            stat_spread[i] = ar_coef * stat_spread[i - 1] + rng.normal(0, 0.02)

        log_price_a = 4.6 + common_walk + idio_noise_a * 0.3
        log_price_b = 4.6 + common_walk - stat_spread
        prices_a = pd.Series(np.exp(log_price_a))
        prices_b = pd.Series(np.exp(log_price_b))

        try:
            result = test_pair(name_a, name_b, prices_a, prices_b)
        except Exception:
            log.exception("demo_scan: test_pair failed for (%s, %s)", name_a, name_b)
            continue
        if not result.is_tradeable:
            continue

        zscore_history = [
            spread_z_score(
                current_spread(float(prices_a.iloc[i]), float(prices_b.iloc[i]), result.hedge_ratio),
                result.spread_mean,
                result.spread_std,
            )
            for i in range(n)
        ]

        candidates.append({
            "code_a": result.code_a,
            "code_b": result.code_b,
            "cadf_tstat": result.cadf_tstat,
            "half_life_days": result.half_life_days,
            "zscore_history": zscore_history,
        })

    state.pair_candidates = candidates
    return {"pair_candidates": candidates}


def _validate_symbol(symbol: str) -> str:
    symbol = symbol.upper().strip()
    if not symbol.isalnum() or len(symbol) > 6:
        raise HTTPException(status_code=400, detail=f"'{symbol}' does not look like a valid US equity ticker")
    return symbol


@app.get("/api/chart/{symbol}")
async def get_symbol_chart(symbol: str, days: int = 180, interval: str = "1d") -> dict:
    """On-demand price chart for an arbitrary US equity ticker — a UI
    lookup convenience, NOT part of the strategy pipeline.

    interval="1d" (default): daily bars via python/data/price_cache.py, so
    charts get the exact same IBKR-then-yfinance source policy, on-disk
    cache, and IB pacing as the backtest/research path. First request for
    a symbol/range can take a few seconds (real fetch + on-disk cache
    write); subsequent requests for an already-covered range are instant.

    interval in ("1m", "5m", "15m"): intraday bars via python/data/intraday_cache.py,
    READ FROM THE LOCAL 1-MINUTE CACHE ONLY (data/history_1m/) — this endpoint
    never triggers a live IB fetch, matching intraday_cache.get_cached_intraday_panel's
    contract. "5m"/"15m" are resampled on the fly from the same 1-minute cache
    (intraday_cache.resample_ohlcv) — there is no separate 5-/15-minute cache
    or extra IB request per bar size. Run scripts/backfill_intraday.py first
    to populate the cache; a 404 here means "no cached 1-minute bars in
    range", not "no data exists"."""
    import pandas as pd

    symbol = _validate_symbol(symbol)
    loop = asyncio.get_event_loop()

    if interval in ("1m", "5m", "15m"):
        from python.data.intraday_cache import get_cached_intraday_panel, latest_cached_bar_time, resample_ohlcv

        days = max(1, min(days, 60))
        end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=days)

        def _load(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
            panel = get_cached_intraday_panel([symbol], start, end)
            return panel.xs(symbol, level="code").sort_index()

        try:
            df = await loop.run_in_executor(None, lambda: _load(start, end))
        except Exception as exc:
            # A naive "now - N days" window can land entirely on empty
            # calendar days (weekend, holiday, before today's session has
            # traded, or simply before backfill has run yet) even though
            # good recent data is cached. Retry anchored on the most recent
            # bar actually on disk before giving up — this is what "show me
            # the last N days" should mean for a short intraday window.
            anchor = await loop.run_in_executor(None, lambda: latest_cached_bar_time(symbol))
            if anchor is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No cached 1-minute bars for {symbol} in range — run "
                           f"scripts/backfill_intraday.py first ({exc})",
                ) from exc
            end = anchor.normalize() + pd.Timedelta(days=1)
            start = end - pd.Timedelta(days=days)
            try:
                df = await loop.run_in_executor(None, lambda: _load(start, end))
            except Exception as exc2:
                raise HTTPException(
                    status_code=404,
                    detail=f"No cached 1-minute bars for {symbol} in range — run "
                           f"scripts/backfill_intraday.py first ({exc2})",
                ) from exc2

        if interval != "1m":
            freq = {"5m": "5min", "15m": "15min"}[interval]
            df = await loop.run_in_executor(None, lambda: resample_ohlcv(df, freq))

        return {
            "symbol": symbol,
            "interval": interval,
            "dates": [t.isoformat() for t in df.index],
            "open": df["open"].round(4).tolist(),
            "high": df["high"].round(4).tolist(),
            "low": df["low"].round(4).tolist(),
            "close": df["close"].round(4).tolist(),
            "volume": [int(v) for v in df["volume"]],
            "source": "ibkr_cache_1m" if interval == "1m" else f"ibkr_cache_1m_resampled_{interval}",
            "quality_flagged": False,
        }

    if interval != "1d":
        raise HTTPException(status_code=400, detail=f"interval must be one of '1d', '1m', '5m', '15m', got {interval!r}")

    from python.data.price_cache import get_cached_price_panel

    days = max(1, min(days, 3650))
    end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=days)

    try:
        panel, quality_flags, meta = await loop.run_in_executor(
            None, lambda: get_cached_price_panel([symbol], start, end),
        )
        df = panel.xs(symbol, level=1)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"No price data available for {symbol}: {exc}") from exc

    return {
        "symbol": symbol,
        "interval": "1d",
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "open": df["open"].round(4).tolist(),
        "high": df["high"].round(4).tolist(),
        "low": df["low"].round(4).tolist(),
        "close": df["close"].round(4).tolist(),
        "volume": [int(v) for v in df["volume"]],
        "source": meta.get("fetched_source") or next(iter(meta.get("sources", {})), "cache"),
        "quality_flagged": symbol in quality_flags,
    }


_CONTEXT_SIGNALS = ["sweep_reclaim", "fvg_retest", "orb_vwap", "l2_absorption"]


@app.get("/api/chart/{symbol}/context")
async def get_symbol_intraday_context(symbol: str, date: str | None = None, or_minutes: int = 15) -> dict:
    """Microstructure context for ONE cached 1-minute session
    (python/microstructure/context.py's compute_context) — VWAP + 1/2 sigma
    bands, liquidity levels (YDH/YDL, PMH/PML, equal highs/lows, round
    numbers), an approximate volume profile, the opening range, AND every
    sweep_reclaim/fvg_retest/orb_vwap pattern detected that session
    (python/backtest/intraday_engine.scan_signals_for_session, using each
    signal's CURRENT configs/strategy.yaml parameters) for chart markers.
    Purely a chart-overlay convenience for the dashboard; report-only, same
    as the rest of the microstructure diagnostic layer — computes nothing
    that feeds back into any live signal or order, and (unlike a real
    backtest) does not simulate fills/exits or skip bars while "in a
    position" — see scan_signals_for_session's docstring.

    `date` defaults to the most recent cached session for `symbol`. Reads
    from data/history_1m/ ONLY (no live IB fetch) — 404 if nothing is
    cached, with a hint to run scripts/backfill_intraday.py."""
    import yaml
    import pandas as pd

    from python.backtest.intraday_engine import SIGNAL_PARAM_KEYS, scan_signals_for_session
    from python.data.intraday_cache import get_cached_intraday_panel
    from python.microstructure.context import compute_context

    symbol = _validate_symbol(symbol)
    loop = asyncio.get_event_loop()

    # Look back far enough to comfortably find both the target session and
    # its prior trading day even across weekends/holidays.
    end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=20)
    try:
        panel = await loop.run_in_executor(None, lambda: get_cached_intraday_panel([symbol], start, end))
        df = panel.xs(symbol, level="code").sort_index()
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"No cached 1-minute bars for {symbol} — run scripts/backfill_intraday.py first ({exc})",
        ) from exc

    available_dates = sorted(set(df.index.normalize()))
    if not available_dates:
        raise HTTPException(status_code=404, detail=f"No cached 1-minute sessions for {symbol}")

    if date is not None:
        target_date = pd.Timestamp(date).normalize()
        if target_date not in available_dates:
            raise HTTPException(
                status_code=404,
                detail=f"No cached session for {symbol} on {date} — available dates include "
                       f"{[d.strftime('%Y-%m-%d') for d in available_dates[-5:]]}",
            )
    else:
        target_date = available_dates[-1]

    prior_dates = [d for d in available_dates if d < target_date]
    bars_today = df.loc[df.index.normalize() == target_date]
    prior_day_bars = df.loc[df.index.normalize() == prior_dates[-1]] if prior_dates else None

    ctx_state = compute_context(bars_today, prior_day_bars, or_minutes=max(1, min(or_minutes, 120)))
    vwap_df = ctx_state.vwap

    with open("configs/strategy.yaml", encoding="utf-8") as f:
        strategy_cfg = yaml.safe_load(f)
    detected_signals = []
    for sig_name in _CONTEXT_SIGNALS:
        base_cfg = strategy_cfg.get(sig_name, {})
        params = {k: base_cfg[k] for k in SIGNAL_PARAM_KEYS[sig_name] if k in base_cfg}
        try:
            hits = await loop.run_in_executor(
                None, lambda sn=sig_name, p=params: scan_signals_for_session(sn, bars_today, p, prior_day_bars=prior_day_bars),
            )
        except Exception:
            log.exception("scan_signals_for_session failed for %s/%s — omitting from context response", symbol, sig_name)
            continue
        detected_signals.extend({
            "strategy": sig_name,
            "time": s.signal_time.isoformat(),
            "direction": s.direction,
            "entry_price": round(s.entry_price, 4),
        } for s in hits)

    return {
        "symbol": symbol,
        "date": target_date.strftime("%Y-%m-%d"),
        "vwap": {
            "dates": [t.isoformat() for t in vwap_df.index],
            "vwap": vwap_df["vwap"].round(4).tolist(),
            "upper_1": vwap_df["upper_1"].round(4).tolist(),
            "lower_1": vwap_df["lower_1"].round(4).tolist(),
            "upper_2": vwap_df["upper_2"].round(4).tolist(),
            "lower_2": vwap_df["lower_2"].round(4).tolist(),
        },
        "liquidity": {
            "ydh": ctx_state.liquidity.ydh,
            "ydl": ctx_state.liquidity.ydl,
            "pmh": ctx_state.liquidity.pmh,
            "pml": ctx_state.liquidity.pml,
            "eq_highs": ctx_state.liquidity.eq_highs,
            "eq_lows": ctx_state.liquidity.eq_lows,
            "round_levels": ctx_state.liquidity.round_levels,
        },
        "volume_profile": {
            "poc": ctx_state.volume_profile.poc,
            "vah": ctx_state.volume_profile.vah,
            "val": ctx_state.volume_profile.val,
            "bin_edges": ctx_state.volume_profile.bin_edges,
            "bin_volume": ctx_state.volume_profile.bin_volume,
        },
        "opening_range": {
            "high": ctx_state.opening_range.high,
            "low": ctx_state.opening_range.low,
            "start": ctx_state.opening_range.start.isoformat() if ctx_state.opening_range.start is not None else None,
            "end": ctx_state.opening_range.end.isoformat() if ctx_state.opening_range.end is not None else None,
        },
        "signals": detected_signals,
        "available_dates": [d.strftime("%Y-%m-%d") for d in available_dates],
    }


@app.get("/api/regime/{symbol}")
async def get_symbol_regime(symbol: str, years: int = 5, window: int = 20, threshold: float = 0.02) -> dict:
    """Report-only Markov regime diagnostic (python/analytics/regime.py) —
    NOT wired to any strategy or order. Daily bars via
    python/data/price_cache.py (same IBKR-first/yfinance-fallback/on-disk
    cache as /api/chart), so this reads real historical prices, never
    yfinance directly. See regime.py's module docstring for the full
    honesty contract, including why the "naive_backtest" figures in the
    response are illustrative only and not a validated strategy result."""
    import pandas as pd

    from python.analytics.regime import compute_regime_report
    from python.data.price_cache import get_cached_price_panel

    symbol = _validate_symbol(symbol)
    years = max(1, min(years, 20))
    window = max(2, min(window, 250))
    threshold = max(0.001, min(threshold, 0.5))

    end = pd.Timestamp.now().normalize()
    start = end - pd.DateOffset(years=years)
    loop = asyncio.get_event_loop()
    try:
        panel, _quality_flags, _meta = await loop.run_in_executor(
            None, lambda: get_cached_price_panel([symbol], start, end),
        )
        close = panel.xs(symbol, level=1)["close"]
        report = await loop.run_in_executor(
            None, lambda: compute_regime_report(close, symbol, window=window, threshold=threshold),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"No price data available for {symbol}: {exc}") from exc

    return report.to_dict()


_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
else:
    @app.get("/")
    async def _no_frontend_build() -> dict:
        return {
            "message": "Frontend not built yet. Run `npm install && npm run build` in frontend/, "
                       "or run `npm run dev` there and open http://localhost:5173",
        }
