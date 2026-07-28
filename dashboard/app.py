"""
FastAPI backend for the React/Vite dashboard.

Endpoints:
  GET  /api/state              — full DashboardState snapshot (polled by the UI)
  POST /api/engine/start       — start the paper-trading engine (observe mode only)
  POST /api/engine/stop        — stop it
  POST /api/backtest/run       — run a quick demo cross-sectional backtest and
                                  store the summary in DashboardState
  GET  /api/backtest/latest    — latest stored backtest summary

Serves the built frontend (frontend/dist) as static files when present, so
`uvicorn dashboard.app:app` alone is enough for a production-style demo;
during development, run the Vite dev server separately (`npm run dev` in
frontend/) and it proxies /api to this backend (see frontend/vite.config.ts).
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
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

runtime = EngineRuntime(state)


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

    summary = {
        "label": "SYNTHETIC DEMO DATA — not a real market backtest",
        "sharpe_annualized": result.sharpe_annualized,
        "max_drawdown": result.max_drawdown,
        "cagr": result.cagr,
        "n_days": int(len(result.daily_returns)),
    }
    state.set_backtest_summary(summary)
    return summary


@app.get("/api/backtest/latest")
async def latest_backtest() -> dict:
    return state.latest_backtest_summary or {}


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
