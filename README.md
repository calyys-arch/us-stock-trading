# US Equity Quant Trading System

A US-equities algorithmic day-trading system built on Ernest P. Chan's
*Quantitative Trading: How to Build Your Own Algorithmic Trading Business*
(Wiley, 2009). Architecture is derived from a proven forex event-driven
system (`../forex-trading`), rebuilt for US equities with Chan's statistical
methodology as the strategy layer.

> **System scope**: US-listed equities only. No forex, no options, no crypto.
> Two strategies: cointegrated pairs trading (can hold overnight) and
> intraday cross-sectional mean reversion (flat by 15:55 ET).

## Why this design (see `docs/` for full research trail)

- `docs/lessons_from_forex_trading.md` — concrete bugs found in the
  predecessor system (dead strategies, dead config keys, timestamp bugs) and
  the specific test/guard added here to prevent each one.
- Strategy design follows Chan's book directly rather than a generic ML
  classifier: max 5 free parameters per strategy, mandatory train/test split,
  look-ahead-bias truncation test, Kelly-based capital allocation, and
  regime-appropriate exits (no stop-loss on mean-reversion signals).

## Architecture

```
Feed (IbkrFeed: Stock contracts) ─┐
yfinance (historical/backtest)  ──┼─→ DataEngine (RTH gate, corp-action adj, bad-tick filter)
                                   │        │
                     Universe Builder        ▼ snapshot
                  (point-in-time S&P 500) MessageBus
                                   │        │
                            PairScanner     ▼
                            (CADF test) SignalEngine ── PairsTradingStrategy (per-pair evaluate)
                                              └──────── CrossSectionalMeanReversionStrategy (PortfolioStrategy, daily)
                                                                │ raw_signal / target weights
                                                                ▼
                                                          RiskEngine (Kelly-sized, 1% ADV, sector caps, PDT)
                                                                │ qualified_signal
                                                                ▼
                                                        ExecutionGateway (RTH guard, EOD flatten)
                                                                │
                                                          IbkrBroker (Stock orders)
                                                                │ execution_report
                                                                ▼
                                                   PairPositionManager / PositionManager
                                                                │
                                                       DataRecorder (point-in-time DB) ── DashboardState
```

## Directory layout

```
python/
├── core/
│   ├── bus.py                 # MessageBus pub/sub (ported verbatim from forex-trading)
│   ├── types.py               # Tick/Candle/MarketSnapshot/RawSignal/QualifiedSignal/SpreadOrder
│   ├── calendar.py            # NYSE trading calendar (exchange_calendars wrapper)
│   ├── timeutil.py            # single source of truth for timestamp parsing/units
│   ├── data_engine.py         # tick → snapshot, RTH session gate, bad-tick filter
│   ├── kelly.py                # F*=C⁻¹M, half-Kelly, drawdown-capped leverage
│   ├── risk_engine.py         # constraint checks: ADV cap, sector caps, PDT, short-locate
│   ├── execution_gateway.py   # RTH guard, EOD flatten, us_equity guard
│   ├── fees_equity.py         # commission + SEC Section 31 + FINRA TAF + borrow cost
│   ├── rate_limiter.py        # token-bucket limiter for external APIs
│   ├── reconciliation.py      # daily live-vs-backtest fill reconciliation
│   ├── pair_position_manager.py
│   └── strategies/
│       ├── base.py             # BaseStrategy (single-instrument, per-snapshot)
│       ├── portfolio_base.py   # PortfolioStrategy (cross-sectional, daily)
│       ├── pairs_trading.py    # Strategy A — cointegrated pairs
│       └── xsection_mean_reversion.py  # Strategy B — daily cross-sectional
├── stat/
│   ├── cointegration.py       # CADF test + OLS hedge ratio
│   └── half_life.py           # Ornstein-Uhlenbeck half-life estimation
├── interfaces/
│   ├── ibkr_feed.py           # IBKR market data (Stock contracts)
│   └── ibkr_broker.py         # IBKR order submission (Stock contracts)
├── simulation/
│   └── hist_data_us.py        # yfinance historical bar loader (adjustment-verified)
├── data/
│   └── sp500_universe.py      # point-in-time S&P 500 membership (Wikipedia history)
└── backtest/
    ├── engine.py                # event-driven tick replay (pairs strategy)
    ├── vector_engine.py         # vectorized pandas backtest (cross-sectional strategy)
    ├── walk_forward.py          # rolling IS/OOS optimizer (ported from forex-trading)
    ├── monte_carlo.py           # bootstrap P&L resampling (ported from forex-trading)
    └── reality_check.py         # White's Reality Check (ported from forex-trading)
dashboard/                       # FastAPI + React/Vite UI (ported pattern from forex-trading)
configs/                         # risk.yaml, strategy.yaml, goal.yaml
scripts/                         # run_backtest.py, decade_backtest.py, health_check.py
tests/                           # pytest — includes Chan-guard and wiring-parity tests
docs/                            # research notes, health-check reports
```

## Quick start

```powershell
pip install -r requirements.txt
cd frontend && npm install && cd ..

# Backtest (vectorized, cross-sectional strategy)
python scripts/run_backtest.py --strategy xsection_mean_reversion --start 2015-01-01 --end 2025-01-01

# Backtest (event-driven, pairs strategy)
python scripts/run_backtest.py --strategy pairs_trading --pairs configs/pairs.yaml

# Start dashboard (engine does not auto-trade; requires explicit start via UI or API)
python scripts/start_dashboard.py --host 127.0.0.1 --port 8082
```

## Tests

```powershell
pytest tests/ -v
```

Includes Chan-methodology guard tests (`test_chan_guards.py`, `test_lookahead_bias.py`)
and forex-lesson regression tests (`test_wiring_parity.py`, `test_config_enforcement.py`).

## Known limitations (MVP)

- Backtests use yfinance + a Wikipedia-derived point-in-time S&P 500 membership
  list — this is **not** a fully survivorship-bias-free, professional
  point-in-time database (e.g. CRSP, Polygon). Health-check reports carry an
  explicit disclaimer banner until a paid vendor is integrated.
- Phase-3 ML classification layer (LightGBM) is deferred — Chan's book argues
  against high-parameter-count models for retail-scale quant trading; the MVP
  strategies deliberately use ≤ 5 free parameters each.
