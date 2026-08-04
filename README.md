# US Equity Quant Trading System

A US-equities algorithmic day-trading system built on Ernest P. Chan's
*Quantitative Trading: How to Build Your Own Algorithmic Trading Business*
(Wiley, 2009). Architecture is derived from a proven forex event-driven
system (`../forex-trading`), rebuilt for US equities with Chan's statistical
methodology as the strategy layer.

> **System scope**: US-listed equities only. No forex, no options, no crypto.
> Two strategies: cointegrated pairs trading (can hold overnight) and
> intraday cross-sectional mean reversion (flat by 15:55 ET). Tradeable
> universe: point-in-time (S&P 500 UNION Nasdaq-100), narrowed daily to the
> most liquid names by trailing dollar volume — not S&P-500-only — so
> heavily-traded Nasdaq names that the S&P Index Committee excludes (e.g.
> recently-IPO'd mega-caps) aren't systematically missed.

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
        (point-in-time S&P 500 ∪ Nasdaq-100,   MessageBus
         top-K by trailing $ volume)  │        │
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
│   ├── ibkr_broker.py         # IBKR order submission (Stock contracts)
│   ├── finnhub_calendar.py   # Finnhub earnings calendar -> ReferenceData.is_earnings_today
│   └── finnhub_news.py       # Finnhub company news -> DataEngine.news_event_checker;
│                              # general market headlines (informational only, see file docstring)
├── simulation/
│   └── hist_data_us.py        # yfinance historical bar loader (adjustment-verified)
├── data/
│   ├── wiki_fetch.py           # shared Wikipedia table-fetch helper (UA header workaround)
│   ├── index_membership.py     # shared point-in-time walk-backward algorithm
│   ├── sp500_universe.py       # point-in-time S&P 500 membership (Wikipedia history)
│   ├── nasdaq100_universe.py   # point-in-time Nasdaq-100 membership (Wikipedia history)
│   └── liquid_universe.py      # (S&P 500 ∪ Nasdaq-100) unioned, narrowed by trailing $ volume
└── backtest/
    ├── engine.py                # event-driven tick replay (pairs strategy)
    ├── vector_engine.py         # vectorized pandas backtest (cross-sectional strategy)
    ├── walk_forward.py          # rolling IS/OOS optimizer (ported from forex-trading)
    ├── monte_carlo.py           # bootstrap P&L resampling (ported from forex-trading)
    └── reality_check.py         # White's Reality Check (ported from forex-trading)
dashboard/                       # FastAPI + React/Vite UI (ported pattern from forex-trading)
configs/                         # risk.yaml, strategy.yaml, goal.yaml, broker.yaml
scripts/                         # run_backtest.py, decade_backtest.py, health_check.py
tests/                           # pytest — includes Chan-guard and wiring-parity tests
docs/                            # research notes, design plans (hand-written, not regenerated by scripts)
backtests/                       # ALL generated backtest reports/logs, one place for future review:
├── reports/                     #   us_equity_health_check.md, signal_trap_report.md,
│                                 #   self_improvement_log.md, intraday_backtest_report.{md,json}
└── logs/                        #   promotion_history.jsonl (machine-readable WFO decision audit trail)
```

## Quick start

```powershell
pip install -r requirements.txt
cd frontend && npm install && cd ..

# Optional: enables ReferenceData.is_earnings_today AND DataEngine's
# news_event_checker (same-day company news), both of which
# scripts/pick_10.py uses to exclude names from today's universe before
# calling the strategy (see that script's module docstring). Without a key
# everything fails safe — no exclusions are applied, same as before these
# integrations existed.
cp .env.example .env   # then fill in FINNHUB_API_KEY (free tier at finnhub.io)

# One-time: build the FIXED top-20 dollar-volume backtest universe
# (configs/universe.yaml) and pre-warm the local price cache (data/history/).
# Prices come from IB Gateway when it is running (configs/broker.yaml
# historical_data_source: ibkr), else fall back to yfinance with a warning.
python scripts/refresh_universe.py

# Backtest (vectorized, cross-sectional strategy) — uses the fixed universe +
# local price cache; add --refresh-data to force re-fetch
python scripts/run_backtest.py --strategy xsection_mean_reversion --start 2015-01-01 --end 2025-01-01

# Backtest (event-driven, pairs strategy)
python scripts/run_backtest.py --strategy pairs_trading --pairs configs/pairs.yaml

# Self-improving WFO loop: grid-search + walk-forward validation + gated
# auto write-back of better parameters to configs/strategy.yaml
# (report-only variant: add --no-write). History: backtests/logs/promotion_history.jsonl
# + backtests/reports/self_improvement_log.md. auto_execute is NEVER touched.
python scripts/self_improve_loop.py --demo          # offline synthetic smoke run
python scripts/self_improve_loop.py --iterations 3  # real data, fixed universe

# Report-only diagnostics data (optional): SEC EDGAR 8-K backfill + Finnhub
# news/calendars for the signal-trap report (backtests/reports/signal_trap_report.md),
# and continuous tick/L2 depth capture for its order-book heuristics
python scripts/refresh_event_data.py
python scripts/capture_market_microstructure.py                # source: IB Gateway (default)
python scripts/capture_market_microstructure.py --source futu   # source: Futu/Moomoo OpenD

# Dark-pool-internalization context (Tier 2, coarse/weekly, optional): FINRA's
# free public OTC Transparency API. `recent` covers roughly the trailing
# 3-4 years per symbol in a few minutes; `historic` fills in older weeks but
# is a slow, resumable, per-week market-wide pull (see the script's docstring
# for the time budget before running it for the full universe/history).
python scripts/backfill_finra_ats.py --phase recent
python scripts/backfill_finra_ats.py --phase historic --start 2018-01-01

# Start dashboard (engine does not auto-trade; requires explicit start via UI or API)
python scripts/start_dashboard.py --host 127.0.0.1 --port 8082
```

### Simulated vs. real IBKR paper connection

By default (`configs/broker.yaml: data_source: simulated`), clicking **Start
(Paper)** in the dashboard runs a fully in-memory `SimulatedFeed` +
`SimBroker` — no external connection of any kind, no IB Gateway/TWS needed.

To make **Start (Paper)** connect to a real IB Gateway/TWS paper account,
edit `configs/broker.yaml`:

```yaml
data_source: ibkr_paper
ibkr:
  host: 127.0.0.1
  feed_port: 4002      # IB Gateway paper=4002, TWS paper=7497
  broker_port: 4002
  feed_client_id: 11
  broker_client_id: 21
```

and restart the dashboard backend. IB Gateway/TWS must already be running
and logged into the **paper** account with API access enabled — the
dashboard's toolbar/status bar show `IBKR PAPER — connected` or
`IBKR PAPER — disconnected` based on the real connection state, it never
silently pretends to be live. This is a config-file switch, not a UI
toggle, by design (same philosophy as `configs/strategy.yaml`'s
`auto_execute` — see `docs/lessons_from_forex_trading.md`).

### Tick/L2 capture: IB Gateway vs. Futu/Moomoo (`--source`)

`scripts/capture_market_microstructure.py` (report-only; never touches the
trading engine — see "Known limitations" below) supports two interchangeable
sources for the tick-by-tick + Level-2 archive that feeds
`python/signals/trap_detector.py`:

- `--source ibkr` (default) — `python/interfaces/ibkr_tick_capture.py`.
  Needs IB Gateway/TWS running + logged in, **and** a real-time
  market-data subscription on the connected account. A free-standing IBKR
  **Demo** account (Client Portal shows "This is not a brokerage account" /
  Customer Type "Individual (Demo)") cannot get one — Demo accounts have no
  linked live account to subscribe against, so they are permanently capped
  at 15-20min delayed data (`Error 10189`/`10190`, `Warning 2152`). Only a
  Paper Trading account created *from* a real (KYC-approved, not
  necessarily funded) live account can share that live account's real-time
  subscriptions (Client Portal → Settings → Paper Trading Account → Share
  real-time market data).
- `--source futu` — `python/interfaces/futu_tick_capture.py`. Needs
  Futu/Moomoo's local **OpenD** gateway app running + logged in (default
  port 11111, `configs/broker.yaml`'s `futu:` block), with a funded account
  that has LV3 (or better) US-equity quote permission. Genuinely real-time,
  no subscription dead-end. Trade-offs vs. IB: no per-print venue/condition
  codes (Futu's US Ticker feed doesn't expose them), so
  `dark_pool_internalization_score` and `print_lag_score` correctly report
  "unavailable" for Futu-sourced days; and Level-2 depth arrives as full
  ordered snapshots rather than IB's native insert/update/delete diff
  stream, so `futu_tick_capture.py` synthesizes diff events by comparing
  consecutive snapshots position-by-position (an approximation, documented
  in that module's docstring). Both sources write into the SAME
  `data/ticks/` + `data/depth/` archive, tagged `"source": "futu"` /
  `"source": "ibkr"` per row.

## Tests

```powershell
pytest tests/ -v
```

Includes Chan-methodology guard tests (`test_chan_guards.py`, `test_lookahead_bias.py`)
and forex-lesson regression tests (`test_wiring_parity.py`, `test_config_enforcement.py`).

## Known limitations (MVP)

- Backtests use yfinance + a Wikipedia-derived point-in-time (S&P 500 UNION
  Nasdaq-100) membership list, further narrowed daily by trailing dollar
  volume (`python/data/liquid_universe.py`) — this is **not** a fully
  survivorship-bias-free, professional point-in-time database (e.g. CRSP,
  Polygon), and Wikipedia's Nasdaq-100 "changes" table has the same
  best-effort completeness caveat as the S&P 500 one. Health-check reports
  carry an explicit disclaimer banner until a paid vendor is integrated.
- Phase-3 ML classification layer (LightGBM) is deferred — Chan's book argues
  against high-parameter-count models for retail-scale quant trading; the MVP
  strategies deliberately use ≤ 5 free parameters each.
- Earnings calendar (`finnhub_calendar.py` -> `ReferenceData.is_earnings_today`)
  and company news (`finnhub_news.py` -> `DataEngine.news_event_checker`)
  are both live via Finnhub's free tier (`FINNHUB_API_KEY` in `.env`).
  `scripts/pick_10.py` is the one place that actually *consumes* both flags
  today — it drops any symbol with earnings or company news today from the
  universe passed into `strategy.evaluate()` (per `PortfolioStrategy`'s
  docstring: exclusion is the universe builder's job, not the strategy's).
  The always-running dashboard engine (`dashboard/engine_bridge.py`) wires
  both providers into `DataEngine` so `MarketSnapshot.is_earnings_today` /
  `.has_news_event` are populated, but the live engine does not yet run
  `CrossSectionalMeanReversionStrategy.evaluate()` on a daily schedule at
  all (no daily scheduler exists yet) — `pick_10.py` is currently the only
  place where the exclusion has a real effect on a portfolio decision.
- General *market-wide* headlines (`finnhub_news.py`'s
  `general_market_headlines_today()`, Finnhub `/news?category=general`)
  are fetched and printed by `scripts/pick_10.py` for context, but are
  deliberately NOT wired into any exclusion filter — there is no
  principled threshold for "how many general-news items today = an
  excludable day" without inventing an unenforced magic number, which
  `architecture-rules.mdc`'s Chan discipline forbids.
- Dark-pool/internalization detection (`python/signals/trap_detector.py`)
  is two-tiered and both tiers are report-only: Tier 1
  (`pinging_score`, `dark_pool_internalization_score`) reads OUR OWN
  captured tick archive (`data/ticks/`) and its off-exchange venue-code set
  (`DARK_POOL_EXCHANGE_CODES`) is a best-effort guess UNVERIFIED against a
  real capture sample as of 2026-07-29 — calibrate it once
  `scripts/capture_market_microstructure.py` has run for a few sessions.
  Tier 2 (`python/data/finra_ats.py` -> `dark_pool_participation_elevated`
  event flag) is a coarse, symbol-relative, week-level ATS-participation
  flag from FINRA's free public API — it has no per-day/per-signal
  granularity and needs `scripts/backfill_finra_ats.py` run beforehand
  (cache-only at report time, same as every other evidence source here).
  A third sub-score, `print_lag_score`, flags a day's share of prints
  carrying a CTA/UTP late/out-of-sequence condition code (L/Z/U/T — "this
  print's timestamp isn't when the trade actually happened"); its
  `LATE_PRINT_CONDITION_CODES` set is, same as `DARK_POOL_EXCHANGE_CODES`,
  UNVERIFIED against a real `data/ticks/` capture sample as of 2026-07-29.
  A fourth sub-score, `order_flow_imbalance_score`, estimates buy-vs-sell
  AGGRESSOR volume imbalance with the plain tick rule (Lee, 1991) since
  captured trades carry no true side flag and there is no time-aligned
  quote to run the fuller Lee-Ready test against; its accuracy is a
  documented ~85-90% in the literature on other markets/eras and, same as
  the two sub-scores above, UNVERIFIED against IB's actual tick-by-tick feed
  as of 2026-07-29.
- **S4 — L2 Absorption (`python/microstructure/signals/l2_absorption.py`)**
  is a BAR-ONLY proxy for the plan's full L2-confirmed absorption signal
  (docs/microstructure_pivot_plan.md §1/§4b): high-volume bar touches a
  recent support/resistance level without closing through it. The real
  version needs `python/backtest/depth_replay.py` (Phase 3, not yet
  built) plus weeks of `scripts/capture_market_microstructure.py` depth
  archive (`data/depth/` is currently empty — nothing has been captured
  yet, and IB provides no *historical* depth to backfill from, only
  going-forward capture) to confirm the level is being defended by a
  resting/iceberg order rather than just being a quiet, illiquid bar.
  Every signal this module emits is tagged
  `context["tier"] = "bar_only_proxy_no_l2_confirmation"`. It is wired
  into `dashboard/app.py`'s `/api/chart/{symbol}/context` overlay
  (`configs/strategy.yaml`'s `l2_absorption` block, `enabled: true`) for
  observe-only visualization, but deliberately left OUT of
  `scripts/run_intraday_backtest.py`'s `SIGNALS` list — per the plan, it
  cannot earn a WFO GO/promotion until real L2 confirmation exists.
- **Order type policy (all strategies): Limit/Stop-Limit only, never
  Market.** `ExecutionGateway._submit_order` (`python/core/
  execution_gateway.py`) is the ONE chokepoint every order-submitting
  method in that class routes through, and it hard-rejects `order_type=
  "market"` and any limit/stop-limit request missing a valid price — it
  never silently falls back to a market order. `RiskEngine.
  qualify_spread_order` / `qualify_portfolio_order` /
  `qualify_microstructure_order` (`python/core/risk_engine.py`) compute a
  bounded "marketable limit" price (`limit_price_buffer_bps` etc. in
  `configs/risk.yaml`) from the same snapshot price used for sizing, so
  the exact worst-case execution price is always known in advance — this
  is deliberate (control execution price, avoid PFOF-routed market-order
  fills), not an oversight. `flatten_intraday_positions` /
  `flatten_position` / `emergency_flatten_all` need their own live price
  source (`ExecutionGateway.set_price_lookup`) to build a bounded exit
  limit; until one is wired into `dashboard/engine_bridge.py`, those calls
  SKIP a position (loud warning) rather than ever falling back to a market
  order — flatten-completion guarantees were explicitly traded away in
  favor of price control. `SimBroker`/`IbkrBroker.place_order()` still
  technically accept `order_type="market"` as a low-level capability for
  direct/unit-test callers; the enforcement lives at the gateway
  chokepoint, per `architecture-rules.mdc`'s "ExecutionGateway is the ONLY
  module allowed to submit orders" rule (see `tests/
  test_never_market_orders.py`).
- **RiskEngine/ExecutionGateway microstructure extensions**
  (`qualify_microstructure_order`, `DailyLossTracker`,
  `python/core/event_blackout.py`, `_on_microstructure_order`) size and
  qualify one `MicroSignal` at a time — daily-loss kill switch
  (`max_daily_loss_pct`), event blackout (earnings/8-K/econ calendar, hard
  reject unlike `trap_detector`'s report-only near-event flags), and a
  max-open-positions cap. Same caveat as the daily strategies above: this
  pipeline is fully tested (`tests/test_risk_engine_microstructure.py`,
  `tests/test_never_market_orders.py`) but nothing yet calls
  `qualify_microstructure_order` from a live 1-minute scheduler — that
  wiring, plus a real `price_lookup` for flatten/EOD-close, is the next
  step before any of this can place a real order.
