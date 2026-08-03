# US Equity Strategy Health Check

Generated: 2026-07-28T09:32:32.707678+00:00

> **Disclaimer**: backtests using `--demo` mode use synthetic data and validate
> PIPELINE CORRECTNESS only, not strategy edge. Backtests using real data use the
> FIXED top-N dollar-volume universe from configs/universe.yaml (built by
> scripts/refresh_universe.py — one liquidity snapshot applied across time, which
> carries a mild survivorship flavor documented in python/data/fixed_universe.py)
> and daily bars from the local price cache (IB Gateway ADJUSTED_LAST, yfinance
> fallback — the Data line above names the actual source). See README.md 'Known
> limitations (MVP)' before trusting these numbers for capital allocation decisions.

## pairs_trading

- Data: AMAT / LRCX daily bars via local price cache (ibkr)
- Free parameters: 5 (Chan Ch.3 ceiling: 5)
- Trading days tested: 1761
- pair: AMAT/LRCX
- n_trades: 8
- total_net_pnl: -10181.121712625007
- win_rate: 0.5
- sharpe_ratio: -0.5424946418874982
- max_drawdown: -0.010158674127666134

**Acceptance gates:**
- [x] sufficient_sample_size
- [x] has_trades
- [ ] monte_carlo_p5_sharpe_nonneg

**Overall: NO-GO**
