# US Equity Strategy Health Check

Generated: 2026-07-28T04:30:01.015567+00:00

> **Disclaimer**: backtests using `--demo` mode use synthetic data and validate
> PIPELINE CORRECTNESS only, not strategy edge. Backtests using real data rely on
> yfinance + a Wikipedia-derived point-in-time S&P 500 universe — see
> README.md 'Known limitations (MVP)' before trusting these numbers for capital
> allocation decisions.

## xsection_mean_reversion

- Data: SYNTHETIC DEMO DATA — not a real market backtest
- Free parameters: 3 (Chan Ch.3 ceiling: 5)
- Trading days tested: 970
- sharpe_annualized: -1.002933750774609
- max_drawdown: -0.2089932978936806
- cagr: -0.0456917671720356

**Acceptance gates:**
- [x] sufficient_sample_size
- [ ] monte_carlo_p5_sharpe_nonneg
- [ ] reality_check_pass

**Overall: NO-GO**

## pairs_trading

- Data: SYNTHETIC DEMO DATA (genuinely cointegrated by construction)
- Free parameters: 5 (Chan Ch.3 ceiling: 5)
- Trading days tested: 1000
- pair: SYNA/SYNB
- n_trades: 33
- total_net_pnl: 170464.47367641493
- win_rate: 0.6666666666666666
- sharpe_ratio: 0.7435430039319609
- max_drawdown: -0.07020671328288397

**Acceptance gates:**
- [ ] sufficient_sample_size
- [x] has_trades
- [ ] monte_carlo_p5_sharpe_nonneg

**Overall: NO-GO**
