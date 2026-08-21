import { useState } from 'react'
import { getState, startEngine, stopEngine } from './api'
import BacktestPanel from './components/BacktestPanel'
import Header from './components/Header'
import MosaicWindow from './components/MosaicWindow'
import PairsPanel from './components/PairsPanel'
import PortfolioWeightsTable from './components/PortfolioWeightsTable'
import PositionsPanel from './components/PositionsPanel'
import RegimePanel from './components/RegimePanel'
import SignalJournalPanel from './components/SignalJournalPanel'
import SignalsLog from './components/SignalsLog'
import StatCards from './components/StatCards'
import StatusBar from './components/StatusBar'
import SymbolChartPanel, { type MultiChartJump } from './components/SymbolChartPanel'
import { usePolling } from './hooks/usePolling'

export default function App() {
  const { data: state, error } = usePolling(getState, 2000)
  const [chartJump, setChartJump] = useState<MultiChartJump | null>(null)

  if (!state) {
    return (
      <div style={{ padding: 40, color: 'var(--text-dim)' }}>
        {error ? `Cannot reach backend: ${error}. Is the FastAPI server running?` : 'Loading...'}
      </div>
    )
  }

  return (
    <div className="tws-shell">
      <Header
        running={state.running}
        mode={state.mode}
        dataSource={state.data_source}
        ibkrBrokerConnected={state.ibkr_broker_connected}
        ibkrFeedConnected={state.ibkr_feed_connected}
        futuLiveFeedActive={state.futu_live_feed_active}
        armedStrategies={state.armed_strategies}
        pairsRegimeGateOpen={state.pairs_regime_gate_open}
        pairsRegimeGateReason={state.pairs_regime_gate_reason}
        onStart={() => startEngine()}
        onStop={() => stopEngine()}
      />
      <div className="mosaic-grid">
        <MosaicWindow title="Account & Risk" className="area-account">
          <StatCards state={state} />
        </MosaicWindow>

        <MosaicWindow title="Backtest Summary" className="area-backtest">
          <BacktestPanel summary={state.latest_backtest_summary} />
        </MosaicWindow>

        <MosaicWindow title="Strategy B — Cross-Sectional Target Weights" className="area-weights">
          <PortfolioWeightsTable weights={state.latest_portfolio_weights} />
        </MosaicWindow>

        <MosaicWindow title="Strategy A — Cointegrated Pairs" className="area-pairs">
          <PairsPanel openPairs={state.open_pairs} candidates={state.pair_candidates} />
        </MosaicWindow>

        <MosaicWindow title="Open Positions" className="area-positions">
          <PositionsPanel onChartSymbol={(code, slot) => setChartJump({ symbol: code, nonce: Date.now(), slot })} />
        </MosaicWindow>

        <MosaicWindow title="Symbol Chart" className="area-chart">
          <SymbolChartPanel symbols={state.symbols} jumpTo={chartJump} />
        </MosaicWindow>

        <div className="area-signals">
          <SignalsLog signals={state.recent_signals} reports={state.recent_execution_reports} />
        </div>

        <SignalJournalPanel />

        <MosaicWindow title="Market Regime (Report-Only Diagnostic)" className="area-regime">
          <RegimePanel />
        </MosaicWindow>
      </div>
      <StatusBar state={state} />
    </div>
  )
}
