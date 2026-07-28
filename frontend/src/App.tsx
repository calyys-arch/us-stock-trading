import { getState, startEngine, stopEngine } from './api'
import BacktestPanel from './components/BacktestPanel'
import Header from './components/Header'
import PairsPanel from './components/PairsPanel'
import PortfolioWeightsTable from './components/PortfolioWeightsTable'
import SignalsLog from './components/SignalsLog'
import StatCards from './components/StatCards'
import { usePolling } from './hooks/usePolling'

export default function App() {
  const { data: state, error } = usePolling(getState, 2000)

  if (!state) {
    return (
      <div style={{ padding: 40, color: 'var(--text-dim)' }}>
        {error ? `Cannot reach backend: ${error}. Is the FastAPI server running?` : 'Loading...'}
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'auto' }}>
      <Header
        running={state.running}
        mode={state.mode}
        onStart={() => startEngine()}
        onStop={() => stopEngine()}
      />
      <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 16 }}>
        <StatCards state={state} />
        <BacktestPanel summary={state.latest_backtest_summary} />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <PortfolioWeightsTable weights={state.latest_portfolio_weights} />
          <PairsPanel openPairs={state.open_pairs} candidates={state.pair_candidates} />
        </div>
        <SignalsLog signals={state.recent_signals} reports={state.recent_execution_reports} />
      </div>
    </div>
  )
}
