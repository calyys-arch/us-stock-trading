import { useState } from 'react'
import { runDemoBacktest } from '../api'
import type { BacktestSummary } from '../types'
import EquityAreaChart from './charts/EquityAreaChart'

interface Props {
  summary: BacktestSummary | null
}

export default function BacktestPanel({ summary }: Props) {
  const [loading, setLoading] = useState(false)

  const handleRun = async () => {
    setLoading(true)
    try {
      await runDemoBacktest()
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <button
        onClick={handleRun}
        disabled={loading}
        className="tws-btn primary"
        style={{ fontSize: 11, padding: '4px 10px', marginBottom: 10 }}
      >
        {loading ? 'Running…' : 'Run demo backtest'}
      </button>
      {summary?.label && (
        <div style={{ fontSize: 11, color: 'var(--amber)', marginBottom: 8 }}>{summary.label}</div>
      )}
      {summary ? (
        <div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 18, marginBottom: 12 }}>
            <Metric label="Sharpe (ann.)" value={summary.sharpe_annualized?.toFixed(2)} />
            <Metric label="Max Drawdown" value={`${(summary.max_drawdown * 100).toFixed(1)}%`} />
            <Metric label="CAGR" value={`${(summary.cagr * 100).toFixed(1)}%`} />
            <Metric label="Days" value={String(summary.n_days)} />
          </div>
          {summary.equity_curve && summary.equity_curve.length > 1 && (
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-faint)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 4 }}>
                Equity curve (indexed to 100)
              </div>
              <EquityAreaChart data={summary.equity_curve} height={140} />
            </div>
          )}
        </div>
      ) : (
        <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>
          No backtest run yet. For real research use scripts/run_backtest.py — this panel runs a
          synthetic-data demo only.
        </div>
      )}
    </div>
  )
}

function Metric({ label, value }: { label: string; value?: string }) {
  return (
    <div>
      <div style={{ fontSize: 10, color: 'var(--text-dim)', textTransform: 'uppercase' }}>{label}</div>
      <div className="num" style={{ fontSize: 15, fontWeight: 700 }}>{value ?? '—'}</div>
    </div>
  )
}
