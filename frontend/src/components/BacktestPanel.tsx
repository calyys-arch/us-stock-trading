import { useState } from 'react'
import { runDemoBacktest } from '../api'
import type { BacktestSummary } from '../types'

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
    <div className="panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>Backtest Summary</h3>
        <button
          onClick={handleRun}
          disabled={loading}
          style={{ background: 'transparent', border: '1px solid var(--accent)', color: 'var(--accent)', borderRadius: 6, padding: '5px 12px', fontSize: 12 }}
        >
          {loading ? 'Running...' : 'Run demo backtest'}
        </button>
      </div>
      {summary?.label && (
        <div style={{ fontSize: 11, color: 'var(--amber)', marginTop: 6 }}>{summary.label}</div>
      )}
      {summary ? (
        <div style={{ display: 'flex', gap: 24, marginTop: 12 }}>
          <Metric label="Sharpe (ann.)" value={summary.sharpe_annualized?.toFixed(2)} />
          <Metric label="Max Drawdown" value={`${(summary.max_drawdown * 100).toFixed(1)}%`} />
          <Metric label="CAGR" value={`${(summary.cagr * 100).toFixed(1)}%`} />
          <Metric label="Days" value={String(summary.n_days)} />
        </div>
      ) : (
        <div style={{ color: 'var(--text-dim)', fontSize: 13, marginTop: 8 }}>
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
      <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 700 }}>{value ?? '—'}</div>
    </div>
  )
}
