import { useState } from 'react'
import { getSymbolRegime } from '../api'
import type { RegimeReportDto } from '../types'

const STATE_COLOR: Record<string, string> = {
  Bull: 'var(--positive, #4caf50)',
  Bear: 'var(--negative, #e05252)',
  Sideways: 'var(--text-dim)',
}

const STATES = ['Bear', 'Sideways', 'Bull'] as const

function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`
}

/** Report-only Markov regime diagnostic panel — /api/regime/{symbol},
 * python/analytics/regime.py. Purely informational: nothing here places,
 * filters, or gates a trade. See that module's docstring for why the
 * "naive walk-forward" numbers are illustrative-only, not a validated
 * strategy result. */
export default function RegimePanel() {
  const [input, setInput] = useState('SPY')
  const [report, setReport] = useState<RegimeReportDto | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async (symbol: string) => {
    if (!symbol.trim()) return
    setLoading(true)
    setError(null)
    try {
      setReport(await getSymbolRegime(symbol))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setReport(null)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    void load(input)
  }

  return (
    <div>
      <div style={{ fontSize: 10, color: 'var(--text-faint)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 4 }}>
        Markov regime diagnostic (report-only — not wired to any strategy)
      </div>
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: 6, marginBottom: 10, alignItems: 'center' }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value.toUpperCase())}
          placeholder="TICKER"
          maxLength={6}
          className="num"
          style={{
            background: 'var(--bg)', border: '1px solid var(--border-light)', borderRadius: 2,
            color: 'var(--text)', padding: '5px 8px', fontSize: 12, width: 90, textTransform: 'uppercase',
          }}
        />
        <button type="submit" className="tws-btn primary" style={{ fontSize: 11, padding: '5px 10px' }} disabled={loading}>
          {loading ? '…' : 'Load'}
        </button>
        {report && (
          <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
            {report.symbol} as of {report.as_of} · window={report.window}d · threshold={pct(report.threshold)} · {report.n_days_labeled} labeled days
          </span>
        )}
      </form>

      {error && <div style={{ color: 'var(--negative, #e05252)', fontSize: 12, marginBottom: 8 }}>{error}</div>}

      {report && (
        <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-faint)', marginBottom: 4 }}>Current regime</div>
            <div style={{ fontSize: 20, fontWeight: 600, color: STATE_COLOR[report.current_state] }}>
              {report.current_state}
            </div>
          </div>

          <div>
            <div style={{ fontSize: 10, color: 'var(--text-faint)', marginBottom: 4 }}>Transition matrix (rows=from, cols=to)</div>
            <table>
              <thead>
                <tr>
                  <th></th>
                  {STATES.map((s) => <th key={s}>{s}</th>)}
                </tr>
              </thead>
              <tbody>
                {STATES.map((from) => (
                  <tr key={from}>
                    <td style={{ color: STATE_COLOR[from] }}>{from}</td>
                    {STATES.map((to) => (
                      <td
                        key={to}
                        className="num"
                        style={from === to ? { color: 'var(--text)', fontWeight: 600 } : { color: 'var(--text-dim)' }}
                      >
                        {pct(report.transition_matrix[from]?.[to] ?? 0)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div>
            <div style={{ fontSize: 10, color: 'var(--text-faint)', marginBottom: 4 }}>Long-run mix (stationary distribution)</div>
            <table>
              <thead><tr>{STATES.map((s) => <th key={s}>{s}</th>)}</tr></thead>
              <tbody>
                <tr>
                  {STATES.map((s) => (
                    <td key={s} className="num" style={{ color: STATE_COLOR[s] }}>
                      {pct(report.stationary_distribution[s] ?? 0)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>

          <div style={{ maxWidth: 320 }}>
            <div style={{ fontSize: 10, color: 'var(--text-faint)', marginBottom: 4 }}>Naive walk-forward (illustrative only)</div>
            {report.naive_backtest ? (
              <>
                <div style={{ fontSize: 12 }}>
                  Sharpe: <span className="num">{report.naive_backtest.sharpe_naive_no_cost.toFixed(2)}</span>
                  {'  ·  '}
                  Max DD: <span className="num">{pct(report.naive_backtest.max_drawdown_naive_no_cost)}</span>
                  {'  ·  '}
                  {report.naive_backtest.n_days} days
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-faint)', marginTop: 4 }}>{report.naive_backtest.note}</div>
              </>
            ) : (
              <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>Not enough history for even the naive illustration.</div>
            )}
          </div>
        </div>
      )}

      {!report && !error && !loading && (
        <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>Enter a ticker and click Load.</div>
      )}
    </div>
  )
}
