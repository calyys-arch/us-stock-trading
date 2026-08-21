import { getSignalJournalToday } from '../api'
import { usePolling } from '../hooks/usePolling'
import MosaicWindow from './MosaicWindow'

/** Compact rendering of a journal entry's free-form `context` dict —
 * every signals/*.py module fills this with a different set of keys
 * (see python/microstructure/signals/__init__.py's `MicroSignal.context`
 * docstring), so there's no fixed shape to destructure. A single-line
 * JSON string keeps this panel from needing per-strategy special-casing
 * while still surfacing the full triggering context on hover/inline. */
function formatContext(context: Record<string, unknown>): string {
  const entries = Object.entries(context)
  if (entries.length === 0) return '—'
  return entries
    .map(([k, v]) => `${k}=${typeof v === 'number' ? v.toLocaleString(undefined, { maximumFractionDigits: 4 }) : String(v)}`)
    .join(', ')
}

function formatPrice(value: number | null): string {
  return value == null ? '—' : value.toLocaleString(undefined, { maximumFractionDigits: 4 })
}

/** Read-only "今日訊號" (today's signals) panel — dashboard/app.py's
 * GET /api/signal_journal/today, python/microstructure/signal_journal.py.
 * Renders every live microstructure signal recorded so far today,
 * whether or not RiskEngine's gate approved it, so a viewer can see the
 * full firehose (not just the GO signals SignalsLog's `recent_signals`
 * already shows). Purely a display surface — no action here can submit
 * an order or otherwise touch engine state. */
export default function SignalJournalPanel() {
  const { data, error } = usePolling(getSignalJournalToday, 2000)

  if (error) {
    return (
      <MosaicWindow title="Today's Signal Journal" className="area-journal">
        <div style={{ color: 'var(--red)', fontSize: 12 }}>Cannot reach backend: {error}</div>
      </MosaicWindow>
    )
  }

  const signals = data?.signals ?? []

  return (
    <MosaicWindow title="Today's Signal Journal" className="area-journal">
      {signals.length === 0 ? (
        <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No signals recorded today yet.</div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Symbol</th>
              <th>Strategy</th>
              <th>Direction</th>
              <th>Entry</th>
              <th>Stop</th>
              <th>Target</th>
              <th>Risk Gate</th>
              <th>Context</th>
              <th>Outcome</th>
            </tr>
          </thead>
          <tbody>
            {signals.map((s, i) => (
              <tr key={`${s.symbol}-${s.signal_time}-${i}`}>
                <td className="num">{s.signal_time ? new Date(s.signal_time).toLocaleTimeString() : '—'}</td>
                <td className="num">{s.symbol}</td>
                <td>{s.strategy}</td>
                <td className={s.direction === 'long' ? 'positive' : 'negative'}>{s.direction.toUpperCase()}</td>
                <td className="num">{formatPrice(s.entry_price)}</td>
                <td className="num">{formatPrice(s.stop_price)}</td>
                <td className="num">{formatPrice(s.target_price)}</td>
                <td>
                  {s.risk_passed ? (
                    <span className="pill auto">Approved</span>
                  ) : (
                    <span className="pill error" title={s.rejection_reason ?? undefined}>
                      Filtered{s.rejection_reason ? `: ${s.rejection_reason}` : ''}
                    </span>
                  )}
                </td>
                <td style={{ color: 'var(--text-dim)', fontSize: 11 }} title={formatContext(s.context)}>
                  {formatContext(s.context)}
                </td>
                <td style={{ color: 'var(--text-dim)' }}>{s.outcome.status === 'pending' ? 'Pending' : 'Closed'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </MosaicWindow>
  )
}
