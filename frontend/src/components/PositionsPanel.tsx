import { useState } from 'react'
import { flattenPosition, getPositions } from '../api'
import { usePolling } from '../hooks/usePolling'

interface Props {
  /** slot: which of the two chart slots (see SymbolChartPanel) to load the
   * symbol into. */
  onChartSymbol: (code: string, slot: 0 | 1) => void
}

/** Real per-symbol broker positions (GET /api/positions) — the system may
 * hold several stocks at once across both strategies, so each row gets its
 * own Exit button (in addition to the header's global "Exit All
 * Positions"). This is independent of the pairs/weights panels, which show
 * strategy INTENT, not actual broker fills. */
export default function PositionsPanel({ onChartSymbol }: Props) {
  const { data, error } = usePolling(getPositions, 2000)
  const [busyCode, setBusyCode] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)

  const positions = data?.positions ?? []

  const handleExit = async (code: string) => {
    if (!window.confirm(`Exit position in ${code}?\n\nThis immediately market-closes it, right now.`)) return
    setBusyCode(code)
    setActionError(null)
    try {
      await flattenPosition(code)
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusyCode(null)
    }
  }

  if (error) {
    return <div style={{ color: 'var(--red)', fontSize: 12 }}>Cannot reach backend: {error}</div>
  }

  if (positions.length === 0) {
    return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No open positions.</div>
  }

  return (
    <div>
      {actionError && <div style={{ color: 'var(--red)', fontSize: 11, marginBottom: 6 }}>{actionError}</div>}
      <table>
        <thead>
          <tr><th>Symbol</th><th>Qty</th><th>Side</th><th></th></tr>
        </thead>
        <tbody>
          {positions.map((p) => (
            <tr key={p.code}>
              <td className="num">{p.code}</td>
              <td className="num">{Math.abs(p.qty).toLocaleString()}</td>
              <td className={p.side === 'long' ? 'positive' : 'negative'}>{p.side.toUpperCase()}</td>
              <td style={{ display: 'flex', gap: 4, justifyContent: 'flex-end' }}>
                <span style={{ fontSize: 9, color: 'var(--text-faint)', alignSelf: 'center' }}>Chart:</span>
                <button
                  className="tws-btn"
                  style={{ fontSize: 10, padding: '2px 7px' }}
                  onClick={() => onChartSymbol(p.code, 0)}
                  title="Load into left chart slot"
                >
                  1
                </button>
                <button
                  className="tws-btn"
                  style={{ fontSize: 10, padding: '2px 7px' }}
                  onClick={() => onChartSymbol(p.code, 1)}
                  title="Load into right chart slot"
                >
                  2
                </button>
                <button
                  className="tws-btn danger"
                  style={{ fontSize: 10, padding: '2px 8px' }}
                  onClick={() => handleExit(p.code)}
                  disabled={busyCode === p.code}
                >
                  Exit
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
