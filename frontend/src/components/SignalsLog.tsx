import type { ExecutionReportDto, SignalDto } from '../types'
import MosaicWindow from './MosaicWindow'

interface Props {
  signals: SignalDto[]
  reports: ExecutionReportDto[]
}

export default function SignalsLog({ signals, reports }: Props) {
  return (
    <div className="signals-grid">
      <MosaicWindow title="Recent Signals">
        {signals.length === 0 ? (
          <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No signals yet.</div>
        ) : (
          <table>
            <thead><tr><th>Strategy</th><th>Time</th></tr></thead>
            <tbody>
              {signals.map((s, i) => (
                <tr key={s.id ?? i}>
                  <td>{s.strategy}</td>
                  <td className="num">{s.timestamp ? new Date(s.timestamp).toLocaleTimeString() : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </MosaicWindow>

      <MosaicWindow title="Execution Reports">
        {reports.length === 0 ? (
          <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No execution reports yet.</div>
        ) : (
          <table>
            <thead><tr><th>Type</th><th>Reason</th></tr></thead>
            <tbody>
              {reports.map((r, i) => (
                <tr key={i}>
                  <td>{r.type}</td>
                  <td>{r.reason ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </MosaicWindow>
    </div>
  )
}
