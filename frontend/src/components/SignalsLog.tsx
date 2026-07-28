import type { ExecutionReportDto, SignalDto } from '../types'

interface Props {
  signals: SignalDto[]
  reports: ExecutionReportDto[]
}

export default function SignalsLog({ signals, reports }: Props) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
      <div className="panel" style={{ overflow: 'auto', maxHeight: 260 }}>
        <h3 style={{ marginTop: 0, fontSize: 14 }}>Recent Signals</h3>
        {signals.length === 0 ? (
          <div style={{ color: 'var(--text-dim)', fontSize: 13 }}>No signals yet.</div>
        ) : (
          <table>
            <thead><tr><th>Strategy</th><th>Time</th></tr></thead>
            <tbody>
              {signals.map((s, i) => (
                <tr key={s.id ?? i}>
                  <td>{s.strategy}</td>
                  <td>{s.timestamp ? new Date(s.timestamp).toLocaleTimeString() : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="panel" style={{ overflow: 'auto', maxHeight: 260 }}>
        <h3 style={{ marginTop: 0, fontSize: 14 }}>Execution Reports</h3>
        {reports.length === 0 ? (
          <div style={{ color: 'var(--text-dim)', fontSize: 13 }}>No execution reports yet.</div>
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
      </div>
    </div>
  )
}
