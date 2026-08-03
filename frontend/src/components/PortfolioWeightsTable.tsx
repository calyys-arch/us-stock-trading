import WeightsBarChart from './charts/WeightsBarChart'

interface Props {
  weights: Record<string, number>
}

export default function PortfolioWeightsTable({ weights }: Props) {
  const entries = Object.entries(weights).sort((a, b) => b[1] - a[1])

  if (entries.length === 0) {
    return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No target weights yet — run a backtest or start the engine.</div>
  }

  return (
    <div>
      <WeightsBarChart items={entries.map(([code, w]) => ({ label: code, value: w }))} />
      <table>
        <thead>
          <tr><th>Code</th><th>Weight</th><th>Side</th></tr>
        </thead>
        <tbody>
          {entries.map(([code, w]) => (
            <tr key={code}>
              <td className="num">{code}</td>
              <td className={`num ${w >= 0 ? 'positive' : 'negative'}`}>{(w * 100).toFixed(2)}%</td>
              <td className={w >= 0 ? 'positive' : 'negative'}>{w >= 0 ? 'LONG' : 'SHORT'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
