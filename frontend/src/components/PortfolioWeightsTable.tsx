interface Props {
  weights: Record<string, number>
}

export default function PortfolioWeightsTable({ weights }: Props) {
  const entries = Object.entries(weights).sort((a, b) => b[1] - a[1])

  return (
    <div className="panel" style={{ overflow: 'auto', maxHeight: 320 }}>
      <h3 style={{ marginTop: 0, fontSize: 14 }}>Strategy B — Cross-Sectional Target Weights</h3>
      {entries.length === 0 ? (
        <div style={{ color: 'var(--text-dim)', fontSize: 13 }}>No target weights yet — run a backtest or start the engine.</div>
      ) : (
        <table>
          <thead>
            <tr><th>Code</th><th>Weight</th><th>Side</th></tr>
          </thead>
          <tbody>
            {entries.map(([code, w]) => (
              <tr key={code}>
                <td>{code}</td>
                <td className={w >= 0 ? 'positive' : 'negative'}>{(w * 100).toFixed(2)}%</td>
                <td className={w >= 0 ? 'positive' : 'negative'}>{w >= 0 ? 'LONG' : 'SHORT'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
