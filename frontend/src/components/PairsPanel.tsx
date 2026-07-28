import type { OpenPair, PairCandidate } from '../types'

interface Props {
  openPairs: OpenPair[]
  candidates: PairCandidate[]
}

export default function PairsPanel({ openPairs, candidates }: Props) {
  return (
    <div className="panel" style={{ overflow: 'auto', maxHeight: 320 }}>
      <h3 style={{ marginTop: 0, fontSize: 14 }}>Strategy A — Cointegrated Pairs</h3>

      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>Open positions</div>
      {openPairs.length === 0 ? (
        <div style={{ color: 'var(--text-dim)', fontSize: 13, marginBottom: 12 }}>No open pairs.</div>
      ) : (
        <table style={{ marginBottom: 16 }}>
          <thead><tr><th>Pair</th><th>Side</th><th>Entry Z</th><th>Since</th></tr></thead>
          <tbody>
            {openPairs.map((p) => (
              <tr key={`${p.code_a}-${p.code_b}`}>
                <td>{p.code_a} / {p.code_b}</td>
                <td>{p.side}</td>
                <td>{p.entry_z.toFixed(2)}</td>
                <td>{new Date(p.entry_time).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>Candidate pairs (last scan)</div>
      {candidates.length === 0 ? (
        <div style={{ color: 'var(--text-dim)', fontSize: 13 }}>No candidates yet.</div>
      ) : (
        <table>
          <thead><tr><th>Pair</th><th>CADF t-stat</th><th>Half-life (days)</th></tr></thead>
          <tbody>
            {candidates.map((c) => (
              <tr key={`${c.code_a}-${c.code_b}`}>
                <td>{c.code_a} / {c.code_b}</td>
                <td>{c.cadf_tstat.toFixed(2)}</td>
                <td>{c.half_life_days.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
