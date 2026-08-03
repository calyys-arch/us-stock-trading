import { useState } from 'react'
import { runPairsDemoScan } from '../api'
import type { OpenPair, PairCandidate } from '../types'
import ZScoreChart from './charts/ZScoreChart'

interface Props {
  openPairs: OpenPair[]
  candidates: PairCandidate[]
}

export default function PairsPanel({ openPairs, candidates }: Props) {
  const [loading, setLoading] = useState(false)

  const handleScan = async () => {
    setLoading(true)
    try {
      await runPairsDemoScan()
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div style={{ fontSize: 10, color: 'var(--text-faint)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 4 }}>
        Open positions
      </div>
      {openPairs.length === 0 ? (
        <div style={{ color: 'var(--text-dim)', fontSize: 12, marginBottom: 14 }}>No open pairs.</div>
      ) : (
        <table style={{ marginBottom: 16 }}>
          <thead><tr><th>Pair</th><th>Side</th><th>Entry Z</th><th>Since</th></tr></thead>
          <tbody>
            {openPairs.map((p) => (
              <tr key={`${p.code_a}-${p.code_b}`}>
                <td className="num">{p.code_a} / {p.code_b}</td>
                <td>{p.side}</td>
                <td className="num">{p.entry_z.toFixed(2)}</td>
                <td className="num">{new Date(p.entry_time).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
        <div style={{ fontSize: 10, color: 'var(--text-faint)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          Candidate pairs (last scan)
        </div>
        <button
          onClick={handleScan}
          disabled={loading}
          className="tws-btn primary"
          style={{ fontSize: 10, padding: '2px 8px' }}
        >
          {loading ? 'Scanning…' : 'Run demo pair scan'}
        </button>
      </div>
      {candidates.length === 0 ? (
        <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No candidates yet.</div>
      ) : (
        <table>
          <thead><tr><th>Pair</th><th>CADF t-stat</th><th>Half-life (days)</th><th>Spread z-score</th></tr></thead>
          <tbody>
            {candidates.map((c) => (
              <tr key={`${c.code_a}-${c.code_b}`}>
                <td className="num">{c.code_a} / {c.code_b}</td>
                <td className="num">{c.cadf_tstat.toFixed(2)}</td>
                <td className="num">{c.half_life_days.toFixed(1)}</td>
                <td><ZScoreChart values={c.zscore_history ?? []} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
