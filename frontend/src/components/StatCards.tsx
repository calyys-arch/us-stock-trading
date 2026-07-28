import type { DashboardStateDto } from '../types'

interface Props {
  state: DashboardStateDto
}

export default function StatCards({ state }: Props) {
  const netLiq = state.account_summary?.NetLiquidation
  const buyingPower = state.account_summary?.BuyingPower
  const openPairsCount = state.open_pairs.length
  const portfolioNames = Object.keys(state.latest_portfolio_weights || {}).length

  const cards = [
    { label: 'Net Liquidation', value: netLiq !== undefined ? `$${netLiq.toLocaleString()}` : '—' },
    { label: 'Buying Power', value: buyingPower !== undefined ? `$${buyingPower.toLocaleString()}` : '—' },
    { label: 'Open Pairs', value: String(openPairsCount) },
    { label: 'Portfolio Names (Strategy B)', value: String(portfolioNames) },
    { label: 'Pair Candidates', value: String(state.pair_candidates.length) },
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cards.length}, 1fr)`, gap: 12 }}>
      {cards.map((c) => (
        <div className="panel" key={c.label}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', textTransform: 'uppercase', marginBottom: 6 }}>{c.label}</div>
          <div style={{ fontSize: 20, fontWeight: 700 }}>{c.value}</div>
        </div>
      ))}
    </div>
  )
}
