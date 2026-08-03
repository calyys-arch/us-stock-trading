import type { DashboardStateDto } from '../types'

interface Props {
  state: DashboardStateDto
}

/** Compact label/value rows, styled like IBKR TWS's Account window. */
export default function StatCards({ state }: Props) {
  const netLiq = state.account_summary?.NetLiquidation
  const buyingPower = state.account_summary?.BuyingPower
  const openPairsCount = state.open_pairs.length
  const portfolioNames = Object.keys(state.latest_portfolio_weights || {}).length

  const rows = [
    { label: 'Net Liquidation', value: netLiq !== undefined ? `$${netLiq.toLocaleString()}` : '—' },
    { label: 'Buying Power', value: buyingPower !== undefined ? `$${buyingPower.toLocaleString()}` : '—' },
    { label: 'Open Pairs', value: String(openPairsCount) },
    { label: 'Portfolio Names (Strategy B)', value: String(portfolioNames) },
    { label: 'Pair Candidates', value: String(state.pair_candidates.length) },
  ]

  return (
    <div>
      {rows.map((r, i) => (
        <div
          key={r.label}
          style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
            padding: '7px 2px', borderBottom: i < rows.length - 1 ? '1px solid var(--border)' : 'none',
          }}
        >
          <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>{r.label}</span>
          <span className="num" style={{ fontSize: 14, fontWeight: 700 }}>{r.value}</span>
        </div>
      ))}
    </div>
  )
}
