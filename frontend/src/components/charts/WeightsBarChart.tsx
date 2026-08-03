import type { CSSProperties } from 'react'

interface Item {
  label: string
  value: number
}

interface Props {
  items: Item[]
  rowHeight?: number
}

/** Horizontal diverging bar chart centered at zero — green bars for long
 * weights, red for short — matching PortfolioWeightsTable's existing
 * positive/negative color convention. */
export default function WeightsBarChart({ items, rowHeight = 18 }: Props) {
  if (items.length === 0) return null
  const maxAbs = Math.max(...items.map((i) => Math.abs(i.value)), 0.0001)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5, marginBottom: 14 }}>
      {items.map((item) => {
        const pct = (Math.abs(item.value) / maxAbs) * 50
        const positive = item.value >= 0
        const barStyle: CSSProperties = {
          position: 'absolute',
          top: 0,
          bottom: 0,
          width: `${pct}%`,
          background: positive ? 'var(--green)' : 'var(--red)',
          borderRadius: 2,
          ...(positive ? { left: '50%' } : { right: '50%' }),
        }
        return (
          <div
            key={item.label}
            style={{ display: 'grid', gridTemplateColumns: '56px 1fr 60px', alignItems: 'center', gap: 8, height: rowHeight }}
          >
            <div className="num" style={{ fontSize: 11, color: 'var(--text-dim)', textAlign: 'right' }}>
              {item.label}
            </div>
            <div style={{ position: 'relative', height: 9, background: 'rgba(124,132,148,0.12)', borderRadius: 2 }}>
              <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: 'var(--border-light)' }} />
              <div style={barStyle} />
            </div>
            <div className={`num ${positive ? 'positive' : 'negative'}`} style={{ fontSize: 11, textAlign: 'right' }}>
              {(item.value * 100).toFixed(2)}%
            </div>
          </div>
        )
      })}
    </div>
  )
}
