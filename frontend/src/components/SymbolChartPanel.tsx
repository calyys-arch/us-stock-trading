import SymbolChartSlot, { type ChartJump } from './SymbolChartSlot'

export interface MultiChartJump extends ChartJump {
  slot: 0 | 1
}

interface Props {
  symbols?: string[]
  /** Routes a jump to one of the two slots (e.g. PositionsPanel's "1"/"2"
   * Chart buttons). */
  jumpTo?: MultiChartJump | null
}

/** Two independent chart slots side by side — the system may trade several
 * stocks at once, so a single chart isn't enough to watch two positions (or
 * compare two candidates) at the same time. */
export default function SymbolChartPanel({ symbols = [], jumpTo }: Props) {
  const defaultA = symbols[0] ?? 'AAPL'
  const defaultB = symbols[1] ?? 'MSFT'
  const jumpA = jumpTo?.slot === 0 ? jumpTo : null
  const jumpB = jumpTo?.slot === 1 ? jumpTo : null

  return (
    <div className="chart-grid">
      <SymbolChartSlot defaultSymbol={defaultA} symbols={symbols} jumpTo={jumpA} />
      <div className="chart-grid-divider">
        <SymbolChartSlot defaultSymbol={defaultB} symbols={symbols} jumpTo={jumpB} />
      </div>
    </div>
  )
}
