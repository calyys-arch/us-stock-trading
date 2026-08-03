import { useEffect, useRef } from 'react'
import { AreaSeries, ColorType, createChart } from 'lightweight-charts'

interface Props {
  data: number[]
  height?: number
  color?: 'auto' | string
}

/** The demo equity curve (dashboard/app.py run_backtest_demo) is an
 * index-normalized series with no calendar dates of its own — synthesize
 * sequential business days so lightweight-charts has valid ascending time
 * values (the axis itself carries no real meaning here, only the shape). */
function syntheticBusinessDays(n: number): string[] {
  const dates: string[] = []
  const d = new Date('2024-01-02T00:00:00Z')
  while (dates.length < n) {
    const day = d.getUTCDay()
    if (day !== 0 && day !== 6) dates.push(d.toISOString().slice(0, 10))
    d.setUTCDate(d.getUTCDate() + 1)
  }
  return dates
}

/** TradingView lightweight-charts area chart for the backtest equity curve. */
export default function EquityAreaChart({ data, height = 140, color = 'auto' }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = containerRef.current
    if (!el || !data || data.length < 2) return

    const up = data[data.length - 1] >= data[0]
    const lineColor = color === 'auto' ? (up ? '#1fae67' : '#e5484d') : color

    const chart = createChart(el, {
      layout: { background: { type: ColorType.Solid, color: '#12151b' }, textColor: '#7c8494', fontSize: 11 },
      grid: { vertLines: { visible: false }, horzLines: { color: 'rgba(38, 43, 52, 0.5)' } },
      rightPriceScale: { borderColor: '#262b34' },
      timeScale: { borderColor: '#262b34', timeVisible: false },
      handleScroll: false,
      handleScale: false,
      autoSize: true,
    })

    const series = chart.addSeries(AreaSeries, {
      lineColor,
      topColor: `${lineColor}44`,
      bottomColor: `${lineColor}00`,
      lineWidth: 2,
      priceLineVisible: false,
    })

    const dates = syntheticBusinessDays(data.length)
    series.setData(data.map((v, i) => ({ time: dates[i], value: v })))
    chart.timeScale().fitContent()

    return () => chart.remove()
  }, [data, color])

  if (!data || data.length < 2) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-faint)', fontSize: 11 }}>
        No data
      </div>
    )
  }

  return <div ref={containerRef} style={{ width: '100%', height }} />
}
