import { useEffect, useRef } from 'react'
import { ColorType, LineSeries, LineStyle, createChart, type UTCTimestamp } from 'lightweight-charts'

interface Props {
  values: number[]
  entryZ?: number
  width?: number
  height?: number
}

/** TradingView lightweight-charts sparkline for a pair's spread z-score
 * history, with dashed entry/exit threshold lines. No real dates exist for
 * this series (see dashboard/app.py demo_scan), so index positions are used
 * as the time axis directly; the axis itself is hidden. */
export default function ZScoreChart({ values, entryZ = 2, width = 160, height = 40 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = containerRef.current
    if (!el || !values || values.length < 2) return

    const last = values[values.length - 1]
    const breached = Math.abs(last) >= entryZ
    const lineColor = breached ? '#e0a83e' : '#3d7de0'

    const chart = createChart(el, {
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#565d6b' },
      grid: { vertLines: { visible: false }, horzLines: { visible: false } },
      rightPriceScale: { visible: false },
      leftPriceScale: { visible: false },
      timeScale: { visible: false },
      crosshair: {
        vertLine: { visible: false, labelVisible: false },
        horzLine: { visible: false, labelVisible: false },
      },
      handleScroll: false,
      handleScale: false,
      autoSize: true,
    })

    const series = chart.addSeries(LineSeries, {
      color: lineColor,
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    })

    series.createPriceLine({ price: entryZ, color: '#e5484d', lineStyle: LineStyle.Dashed, lineWidth: 1, axisLabelVisible: false })
    series.createPriceLine({ price: -entryZ, color: '#e5484d', lineStyle: LineStyle.Dashed, lineWidth: 1, axisLabelVisible: false })
    series.createPriceLine({ price: 0, color: '#363c48', lineStyle: LineStyle.Solid, lineWidth: 1, axisLabelVisible: false })

    series.setData(values.map((v, i) => ({ time: i as UTCTimestamp, value: v })))
    chart.timeScale().fitContent()

    return () => chart.remove()
  }, [values, entryZ])

  if (!values || values.length < 2) {
    return <span style={{ color: 'var(--text-faint)', fontSize: 10 }}>—</span>
  }

  return <div ref={containerRef} style={{ width, height }} />
}
