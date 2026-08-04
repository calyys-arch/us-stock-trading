import { useEffect, useRef } from 'react'
import {
  CandlestickSeries,
  ColorType,
  HistogramSeries,
  LineSeries,
  LineStyle,
  createChart,
  createSeriesMarkers,
  type ISeriesApi,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts'

export type ChartInterval = '1d' | '1m' | '5m' | '15m'

export interface Bar {
  date: string // '1d' -> 'YYYY-MM-DD'; intraday (1m/5m/15m) -> ISO timestamp
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

export interface VwapPoint {
  time: string
  vwap: number
  upper_1: number
  lower_1: number
}

export interface PriceLevel {
  price: number
  label: string
  color: string
}

export interface SignalMarker {
  time: string
  direction: 'long' | 'short'
  label?: string
}

interface Props {
  data: Bar[]
  height?: number
  /** '1d' (default): `date` is a 'YYYY-MM-DD' business-day string, no
   * intraday time axis. '1m'/'5m'/'15m': `date` is an ISO timestamp,
   * rendered with a time-visible axis — python/data/intraday_cache.py /
   * dashboard/app.py's /api/chart/{symbol}?interval=1m|5m|15m (5m/15m are
   * resampled server-side from the 1-minute cache). */
  interval?: ChartInterval
  /** Simple moving-average periods to overlay on the price series, e.g.
   * [5, 10]. Pass an empty array to hide MAs entirely. */
  maPeriods?: number[]
  /** Session VWAP + 1-sigma band — dashboard/app.py's
   * /api/chart/{symbol}/context ("vwap"), python/microstructure/context.py. */
  vwap?: VwapPoint[]
  /** Horizontal reference levels (YDH/YDL, PMH/PML, round numbers, equal
   * highs/lows) from the same /context endpoint's "liquidity" block. */
  levels?: PriceLevel[]
  /** Opening-range high/low, drawn as a pair of dashed price lines. */
  openingRange?: { high: number | null; low: number | null } | null
  /** Signal entries/exits (report-only diagnostic markers — see
   * docs/microstructure_pivot_plan.md; these never drive live orders). */
  markers?: SignalMarker[]
}

const MA_COLORS = ['#e0a72e', '#4c8bf5', '#c084fc']
// 5/10-day (not the classic 20/50-day trend-following pair): this system's
// two live strategies both trade on short-horizon mean reversion, not
// multi-week trend (CrossSectionalMeanReversionStrategy's Chan eq. 3.7 uses
// a 1-day lookback per configs/strategy.yaml — see that file's
// xsection_mean_reversion.lookback_days; PairsTradingStrategy trades a
// spread z-score around a cointegration-estimated mean, not either leg's
// own price trend — see PairsPanel's ZScoreChart for that). A 1-bar "moving
// average" isn't meaningful to plot (it's just the price itself), so 5/10
// is a pragmatic week-scale compromise: still purely a visual convenience
// on this panel (SymbolChartSlot is explicitly independent of the strategy
// engine — no strategy reads these lines), but it no longer sends the
// literally-opposite visual cue that a 20/50-day trend overlay would (this
// system deliberately BUYS short-term underperformers expecting reversion,
// which a "price below MA20 = bearish" read contradicts).
const DEFAULT_MA_PERIODS = [5, 10]

function toChartTime(dateStr: string, interval: ChartInterval): Time {
  if (interval !== '1d') {
    return (Math.floor(new Date(dateStr).getTime() / 1000) as UTCTimestamp) as Time
  }
  return dateStr as Time
}

/** Simple moving average; leading points (before `period` bars exist) are
 * omitted rather than plotted as 0/NaN. */
function sma(data: Bar[], period: number, interval: ChartInterval): { time: Time; value: number }[] {
  const out: { time: Time; value: number }[] = []
  let sum = 0
  for (let i = 0; i < data.length; i++) {
    sum += data[i].close
    if (i >= period) sum -= data[i - period].close
    if (i >= period - 1) out.push({ time: toChartTime(data[i].date, interval), value: sum / period })
  }
  return out
}

/** TradingView lightweight-charts candlestick + volume chart for the Symbol
 * Chart panel's real OHLCV bars (see /api/chart/{symbol} in dashboard/app.py).
 * Self-hosted, no network calls to tradingview.com — this is TradingView's
 * open-source charting engine (MIT), not their hosted widget. Optionally
 * overlays microstructure context (VWAP/bands, liquidity levels, opening
 * range, signal markers) when interval='1m' — all purely visual, report-
 * only diagnostics with no path back into the live strategy engine. */
export default function CandlestickChart({
  data,
  height = 260,
  interval = '1d',
  maPeriods = DEFAULT_MA_PERIODS,
  vwap = [],
  levels = [],
  openingRange = null,
  markers = [],
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = containerRef.current
    if (!el || data.length === 0) return

    const chart = createChart(el, {
      layout: {
        background: { type: ColorType.Solid, color: '#12151b' },
        textColor: '#7c8494',
        fontSize: 11,
        fontFamily: "'SF Mono', 'Menlo', monospace",
      },
      grid: {
        vertLines: { color: 'rgba(38, 43, 52, 0.5)' },
        horzLines: { color: 'rgba(38, 43, 52, 0.5)' },
      },
      rightPriceScale: { borderColor: '#262b34' },
      timeScale: { borderColor: '#262b34', timeVisible: interval !== '1d', secondsVisible: false },
      autoSize: true,
    })

    const candleSeries: ISeriesApi<'Candlestick'> = chart.addSeries(CandlestickSeries, {
      upColor: '#1fae67',
      downColor: '#e5484d',
      borderVisible: false,
      wickUpColor: '#1fae67',
      wickDownColor: '#e5484d',
    })
    candleSeries.priceScale().applyOptions({ scaleMargins: { top: 0.1, bottom: 0.3 } })

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: '',
    })
    volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.75, bottom: 0 } })

    candleSeries.setData(
      data.map((b) => ({ time: toChartTime(b.date, interval), open: b.open, high: b.high, low: b.low, close: b.close })),
    )
    volumeSeries.setData(
      data.map((b) => ({
        time: toChartTime(b.date, interval),
        value: b.volume ?? 0,
        color: b.close >= b.open ? 'rgba(31, 174, 103, 0.5)' : 'rgba(229, 72, 77, 0.5)',
      })),
    )

    maPeriods.forEach((period, i) => {
      const points = sma(data, period, interval)
      if (points.length === 0) return
      const maSeries = chart.addSeries(LineSeries, {
        color: MA_COLORS[i % MA_COLORS.length],
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      })
      maSeries.setData(points)
    })

    if (vwap.length > 0) {
      const vwapSeries = chart.addSeries(LineSeries, {
        color: '#f5b83d', lineWidth: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
      })
      vwapSeries.setData(vwap.map((p) => ({ time: toChartTime(p.time, interval), value: p.vwap })))

      const bandStyle = { color: 'rgba(245, 184, 61, 0.45)', lineWidth: 1 as const, lineStyle: LineStyle.Dashed, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false }
      const upperSeries = chart.addSeries(LineSeries, bandStyle)
      upperSeries.setData(vwap.map((p) => ({ time: toChartTime(p.time, interval), value: p.upper_1 })))
      const lowerSeries = chart.addSeries(LineSeries, bandStyle)
      lowerSeries.setData(vwap.map((p) => ({ time: toChartTime(p.time, interval), value: p.lower_1 })))
    }

    const priceLines = levels.map((lvl) =>
      candleSeries.createPriceLine({
        price: lvl.price,
        color: lvl.color,
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: true,
        title: lvl.label,
      }),
    )
    if (openingRange?.high != null) {
      priceLines.push(candleSeries.createPriceLine({
        price: openingRange.high, color: '#8b93a5', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: 'OR High',
      }))
    }
    if (openingRange?.low != null) {
      priceLines.push(candleSeries.createPriceLine({
        price: openingRange.low, color: '#8b93a5', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: 'OR Low',
      }))
    }

    let markersPlugin: ReturnType<typeof createSeriesMarkers<Time>> | null = null
    if (markers.length > 0) {
      // No per-marker text label: scan_signals_for_session (see its
      // docstring) deliberately does NOT skip bars while "in a position",
      // so a busy session can legitimately report a signal on a large
      // fraction of bars — repeating "sweep reclaim"-style text under
      // every single arrow turns into an unreadable wall of overlapping
      // text well before the arrows themselves get too dense to read.
      // The per-strategy signal counts are already surfaced as plain text
      // above the chart (see SymbolChartSlot); hovering a marker's tooltip
      // (lightweight-charts default) still shows the exact bar/price.
      const seriesMarkers: SeriesMarker<Time>[] = markers.map((m) => ({
        time: toChartTime(m.time, interval),
        position: m.direction === 'long' ? 'belowBar' : 'aboveBar',
        color: m.direction === 'long' ? '#1fae67' : '#e5484d',
        shape: m.direction === 'long' ? 'arrowUp' : 'arrowDown',
        size: 0.6,
      }))
      markersPlugin = createSeriesMarkers(candleSeries, seriesMarkers)
    }

    chart.timeScale().fitContent()

    return () => {
      markersPlugin?.detach()
      priceLines.forEach((pl) => candleSeries.removePriceLine(pl))
      chart.remove()
    }
  }, [data, interval, maPeriods, vwap, levels, openingRange, markers])

  if (!data || data.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-faint)', fontSize: 11 }}>
        No data
      </div>
    )
  }

  return (
    <div style={{ position: 'relative' }}>
      {maPeriods.length > 0 && (
        <div style={{ position: 'absolute', top: 2, left: 4, zIndex: 1, display: 'flex', gap: 8 }}>
          {maPeriods.map((period, i) => (
            <span key={period} style={{ fontSize: 9, color: MA_COLORS[i % MA_COLORS.length] }}>
              MA{period}
            </span>
          ))}
          {vwap.length > 0 && <span style={{ fontSize: 9, color: '#f5b83d' }}>VWAP</span>}
        </div>
      )}
      <div ref={containerRef} style={{ width: '100%', height }} />
    </div>
  )
}
