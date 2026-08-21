import { useEffect, useState } from 'react'
import type { ChartInterval } from '../api'
import { getSymbolChart, getSymbolContext } from '../api'
import type { SymbolChartDto, SymbolContextDto } from '../types'
import CandlestickChart, { type Bar, type PriceLevel, type SignalMarker } from './charts/CandlestickChart'

const DAILY_RANGE_OPTIONS: { label: string; days: number }[] = [
  { label: '3D', days: 3 },
  { label: '1M', days: 30 },
  { label: '3M', days: 90 },
  { label: '6M', days: 180 },
  { label: '1Y', days: 365 },
]

// 1-minute bars are ~390/session (RTH) — a handful of days is already
// dense on screen, and dashboard/app.py's /api/chart caps intraday
// intervals at 60 calendar days server-side anyway. 5-Min/15-Min reuse the
// same day ranges (they're resampled from the same 1-minute cache), just
// with far fewer bars per session so the same window looks less crowded.
const INTRADAY_RANGE_OPTIONS: { label: string; days: number }[] = [
  { label: '1D', days: 2 },
  { label: '3D', days: 4 },
  { label: '5D', days: 7 },
  { label: '10D', days: 14 },
]

const TIMEFRAME_OPTIONS: { value: ChartInterval; label: string }[] = [
  { value: '1d', label: 'Daily' },
  { value: '1m', label: '1-Min' },
  { value: '5m', label: '5-Min' },
  { value: '15m', label: '15-Min' },
]

const LEVEL_COLORS: Record<string, string> = {
  ydh: '#4c8bf5', ydl: '#4c8bf5', pmh: '#c084fc', pml: '#c084fc',
  eq: 'rgba(224, 167, 46, 0.7)', round: 'rgba(124, 132, 148, 0.5)',
}

function buildLevels(context: SymbolContextDto | null): PriceLevel[] {
  if (!context) return []
  const levels: PriceLevel[] = []
  const { liquidity } = context
  if (liquidity.ydh != null) levels.push({ price: liquidity.ydh, label: 'YDH', color: LEVEL_COLORS.ydh })
  if (liquidity.ydl != null) levels.push({ price: liquidity.ydl, label: 'YDL', color: LEVEL_COLORS.ydl })
  if (liquidity.pmh != null) levels.push({ price: liquidity.pmh, label: 'PMH', color: LEVEL_COLORS.pmh })
  if (liquidity.pml != null) levels.push({ price: liquidity.pml, label: 'PML', color: LEVEL_COLORS.pml })
  liquidity.eq_highs.forEach((p, i) => levels.push({ price: p, label: `EQH${i + 1}`, color: LEVEL_COLORS.eq }))
  liquidity.eq_lows.forEach((p, i) => levels.push({ price: p, label: `EQL${i + 1}`, color: LEVEL_COLORS.eq }))
  liquidity.round_levels.forEach((p) => levels.push({ price: p, label: '', color: LEVEL_COLORS.round }))
  return levels
}

function buildMarkers(context: SymbolContextDto | null): SignalMarker[] {
  if (!context) return []
  return context.signals.map((s) => ({
    time: s.time,
    direction: s.direction,
    label: s.strategy.replace('_', ' '),
  }))
}

export interface ChartJump {
  symbol: string
  nonce: number
}

interface Props {
  defaultSymbol: string
  /** Live-traded symbol universe (DashboardState.symbols) — rendered as
   * quick-select chips so switching between several actively-traded stocks
   * doesn't require retyping a ticker each time. */
  symbols?: string[]
  /** Set by a sibling panel (e.g. PositionsPanel's per-slot Chart buttons)
   * to jump this slot to a specific symbol. `nonce` must change on every
   * jump (even to the same symbol) so the effect re-fires. */
  jumpTo?: ChartJump | null
}

/** One independent ticker/chart slot — a UI convenience for eyeballing any
 * US equity, independent of the strategy engine. Daily mode pulls real
 * daily bars via dashboard/app.py's /api/chart/{symbol} (IBKR-first,
 * yfinance-fallback, on-disk cache). 1-Min/5-Min/15-Min modes pull cached
 * 1-minute bars (?interval=1m|5m|15m, data/history_1m/ — built by
 * scripts/backfill_intraday.py, no live IB fetch from this endpoint; 5-Min
 * and 15-Min are resampled server-side from the same 1-minute cache) plus
 * /api/chart/{symbol}/context for the VWAP/liquidity-level/opening-range/
 * signal-marker overlays (python/microstructure/*, report-only diagnostics
 * — see docs/microstructure_pivot_plan.md). SymbolChartPanel renders two
 * of these side by side so two different stocks can be watched at once. */
export default function SymbolChartSlot({ defaultSymbol, symbols = [], jumpTo }: Props) {
  const [input, setInput] = useState(defaultSymbol)
  const [interval, setInterval_] = useState<ChartInterval>('5m')
  const [days, setDays] = useState(INTRADAY_RANGE_OPTIONS[1].days)
  const [data, setData] = useState<SymbolChartDto | null>(null)
  const [context, setContext] = useState<SymbolContextDto | null>(null)
  const [contextError, setContextError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Defaults to OFF: scan_signals_for_session deliberately does not skip
  // bars while "in a position" (see its docstring), so a busy session can
  // legitimately report a signal on most bars — dense enough by design
  // that it's better shown on demand than always cluttering the chart.
  const [showSignals, setShowSignals] = useState(false)

  const isIntraday = interval !== '1d'

  const load = async (symbol: string, rangeDays: number, tf: ChartInterval, contextDate?: string) => {
    const trimmed = symbol.trim().toUpperCase()
    if (!trimmed) return
    setInput(trimmed)
    setLoading(true)
    setError(null)
    setContextError(null)
    try {
      const result = await getSymbolChart(trimmed, rangeDays, tf)
      setData(result)
      if (tf !== '1d') {
        try {
          setContext(await getSymbolContext(trimmed, contextDate))
        } catch (ctxErr) {
          setContext(null)
          setContextError(ctxErr instanceof Error ? ctxErr.message : String(ctxErr))
        }
      } else {
        setContext(null)
      }
    } catch (err) {
      setData(null)
      setContext(null)
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (jumpTo?.symbol) load(jumpTo.symbol, days, interval)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only re-jump when the jump itself changes
  }, [jumpTo])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    load(input, days, interval)
  }

  const handleRangeClick = (rangeDays: number) => {
    setDays(rangeDays)
    if (data) load(data.symbol, rangeDays, interval)
  }

  const handleTimeframeClick = (tf: ChartInterval) => {
    if (tf === interval) return
    const defaultDays = tf !== '1d' ? INTRADAY_RANGE_OPTIONS[1].days : 180
    setInterval_(tf)
    setDays(defaultDays)
    if (data) load(data.symbol, defaultDays, tf)
  }

  const handleContextDateClick = (date: string) => {
    if (data) load(data.symbol, days, interval, date)
  }

  const rangeOptions = isIntraday ? INTRADAY_RANGE_OPTIONS : DAILY_RANGE_OPTIONS

  const first = data?.close[0]
  const last = data?.close[data.close.length - 1]
  const changePct = first && last ? ((last - first) / first) * 100 : null
  const periodHigh = data ? Math.max(...data.high) : null
  const periodLow = data ? Math.min(...data.low) : null
  const bars: Bar[] = data
    ? data.dates.map((date, i) => ({
        date,
        open: data.open[i],
        high: data.high[i],
        low: data.low[i],
        close: data.close[i],
        volume: data.volume[i],
      }))
    : []

  const vwapPoints = context
    ? context.vwap.dates.map((t, i) => ({
        time: t, vwap: context.vwap.vwap[i], upper_1: context.vwap.upper_1[i], lower_1: context.vwap.lower_1[i],
      }))
    : []

  return (
    <div>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8, alignItems: 'center' }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value.toUpperCase())}
          placeholder="TICKER"
          maxLength={6}
          className="num"
          style={{
            background: 'var(--bg)', border: '1px solid var(--border-light)', borderRadius: 2,
            color: 'var(--text)', padding: '5px 8px', fontSize: 12, width: 90, textTransform: 'uppercase',
          }}
        />
        <button type="submit" className="tws-btn primary" style={{ fontSize: 11, padding: '5px 10px' }} disabled={loading}>
          {loading ? '…' : 'Load'}
        </button>
        <div style={{ display: 'flex', gap: 2 }}>
          {TIMEFRAME_OPTIONS.map(({ value, label }) => (
            <button
              key={value}
              type="button"
              onClick={() => handleTimeframeClick(value)}
              className="tws-btn"
              style={{
                fontSize: 10, padding: '4px 6px',
                borderColor: interval === value ? 'var(--accent)' : 'var(--border-light)',
                color: interval === value ? 'var(--accent)' : 'var(--text-dim)',
              }}
            >
              {label}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 2 }}>
          {rangeOptions.map((r) => (
            <button
              key={r.label}
              type="button"
              onClick={() => handleRangeClick(r.days)}
              className="tws-btn"
              style={{
                fontSize: 10, padding: '4px 6px',
                borderColor: days === r.days ? 'var(--accent)' : 'var(--border-light)',
                color: days === r.days ? 'var(--accent)' : 'var(--text-dim)',
              }}
            >
              {r.label}
            </button>
          ))}
        </div>
      </form>

      {symbols.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 8 }}>
          {symbols.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => load(s, days, interval)}
              className="tws-btn"
              style={{
                fontSize: 10, padding: '3px 7px',
                borderColor: data?.symbol === s ? 'var(--accent)' : 'var(--border-light)',
                color: data?.symbol === s ? 'var(--accent)' : 'var(--text-dim)',
              }}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {error && (
        <div style={{ color: 'var(--red)', fontSize: 12, marginBottom: 8 }}>{error}</div>
      )}

      {!data && !error && !loading && (
        <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>
          Type a US equity ticker and click Load. Daily = real daily bars (IBKR falls back to
          yfinance). 1-Min/5-Min/15-Min = cached microstructure bars, 5-Min and 15-Min resampled
          from the same cache (run scripts/backfill_intraday.py first).
        </div>
      )}

      {data && (
        <div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 8, alignItems: 'baseline' }}>
            <span className="num" style={{ fontSize: 14, fontWeight: 700 }}>{data.symbol}</span>
            {last !== undefined && <span className="num" style={{ fontSize: 14 }}>${last.toFixed(2)}</span>}
            {changePct !== null && (
              <span className={`num ${changePct >= 0 ? 'positive' : 'negative'}`} style={{ fontSize: 11 }}>
                {changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}% ({days}d)
              </span>
            )}
            {periodHigh !== null && periodLow !== null && (
              <span style={{ fontSize: 10, color: 'var(--text-dim)' }}>
                <span className="num">${periodLow.toFixed(2)}</span>–<span className="num">${periodHigh.toFixed(2)}</span>
              </span>
            )}
            <span style={{ fontSize: 9, color: 'var(--text-faint)', textTransform: 'uppercase', marginLeft: 'auto' }}>
              {data.source}
              {data.quality_flagged && <span style={{ color: 'var(--amber)' }}> · flagged</span>}
            </span>
          </div>

          {isIntraday && context && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 8, alignItems: 'center' }}>
              <span style={{ fontSize: 9, color: 'var(--text-faint)', textTransform: 'uppercase' }}>Session:</span>
              {context.available_dates.slice(-8).map((d) => (
                <button
                  key={d}
                  type="button"
                  onClick={() => handleContextDateClick(d)}
                  className="tws-btn"
                  style={{
                    fontSize: 9, padding: '2px 5px',
                    borderColor: context.date === d ? 'var(--accent)' : 'var(--border-light)',
                    color: context.date === d ? 'var(--accent)' : 'var(--text-dim)',
                  }}
                >
                  {d}
                </button>
              ))}
              {context.signals.length > 0 && (
                <>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: 9, color: 'var(--text-dim)', marginLeft: 4, cursor: 'pointer' }}>
                    <input type="checkbox" checked={showSignals} onChange={(e) => setShowSignals(e.target.checked)} style={{ margin: 0 }} />
                    Show markers
                  </label>
                  <span style={{ fontSize: 9, color: 'var(--text-faint)' }}>
                    {context.signals.length} signal(s) detected — report-only, does not skip bars while "in a
                    position" (see scan_signals_for_session), so a busy session can look dense; see
                    docs/microstructure_pivot_plan.md
                  </span>
                </>
              )}
            </div>
          )}
          {isIntraday && contextError && (
            <div style={{ color: 'var(--amber)', fontSize: 10, marginBottom: 8 }}>
              Context unavailable: {contextError}
            </div>
          )}

          <CandlestickChart
            data={bars}
            height={230}
            interval={interval}
            maPeriods={isIntraday ? [] : undefined}
            vwap={vwapPoints}
            levels={buildLevels(context)}
            openingRange={context?.opening_range ?? null}
            markers={showSignals ? buildMarkers(context) : []}
          />
        </div>
      )}
    </div>
  )
}
