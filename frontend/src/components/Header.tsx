import { useState } from 'react'
import { flattenAllPositions, startAutoTrading, stopAutoTrading } from '../api'

interface Props {
  running: boolean
  mode: 'observe' | 'auto'
  dataSource: 'simulated' | 'ibkr_paper' | 'futu_live'
  ibkrBrokerConnected: boolean
  ibkrFeedConnected: boolean
  futuLiveFeedActive?: boolean
  armedStrategies?: string[]
  pairsRegimeGateOpen?: boolean
  pairsRegimeGateReason?: string
  onStart: () => void
  onStop: () => void
}

export default function Header({
  running, mode, dataSource, ibkrBrokerConnected, ibkrFeedConnected,
  futuLiveFeedActive, armedStrategies, pairsRegimeGateOpen, pairsRegimeGateReason,
  onStart, onStop,
}: Props) {
  const [autoBusy, setAutoBusy] = useState(false)
  const [flattenBusy, setFlattenBusy] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)

  const handleStartAuto = async () => {
    if (!window.confirm(
      'Start PAPER AUTO TRADING?\n\nThis arms paper order submission for the allowlisted '
      + 'strategies only (absorption_breakout + regime-gated pairs_trading). Retired '
      + 'microstructure signals stay off. Qualified signals go to SimBroker — no per-trade click.\n\n'
      + `Data: ${dataSource}. This is a forward experiment, NOT a WFO GO promotion.`,
    )) return
    setAutoBusy(true)
    setActionError(null)
    try {
      await startAutoTrading()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err))
    } finally {
      setAutoBusy(false)
    }
  }

  const handleStopAuto = async () => {
    setAutoBusy(true)
    setActionError(null)
    try {
      await stopAutoTrading()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err))
    } finally {
      setAutoBusy(false)
    }
  }

  const handleFlattenAll = async () => {
    if (!window.confirm(
      'EXIT ALL POSITIONS?\n\nThis immediately market-closes every open position at the broker, '
      + 'right now, regardless of strategy or time of day. This cannot be undone.',
    )) return
    setFlattenBusy(true)
    setActionError(null)
    try {
      const result = await flattenAllPositions()
      window.alert(result.closed.length > 0 ? `Closed ${result.closed.length} position(s).` : 'No open positions to close.')
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err))
    } finally {
      setFlattenBusy(false)
    }
  }

  const pillClass = !running ? 'stopped' : mode === 'auto' ? 'auto' : 'observe'
  const pillLabel = !running ? 'STOPPED' : mode === 'auto' ? 'AUTO-EXECUTE' : 'OBSERVE ONLY'
  const ledColor = !running ? 'var(--text-faint)' : mode === 'auto' ? 'var(--green)' : 'var(--amber)'

  const isIbkr = dataSource === 'ibkr_paper'
  const isFutu = dataSource === 'futu_live'
  // Broker and feed are two independent IBKR socket connections (different
  // client IDs) — a broker-connected-but-feed-pending state is normal right
  // after Start or outside market hours (no ticks flowing yet), and must
  // not be shown as a scary "disconnected" red badge alongside a genuinely
  // dead connection.
  let dataSourceClass: string = 'observe'
  let dataSourceLabel = 'SIMULATED — no live feed'
  if (isFutu) {
    if (!running) {
      dataSourceClass = 'stopped'
      dataSourceLabel = 'FUTU LIVE — paper (not started)'
    } else if (!futuLiveFeedActive) {
      dataSourceClass = 'observe'
      dataSourceLabel = 'FUTU LIVE — paper, feed pending'
    } else {
      dataSourceClass = 'auto'
      dataSourceLabel = 'FUTU LIVE — paper, feed active'
    }
  } else if (isIbkr) {
    if (!running) {
      dataSourceClass = 'stopped'
      dataSourceLabel = 'IBKR PAPER — not started'
    } else if (!ibkrBrokerConnected) {
      dataSourceClass = 'error'
      dataSourceLabel = 'IBKR PAPER — disconnected'
    } else if (!ibkrFeedConnected) {
      dataSourceClass = 'observe'
      dataSourceLabel = 'IBKR PAPER — broker OK, feed pending'
    } else {
      dataSourceClass = 'auto'
      dataSourceLabel = 'IBKR PAPER — connected'
    }
  }
  const armedLabel = mode === 'auto'
    ? `ARMED: ${(armedStrategies && armedStrategies.length) ? armedStrategies.join(', ') : 'none'}`
    : 'PAPER / OBSERVE'
  const pairsGateLabel = pairsRegimeGateOpen
    ? 'PAIRS GATE OPEN'
    : `PAIRS GATE CLOSED${pairsRegimeGateReason ? ` (${pairsRegimeGateReason})` : ''}`

  return (
    <div className="tws-toolbar">
      <div className="tws-brand">
        <span className="tws-logo">Q</span>
        <span className="tws-brand-text">US Equity Quant Trading</span>
        <span>
          <span className="tws-led" style={{ background: ledColor }} />
          <span className={`pill ${pillClass}`}>{pillLabel}</span>
        </span>
        <span className={`pill ${dataSourceClass}`} title="Set in configs/broker.yaml — not a UI toggle">
          {dataSourceLabel}
        </span>
        <span className={`pill ${mode === 'auto' ? 'auto' : 'observe'}`} title="Allowlisted paper strategies only">
          {armedLabel}
        </span>
        <span className={`pill ${pairsRegimeGateOpen ? 'auto' : 'stopped'}`} title="trend_efficiency_gate on SPY — pairs entries only when mean-reversion-friendly">
          {pairsGateLabel}
        </span>
      </div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        {actionError && (
          <span style={{ color: 'var(--red)', fontSize: 11, maxWidth: 260 }} title={actionError}>{actionError}</span>
        )}
        {!running ? (
          <button className="tws-btn primary" onClick={onStart}>▶ Start (paper)</button>
        ) : (
          <button className="tws-btn danger" onClick={onStop}>■ Stop</button>
        )}
        {mode === 'auto' ? (
          <button
            className="tws-btn"
            style={{ borderColor: 'var(--green)', color: 'var(--green)' }}
            onClick={handleStopAuto}
            disabled={autoBusy}
          >
            ● Auto Trading ON — Disarm
          </button>
        ) : (
          <button
            className="tws-btn"
            style={{ borderColor: 'var(--amber)', color: 'var(--amber)' }}
            onClick={handleStartAuto}
            disabled={!running || autoBusy}
            title={!running ? 'Start the engine (paper) first' : undefined}
          >
            ⚡ Start Auto Trading
          </button>
        )}
        <button className="tws-btn danger" onClick={handleFlattenAll} disabled={flattenBusy}>
          ✕ Exit All Positions
        </button>
      </div>
    </div>
  )
}
