import type { DashboardStateDto } from '../types'

interface Props {
  state: DashboardStateDto
}

/** Bottom status bar, styled after TWS's connection/clock strip. */
export default function StatusBar({ state }: Props) {
  const ledColor = state.running ? 'var(--green)' : 'var(--text-faint)'
  const serverTime = state.server_time ? new Date(state.server_time).toLocaleTimeString() : '—'

  return (
    <div className="tws-statusbar">
      <div className="tws-statusbar-left">
        <span><span className="tws-led" style={{ background: ledColor }} />{state.running ? 'Engine running' : 'Engine stopped'}</span>
        <span>Mode: {state.mode.toUpperCase()}</span>
        <span>
          Data: {state.data_source === 'ibkr_paper'
            ? `IBKR PAPER (broker ${state.ibkr_broker_connected ? 'connected' : 'disconnected'}, feed ${state.ibkr_feed_connected ? 'connected' : 'pending/no ticks yet'})`
            : state.data_source === 'futu_live'
              ? `FUTU LIVE paper (feed ${state.futu_live_feed_active ? 'active' : 'pending/no ticks yet'})`
              : 'SIMULATED'}
        </span>
        <span>Armed: {(state.armed_strategies && state.armed_strategies.length) ? state.armed_strategies.join(', ') : 'none'}</span>
        <span>Pairs gate: {state.pairs_regime_gate_open ? 'OPEN' : `CLOSED (${state.pairs_regime_gate_reason || 'n/a'})`}</span>
        <span>Live gates: {state.live_gate_regime || 'undecided'}{state.live_gate_policy?.strategies?.length
          ? ` — ${state.live_gate_policy.strategies.map(s => `${s.strategy}:${s.allowed ? 'ON' : 'OFF'}(${s.n_gates ?? s.required?.length ?? 0})`).join(' ')}`
          : ''}</span>
        <span>Started: {state.started_at ? new Date(state.started_at).toLocaleString() : '—'}</span>
      </div>
      <div>{serverTime}</div>
    </div>
  )
}
