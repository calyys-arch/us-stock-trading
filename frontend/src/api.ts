import type { BacktestSummary, DashboardStateDto, PairCandidate, PositionDto, RegimeReportDto, SymbolChartDto, SymbolContextDto } from './types'

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) {
    let detail = ''
    try {
      detail = (await res.json())?.detail ?? ''
    } catch {
      // response body wasn't JSON — fall through to the generic message below
    }
    throw new Error(detail || `API ${path} failed: ${res.status}`)
  }
  return res.json() as Promise<T>
}

export function getState(): Promise<DashboardStateDto> {
  return apiFetch('/api/state')
}

export function startEngine(): Promise<{ ok: boolean; running: boolean }> {
  return apiFetch('/api/engine/start', { method: 'POST' })
}

export function stopEngine(): Promise<{ ok: boolean; running: boolean }> {
  return apiFetch('/api/engine/stop', { method: 'POST' })
}

export function startAutoTrading(): Promise<{ ok: boolean; mode: string; armed_strategies: string[] }> {
  return apiFetch('/api/engine/auto/start', { method: 'POST' })
}

export function stopAutoTrading(): Promise<{ ok: boolean; mode: string }> {
  return apiFetch('/api/engine/auto/stop', { method: 'POST' })
}

export function flattenAllPositions(): Promise<{ ok: boolean; closed: unknown[] }> {
  return apiFetch('/api/positions/flatten_all', { method: 'POST' })
}

export function getPositions(): Promise<{ positions: PositionDto[] }> {
  return apiFetch('/api/positions')
}

export function flattenPosition(code: string): Promise<{ ok: boolean; closed: unknown }> {
  return apiFetch(`/api/positions/${encodeURIComponent(code.trim().toUpperCase())}/flatten`, { method: 'POST' })
}

export function runDemoBacktest(): Promise<BacktestSummary> {
  return apiFetch('/api/backtest/run', { method: 'POST' })
}

export function runPairsDemoScan(): Promise<{ pair_candidates: PairCandidate[] }> {
  return apiFetch('/api/pairs/demo_scan', { method: 'POST' })
}

export function getSymbolChart(symbol: string, days = 180, interval: '1d' | '1m' = '1d'): Promise<SymbolChartDto> {
  return apiFetch(`/api/chart/${encodeURIComponent(symbol.trim().toUpperCase())}?days=${days}&interval=${interval}`)
}

/** Microstructure context (VWAP+bands, liquidity levels, volume profile,
 * opening range) for one cached 1-minute session — dashboard/app.py's
 * /api/chart/{symbol}/context, python/microstructure/context.py.
 * `date` omitted -> most recent cached session. */
export function getSymbolContext(symbol: string, date?: string): Promise<SymbolContextDto> {
  const qs = date ? `?date=${encodeURIComponent(date)}` : ''
  return apiFetch(`/api/chart/${encodeURIComponent(symbol.trim().toUpperCase())}/context${qs}`)
}

/** Report-only Markov regime diagnostic — dashboard/app.py's
 * /api/regime/{symbol}, python/analytics/regime.py. NOT wired to any
 * strategy/order; see that module's docstring for the honesty contract. */
export function getSymbolRegime(symbol: string, opts?: { years?: number; window?: number; threshold?: number }): Promise<RegimeReportDto> {
  const params = new URLSearchParams()
  if (opts?.years) params.set('years', String(opts.years))
  if (opts?.window) params.set('window', String(opts.window))
  if (opts?.threshold) params.set('threshold', String(opts.threshold))
  const qs = params.toString() ? `?${params.toString()}` : ''
  return apiFetch(`/api/regime/${encodeURIComponent(symbol.trim().toUpperCase())}${qs}`)
}
