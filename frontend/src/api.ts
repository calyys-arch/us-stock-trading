import type { BacktestSummary, DashboardStateDto } from './types'

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status}`)
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

export function runDemoBacktest(): Promise<BacktestSummary> {
  return apiFetch('/api/backtest/run', { method: 'POST' })
}
