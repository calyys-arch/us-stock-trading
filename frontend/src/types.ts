export interface DashboardStateDto {
  running: boolean
  mode: 'observe' | 'auto'
  started_at: string | null
  open_pairs: OpenPair[]
  pair_candidates: PairCandidate[]
  latest_portfolio_target: PortfolioTarget | null
  latest_portfolio_weights: Record<string, number>
  recent_signals: SignalDto[]
  recent_execution_reports: ExecutionReportDto[]
  latest_backtest_summary: BacktestSummary | null
  account_summary: Record<string, number>
  server_time: string
}

export interface OpenPair {
  code_a: string
  code_b: string
  side: string
  entry_z: number
  entry_time: string
}

export interface PairCandidate {
  code_a: string
  code_b: string
  cadf_tstat: number
  half_life_days: number
}

export interface PortfolioTarget {
  strategy: string
  as_of: string
  weights: Record<string, number>
  metadata: Record<string, unknown>
}

export interface SignalDto {
  id: string
  strategy: string
  timestamp: string
  [key: string]: unknown
}

export interface ExecutionReportDto {
  type: string
  reason?: string
  timestamp?: string
  [key: string]: unknown
}

export interface BacktestSummary {
  label?: string
  sharpe_annualized: number
  max_drawdown: number
  cagr: number
  n_days: number
}
