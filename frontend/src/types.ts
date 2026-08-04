export interface DashboardStateDto {
  running: boolean
  mode: 'observe' | 'auto'
  started_at: string | null
  data_source: 'simulated' | 'ibkr_paper'
  ibkr_broker_connected: boolean
  ibkr_feed_connected: boolean
  symbols: string[]
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
  zscore_history?: number[]
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

export interface SymbolChartDto {
  symbol: string
  interval: '1d' | '1m' | '5m' | '15m'
  dates: string[]
  open: number[]
  high: number[]
  low: number[]
  close: number[]
  volume: number[]
  source: string
  quality_flagged: boolean
}

export interface SymbolContextDto {
  symbol: string
  date: string
  vwap: {
    dates: string[]
    vwap: number[]
    upper_1: number[]
    lower_1: number[]
    upper_2: number[]
    lower_2: number[]
  }
  liquidity: {
    ydh: number | null
    ydl: number | null
    pmh: number | null
    pml: number | null
    eq_highs: number[]
    eq_lows: number[]
    round_levels: number[]
  }
  volume_profile: {
    poc: number | null
    vah: number | null
    val: number | null
    bin_edges: number[]
    bin_volume: number[]
  }
  opening_range: {
    high: number | null
    low: number | null
    start: string | null
    end: string | null
  }
  signals: {
    strategy: 'sweep_reclaim' | 'fvg_retest' | 'orb_vwap'
    time: string
    direction: 'long' | 'short'
    entry_price: number
  }[]
  available_dates: string[]
}

/** Report-only Markov regime diagnostic — dashboard/app.py's
 * /api/regime/{symbol}, python/analytics/regime.py. NOT wired to any
 * strategy/order; `naive_backtest` is explicitly illustrative-only, see
 * that module's docstring. */
export interface RegimeReportDto {
  symbol: string
  as_of: string
  window: number
  threshold: number
  n_days_labeled: number
  current_state: 'Bear' | 'Sideways' | 'Bull'
  transition_matrix: Record<string, Record<string, number>>
  stationary_distribution: Record<string, number>
  recent_history: { date: string; state: string }[]
  naive_backtest: {
    sharpe_naive_no_cost: number
    max_drawdown_naive_no_cost: number
    n_days: number
    note: string
  } | null
}

export interface PositionDto {
  code: string
  qty: number
  side: 'long' | 'short'
}

export interface BacktestSummary {
  label?: string
  sharpe_annualized: number
  max_drawdown: number
  cagr: number
  n_days: number
  equity_curve?: number[]
}
