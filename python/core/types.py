"""
Core data types shared across the signal/risk/execution pipeline.
All types are plain dataclasses — no I/O, no external deps (per
architecture-rules.mdc: the signal layer must be a pure function of these).

Adapted from forex-trading/python/core/types.py. Forex-specific fields
(pip_size, forex_session, forex_leverage, currency-pair helpers) are removed;
US-equity fields (sector, ADV, halt/locate flags, PDT-relevant counters) and
two NEW families of types are added because a single-instrument
RawSignal/QualifiedSignal pair cannot represent this system's two strategies:

  - SpreadSignal / QualifiedSpreadOrder — Strategy A (cointegrated pairs
    trading) trades TWO legs with a hedge ratio and a spread-based exit
    (z-score / half-life / mu), never a single-instrument stop-loss/take-profit.
  - PortfolioTarget / QualifiedPortfolioOrder — Strategy B (cross-sectional
    mean reversion) evaluates the WHOLE universe at once and emits a target
    weight vector, not a per-instrument evaluate() call.

RawSignal / QualifiedSignal are kept for architecture parity and any future
single-instrument strategy (e.g. a momentum overlay), but neither MVP
strategy uses them directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


@dataclass(slots=True)
class Tick:
    code: str
    price: float
    volume: int
    bid: float
    ask: float
    timestamp: datetime
    quote_ready: bool = True
    source: str = "live"


@dataclass(slots=True)
class Candle:
    code: str
    timeframe: str  # "1m" | "5m" | "15m" | "1d"
    open: float
    high: float
    low: float
    close: float
    volume: int
    turnover: float
    timestamp: datetime


@dataclass
class MarketSnapshot:
    # ── Core price / volume (required) ───────────────────────────────────────
    code: str
    price: float
    volume_today: int
    turnover_today: float
    vwap: float
    atr14: float
    atr5: float
    rsi14: float
    ema8: float
    ema20: float
    bb_upper: float
    bb_mid: float
    bb_lower: float
    vol_ma20: float
    vol_ratio: float
    bid_ask_spread_pct: float
    timestamp: datetime
    quote_ready: bool = True

    # ── K-line series (optional; newest last) ────────────────────────────────
    candles_1m: list = field(default_factory=list)
    candles_primary: list = field(default_factory=list)   # DataEngine primary_tf bars
    candles_by_tf: dict = field(default_factory=dict)     # keyed by interval minutes

    # ── Previous session data ─────────────────────────────────────────────────
    prev_day_high: float = 0.0
    prev_day_low: float = 0.0
    prev_day_close: float = 0.0

    # ── VWAP slope over the last 15 minutes ───────────────────────────────────
    vwap_slope_15m: float = 0.0

    # ── Level 2 order book (empty when L2 is unavailable) ────────────────────
    level2_bid_qty: list = field(default_factory=list)
    level2_ask_qty: list = field(default_factory=list)

    # ── News / calendar flags ─────────────────────────────────────────────────
    has_news_event: bool = False       # scheduled earnings/announcement today
    has_volume_spike_60m: bool = False  # any 1m bar in last 60 min > 5x vol_ma20
    rsi_ready: bool = True

    # ── Volatility Spike Guard ────────────────────────────────────────────────
    atr5_spike_ratio: float = 1.0

    # ── US-equity-specific fields (replace forex pip_size/session/leverage) ──
    sector: str = ""                        # GICS sector, e.g. "Information Technology"
    market_cap: float = 0.0                 # USD
    shares_outstanding: float = 0.0
    adv_20d_dollars: float = 0.0            # 20-day average dollar volume (liquidity/impact gate)
    is_halted: bool = False                 # trading halt (LULD / news pending / SSR trigger)
    short_locate_available: bool = True     # False = broker has no shares to borrow
    is_hard_to_borrow: bool = False
    is_earnings_today: bool = False         # exclude from cross-sectional universe today
    is_regular_trading_hours: bool = True   # False = pre/post market (DataEngine session gate)
    prev_close_adjusted: float = 0.0        # split/dividend-adjusted prior close, for gap calc

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def above_vwap(self) -> bool:
        return self.price > self.vwap

    @property
    def bb_position(self) -> float:
        """0 = at lower band, 1 = at upper band."""
        band_width = self.bb_upper - self.bb_lower
        if band_width == 0:
            return 0.5
        return (self.price - self.bb_lower) / band_width

    @property
    def vwap_deviation_atr(self) -> float:
        """Signed deviation from VWAP expressed in ATR5 units."""
        if self.atr5 <= 0:
            return 0.0
        return (self.price - self.vwap) / self.atr5

    @property
    def overnight_gap_pct(self) -> float:
        """Gap from previous adjusted close to current price (%)."""
        if self.prev_close_adjusted <= 0:
            return 0.0
        return (self.price - self.prev_close_adjusted) / self.prev_close_adjusted

    @property
    def is_tradeable(self) -> bool:
        """Chan Ch.5 filter: exclude sub-$5 stocks (wider spreads, higher relative cost)."""
        return (
            self.price >= 5.0
            and not self.is_halted
            and self.is_regular_trading_hours
            and self.quote_ready
        )


@dataclass
class MarketState:
    """Output of the market-regime classification engine.

    `regime` / `confidence` are the GLOBAL aggregate (dashboard display).
    Per-instrument regimes live in `regime_by_pair` / `confidence_by_pair`.
    """

    regime: str          # "trend" | "range" | "event_driven"
    confidence: float    # 0.0 – 1.0
    spy_vol_ratio: float             # SPY realized-vol ratio vs 20d baseline (index proxy)
    spy_intraday_range_pct: float    # SPY intraday range % (index proxy)
    timestamp: datetime

    # SPY 5-min directional bias: +1.0 = up, -1.0 = down, 0.0 = neutral/unknown
    spy_futures_direction: float = 0.0

    regime_by_pair: dict = field(default_factory=dict)
    confidence_by_pair: dict = field(default_factory=dict)

    def regime_for(self, code: str) -> str:
        return self.regime_by_pair.get(code) or self.regime

    def confidence_for(self, code: str) -> float:
        return self.confidence_by_pair.get(code, self.confidence)


# ─────────────────────────────────────────────────────────────────────────────
# Single-instrument signal types (kept for architecture parity / future
# momentum-overlay strategies; NOT used by the two MVP strategies).
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RawSignal:
    id: str
    code: str
    strategy: str
    side: str  # "buy" | "sell"
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timestamp: datetime
    metadata: dict = field(default_factory=dict)
    regime_match_score: float = 0.0

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_per_unit(self) -> float:
        return abs(self.take_profit - self.entry_price)

    @property
    def gross_rr(self) -> float:
        r = self.risk_per_unit
        return self.reward_per_unit / r if r > 0 else 0.0


@dataclass
class QualifiedSignal:
    raw: RawSignal
    suggested_quantity: int
    gross_risk: float
    gross_reward: float
    estimated_cost: float
    net_reward_risk_ratio: float
    approved: bool
    rejection_reason: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy A — cointegrated pairs trading (Chan Ch.7, pp.126-133).
# A pair trade is TWO legs sharing one hedge ratio and one spread-based exit;
# it cannot be represented as two independent RawSignals (that would let the
# risk engine approve one leg and reject the other, breaking the hedge).
# ─────────────────────────────────────────────────────────────────────────────

class SpreadSide(str, Enum):
    LONG_SPREAD = "long_spread"    # long code_a, short hedge_ratio * code_b
    SHORT_SPREAD = "short_spread"  # short code_a, long hedge_ratio * code_b


@dataclass
class CointegrationResult:
    """Output of python/stat/cointegration.py — CADF test + OLS hedge ratio."""
    code_a: str
    code_b: str
    hedge_ratio: float          # OLS beta: code_a ≈ hedge_ratio * code_b + spread
    cadf_tstat: float
    cadf_crit_1pct: float
    cadf_crit_5pct: float
    cadf_crit_10pct: float
    is_cointegrated_5pct: bool
    half_life_days: float       # Ornstein-Uhlenbeck half-life; <=0 means non-mean-reverting
    spread_mean: float
    spread_std: float
    computed_at: datetime
    lookback_days: int

    @property
    def is_tradeable(self) -> bool:
        """Cointegrated at 5%, mean-reverts (positive finite half-life), and
        the half-life is short enough to be statistically estimated reliably
        (Chan p.131: half-life uses the whole series, more robust than trade
        counting, but a half-life longer than the lookback window is not
        trustworthy)."""
        return (
            self.is_cointegrated_5pct
            and 0 < self.half_life_days < self.lookback_days / 2
        )


@dataclass
class SpreadSignal:
    id: str
    strategy: str
    code_a: str
    code_b: str
    hedge_ratio: float
    side: SpreadSide
    z_score: float               # current spread z-score at signal time
    entry_z_threshold: float
    exit_z_threshold: float
    spread_mean: float           # mu — the O-U long-run mean (target)
    half_life_days: float
    confidence: float
    timestamp: datetime
    metadata: dict = field(default_factory=dict)

    @property
    def entry_side_a(self) -> str:
        """buy|sell for the code_a leg."""
        return "buy" if self.side == SpreadSide.LONG_SPREAD else "sell"

    @property
    def entry_side_b(self) -> str:
        """buy|sell for the code_b leg (opposite direction, hedge_ratio-weighted)."""
        return "sell" if self.side == SpreadSide.LONG_SPREAD else "buy"


@dataclass
class QualifiedSpreadOrder:
    raw: SpreadSignal
    qty_a: int                   # shares of code_a (always positive; side above gives direction)
    qty_b: int                   # shares of code_b (hedge_ratio-scaled)
    gross_notional: float        # combined notional of both legs (USD)
    estimated_cost: float        # commission + SEC/FINRA fees + borrow cost estimate (USD)
    kelly_fraction_used: float   # fraction of the pairs-strategy capital pool allocated
    approved: bool
    rejection_reason: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy B — daily cross-sectional mean reversion (Chan Ch.3, Example 3.7/3.8).
# Evaluated once per day across the whole universe; output is a target weight
# vector, not a per-instrument evaluate() call. See strategies/portfolio_base.py.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioTarget:
    """Target dollar-weight vector produced by a PortfolioStrategy for one
    trading day. Weights are signed fractions of the strategy's allocated
    capital: +0.02 = 2% of capital long, -0.015 = 1.5% of capital short.
    """
    strategy: str
    as_of: datetime
    weights: dict = field(default_factory=dict)   # {code: signed_weight_fraction}
    metadata: dict = field(default_factory=dict)


@dataclass
class QualifiedPortfolioOrder:
    raw: PortfolioTarget
    target_shares: dict = field(default_factory=dict)     # {code: signed share count}
    rejected_codes: dict = field(default_factory=dict)    # {code: rejection_reason}
    gross_notional: float = 0.0
    estimated_cost: float = 0.0
    kelly_fraction_used: float = 0.0
    approved: bool = True
    metadata: dict = field(default_factory=dict)
