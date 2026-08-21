"""
Naive options GEX (gamma exposure) — Creamer's "environment" input.

Christopher Creamer (public interview) reads Tanuki's naive GEX on QQQ/NDX
as a VOLATILITY REGIME, not a direction: positive GEX ≈ dealers fade rips /
buy dips (choppy, failed breakouts); negative GEX ≈ dealers amplify.
He also notes call wall / put wall / gamma flip as location magnets.

This module computes the same *class* of naive dealer-gamma snapshot from
an options chain. It is NOT Tanuki, NOT CBOE DEX/GEX, and NOT inferred
from price. Missing chain → None (fail closed: never invent dealer gamma).

Formula (SpotGamma-style 1% move, dealers short customer gamma):
    call_gex(K) = +gamma * OI * 100 * S^2 * 0.01
    put_gex(K)  = -gamma * OI * 100 * S^2 * 0.01
Gamma is Black-Scholes when the chain supplies IV; a row with no usable
IV/OI is dropped, not zero-filled.

Structural constants (not free parameters, not gridded): 45-day DTE cap,
risk-free rate 0. Structural walls = strike of max call GEX / min put GEX.
Flip = lowest strike where cumulative net GEX (low→high) crosses through 0.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

_CONTRACT_MULT = 100.0
_MOVE = 0.01
_DTE_MAX = 45
_RATE = 0.0


@dataclass(frozen=True)
class GexSnapshot:
    symbol: str
    as_of: str
    source: str
    spot: float
    net_gex: float
    call_gex: float
    put_gex: float
    regime: str
    call_wall: float | None
    put_wall: float | None
    gamma_flip: float | None
    expiries_used: list[str] = field(default_factory=list)
    dte_max: int = _DTE_MAX
    formula: str = "naive_dealer_short_gamma_1pct"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> GexSnapshot:
        return cls(
            symbol=str(raw["symbol"]),
            as_of=str(raw["as_of"]),
            source=str(raw.get("source", "unknown")),
            spot=float(raw["spot"]),
            net_gex=float(raw["net_gex"]),
            call_gex=float(raw["call_gex"]),
            put_gex=float(raw["put_gex"]),
            regime=str(raw.get("regime") or regime_from_net(float(raw["net_gex"]))),
            call_wall=_opt_float(raw.get("call_wall")),
            put_wall=_opt_float(raw.get("put_wall")),
            gamma_flip=_opt_float(raw.get("gamma_flip")),
            expiries_used=list(raw.get("expiries_used") or []),
            dte_max=int(raw.get("dte_max", _DTE_MAX)),
            formula=str(raw.get("formula", "naive_dealer_short_gamma_1pct")),
        )


def _opt_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_gamma(spot: float, strike: float, tte_years: float, iv: float, rate: float = _RATE) -> float:
    """Black-Scholes gamma. 0 when any input is unusable (no silent NaN)."""
    if spot <= 0 or strike <= 0 or tte_years <= 0 or iv <= 0:
        return 0.0
    sqrt_t = math.sqrt(tte_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * tte_years) / (iv * sqrt_t)
    denom = spot * iv * sqrt_t
    if denom <= 0:
        return 0.0
    return _norm_pdf(d1) / denom


def regime_from_net(net_gex: float, deadband: float = 0.0) -> str:
    if net_gex > deadband:
        return "positive_gamma"
    if net_gex < -deadband:
        return "negative_gamma"
    return "neutral"


def _tte_years(expiry: date, as_of: date) -> float:
    days = (expiry - as_of).days
    if days < 0:
        return 0.0
    # 0DTE still has a few RTH hours; treat as a quarter-day so gamma is defined.
    return max(days, 0.25) / 365.0


def _parse_expiry(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def compute_naive_gex(
    symbol: str,
    spot: float,
    chains: list[dict],
    as_of: date | str | None = None,
    source: str = "synthetic",
    dte_max: int = _DTE_MAX,
) -> GexSnapshot | None:
    """`chains` is a list of {expiry, calls, puts} where calls/puts are
    iterables of dicts with strike, open_interest (or openInterest), and
    optional implied_volatility / impliedVolatility / gamma.

    Returns None when spot is unusable or no strike contributed any GEX —
    never a zeroed-out fake regime."""
    if spot <= 0:
        return None
    as_of_date = _parse_expiry(as_of) or date.today()
    scale = _CONTRACT_MULT * spot * spot * _MOVE

    per_strike: dict[float, dict[str, float]] = {}
    expiries_used: list[str] = []

    for chain in chains:
        expiry = _parse_expiry(chain.get("expiry"))
        if expiry is None:
            continue
        dte = (expiry - as_of_date).days
        if dte < 0 or dte > dte_max:
            continue
        tte = _tte_years(expiry, as_of_date)
        used = False
        for side, sign in (("calls", 1.0), ("puts", -1.0)):
            for row in chain.get(side) or []:
                strike = _opt_float(row.get("strike"))
                oi = _opt_float(row.get("open_interest", row.get("openInterest")))
                if strike is None or strike <= 0 or oi is None or oi <= 0:
                    continue
                gamma = _opt_float(row.get("gamma"))
                if gamma is None or gamma <= 0:
                    iv = _opt_float(
                        row.get("implied_volatility", row.get("impliedVolatility")),
                    )
                    gamma = bs_gamma(spot, strike, tte, iv or 0.0)
                if gamma <= 0:
                    continue
                gex = sign * gamma * oi * scale
                bucket = per_strike.setdefault(strike, {"call": 0.0, "put": 0.0})
                if sign > 0:
                    bucket["call"] += gex
                else:
                    bucket["put"] += gex
                used = True
        if used:
            expiries_used.append(expiry.isoformat())

    if not per_strike:
        return None

    call_gex = sum(v["call"] for v in per_strike.values())
    put_gex = sum(v["put"] for v in per_strike.values())
    net_gex = call_gex + put_gex

    call_wall = max(per_strike, key=lambda k: per_strike[k]["call"])
    put_wall = min(per_strike, key=lambda k: per_strike[k]["put"])
    if per_strike[call_wall]["call"] <= 0:
        call_wall = None
    if per_strike[put_wall]["put"] >= 0:
        put_wall = None

    flip = None
    cumulative = 0.0
    prev = None
    for strike in sorted(per_strike):
        cumulative += per_strike[strike]["call"] + per_strike[strike]["put"]
        if prev is not None and prev < 0 <= cumulative:
            flip = strike
            break
        prev = cumulative

    return GexSnapshot(
        symbol=symbol.upper(),
        as_of=as_of_date.isoformat(),
        source=source,
        spot=float(spot),
        net_gex=float(net_gex),
        call_gex=float(call_gex),
        put_gex=float(put_gex),
        regime=regime_from_net(net_gex),
        call_wall=float(call_wall) if call_wall is not None else None,
        put_wall=float(put_wall) if put_wall is not None else None,
        gamma_flip=float(flip) if flip is not None else None,
        expiries_used=sorted(set(expiries_used)),
        dte_max=dte_max,
    )


def resolve_gex_env(gex_snapshot: GexSnapshot | dict | None) -> tuple[GexSnapshot | None, GexSnapshot | None]:
    """Accept a single snapshot or ``{"market": ..., "symbol": ...}``.
    Returns (environment, symbol_walls). Environment prefers market (QQQ)
    — that is the index GEX Creamer actually reads."""
    if gex_snapshot is None:
        return None, None
    if isinstance(gex_snapshot, GexSnapshot):
        return gex_snapshot, gex_snapshot
    if not isinstance(gex_snapshot, dict):
        return None, None
    if "market" in gex_snapshot or "symbol" in gex_snapshot:
        market = _coerce_snap(gex_snapshot.get("market"))
        symbol = _coerce_snap(gex_snapshot.get("symbol"))
        return market or symbol, symbol
    try:
        snap = GexSnapshot.from_dict(gex_snapshot)
    except (KeyError, TypeError, ValueError):
        return None, None
    return snap, snap


def _coerce_snap(value) -> GexSnapshot | None:
    if value is None:
        return None
    if isinstance(value, GexSnapshot):
        return value
    if isinstance(value, dict):
        try:
            return GexSnapshot.from_dict(value)
        except (KeyError, TypeError, ValueError):
            return None
    return None
