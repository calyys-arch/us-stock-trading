"""The edge-to-cost ratio a signal converges to as position size goes to zero.

Why this number exists. The engine charges two slippage terms per leg
(`intraday_engine._slippage_price`):

    total_bps = half_spread_bps + impact_bps_per_participation * participation

and they respond to a change in position size in completely different ways:

  * HALF-SPREAD is LINEAR in size. Dollar cost = notional * spread_bps, and
    pre-cost edge is also linear in notional, so halving size halves both and
    the ratio is unchanged. Size is not a lever against spread.
  * IMPACT is QUADRATIC in size, because `impact_bps` itself scales with
    `shares`. Halving size cuts impact cost to a quarter while cutting edge
    only in half, so the ratio doubles.

A position-size sweep therefore does not trend toward infinity — it saturates
once impact is negligible and only spread remains, at

    ceiling = edge_bps / spread_round_trip_bps

That limit turns "is this signal executable at all?" into a single comparison
that costs three replays instead of an hours-long walk-forward: a cell whose
pre-cost edge per trade is thinner than the round-trip spread can never clear
PF 1.0 at ANY size, however perfect the execution. Below 1.0 is a property of
the signal, not of the execution.

Commission is excluded deliberately. It is a per-share floor rather than a
proportional cost, so it gets relatively WORSE as size shrinks and would
confound the limit. The ceiling is the optimistic bound, and a cell that
fails it fails for structural reasons.

The spread figure is MEASURED rather than read off `2 * half_spread_bps`. It
comes out below nominal even though both legs do charge it (see
`_close_position`), because the attribution is a difference between separately
run paths rather than an identity over one path: switching a cost term off
moves fill prices, which moves which bars trigger stops. Callers that need the
size of that residual should compare against the jointly-costed baseline.

One deliberate difference from the first published figures: per-trade edge is
converted to basis points using the notional of the run it was measured on
(the zero-cost path) rather than the baseline's. Both paths sit on the
`max_notional_pct` cap so the two differ by under 0.05%, moving the four
published ceilings by ~0.001 — immaterial, but the consistent choice.
"""
from __future__ import annotations

# Calibration for interpreting a ceiling, anchored on the four cells measured
# by scripts/run_execution_cost_study.py with WFO-tuned params:
#
#   ceiling 4.11 (auction_reclaim_5m) -> cost-adjusted PF 1.198 at full size
#   ceiling 2.61 (vsa_no_demand_5m)   -> PF 0.789, needs 0.25x to reach 1.033
#   ceiling 0.40 (fvg_retest_1m)      -> PF 0.258
#   ceiling 0.21 (vp_breakout_1m)     -> PF 0.518
#
# Four points, so these bands rank candidates rather than predict PF. Only the
# 1.0 boundary is analytic; the rest is interpolation and should be treated as
# a queue order, not a forecast.
IMPOSSIBLE = 1.0   # below this, no position size clears PF 1.0
THIN = 2.5         # above 1.0 but only viable at sizes where net dollars vanish
PROMISING = 3.5    # the band auction_reclaim_5m sits in, i.e. PF > 1 at full size


def edge_bps_from_zero(zero: dict, notional: float | None = None) -> float | None:
    """Pre-cost edge per trade, in basis points of notional.

    `zero` is a replay with both slippage terms switched off. It is expected to
    still pay commission; commission never enters `_slippage_price`, so adding
    it back to recover a gross edge cannot change the trade set or any fill
    price — the correction is exact rather than approximate.

    Split out from `ceiling_from_runs` because it needs only ONE replay. A
    selectivity sweep re-measures edge at many parameter settings while the
    spread side barely moves (it is a property of the cost model and of a
    notional that sits on `max_notional_pct`, not of the signal), so sweeping
    with this alone costs a third of a full attribution per point.
    """
    n = int(zero.get("n_trades") or 0)
    if not n:
        return None
    denom = notional or zero.get("avg_notional")
    if not denom:
        return None
    gross = float(zero["total_net_pnl"]) + float(zero.get("commission") or 0.0)
    return (gross / n) / denom * 1e4


def spread_bps_from_runs(zero: dict, spread_only: dict) -> float | None:
    """Measured round-trip half-spread cost per trade, in basis points.

    This is the part of slippage that position size cannot escape, so it is the
    denominator of the ceiling. Measured rather than read off
    `2 * half_spread_bps`: switching a cost term off moves fill prices, which
    moves which bars trigger stops, so the figure comes out below nominal.
    """
    n = int(spread_only.get("n_trades") or 0)
    if not n:
        return None
    denom = spread_only.get("avg_notional") or zero.get("avg_notional")
    if not denom:
        return None
    cost = (
        float(zero["total_net_pnl"])
        - float(spread_only["total_net_pnl"])
        - float(spread_only.get("commission") or 0.0)
    )
    return (cost / n) / denom * 1e4


def ceiling_from_runs(
    *,
    baseline: dict,
    spread_only: dict,
    zero: dict,
) -> dict | None:
    """Compute the asymptotic edge/spread ratio from three single-path replays.

    Each argument is a metrics dict needing `n_trades`, `total_net_pnl`,
    `commission` and `avg_notional`, as produced by replaying one cell under:

        baseline     full cost model (spread on, impact on)
        spread_only  spread on, impact off
        zero         both off

    Returns None when any leg produced no trades, since every quantity here is
    a per-trade average.

    `zero` is expected to still pay commission. Commission never enters
    `_slippage_price`, so adding it back to recover a commission-free edge
    cannot change the trade set or any fill price — the correction is exact
    rather than approximate.
    """
    n_zero = int(zero.get("n_trades") or 0)
    n_spread = int(spread_only.get("n_trades") or 0)
    if not n_zero or not n_spread:
        return None

    edge_notional = zero.get("avg_notional") or baseline.get("avg_notional")
    edge_bps = edge_bps_from_zero(zero, edge_notional)
    spread_bps = spread_bps_from_runs(zero, spread_only)
    if edge_bps is None or spread_bps is None:
        return None
    return {
        "edge_bps": edge_bps,
        "spread_round_trip_bps": spread_bps,
        "ceiling": (edge_bps / spread_bps) if spread_bps else None,
        "edge_per_trade": edge_bps / 1e4 * edge_notional,
        "n_trades_zero": n_zero,
        "n_trades_spread": n_spread,
        # Total gross edge, in bps-trades. Roughly invariant to the
        # selectivity parameters within a signal (measured: 2%, 19% and 23%
        # drift across default-vs-tuned pairs), which is what makes an
        # envelope sweep meaningful — selectivity redistributes edge into
        # fewer fatter trades rather than creating it.
        "total_edge_bps_trades": edge_bps * n_zero,
    }


def tier(ceiling: float | None) -> str:
    """Bucket a ceiling into a compute-allocation decision."""
    if ceiling is None:
        return "unknown"
    if ceiling < IMPOSSIBLE:
        return "impossible"
    if ceiling < THIN:
        return "thin"
    if ceiling < PROMISING:
        return "marginal"
    return "promising"


def verdict(limit: dict | None, *, negative_edge_note: bool = True) -> str:
    """One sentence on what a ceiling means for spending WFO compute."""
    if not limit or limit.get("ceiling") is None:
        return "無法判讀（重播沒有成交）"
    ceiling = limit["ceiling"]
    edge, spread = limit["edge_bps"], limit["spread_round_trip_bps"]
    head = f"稅前邊緣 {edge:.2f} bps／來回價差 {spread:.2f} bps，上限 {ceiling:.2f}x。"

    if negative_edge_note and edge <= 0:
        return (
            f"{head} **稅前邊緣本身是負的** —— 這格連零成本都虧，"
            "不是成本問題，訊號方向就不對。不要花 WFO。"
        )
    if ceiling < IMPOSSIBLE:
        return (
            f"{head} **任何部位大小都過不了 PF 1.0**，因為邊緣比價差還薄。"
            "這是訊號性質而非執行問題，不要花 WFO。"
        )
    if ceiling < THIN:
        return (
            f"{head} 理論上有解，但只在部位小到淨額微不足道時成立"
            "（vsa_no_demand_5m 在 2.61x 就是這樣：0.25x 全年只賺 $726）。優先度低。"
        )
    if ceiling < PROMISING:
        return (
            f"{head} 邊際地帶，比已知最好的 1m 格子好一個量級，但還沒到"
            " auction_reclaim_5m 的 4.11x。值得排隊，但排在 promising 之後。"
        )
    return (
        f"{head} 餘裕足以在全額部位有正期望，屬於已知最好的一檔。"
        "值得花完整 WFO。"
    )
