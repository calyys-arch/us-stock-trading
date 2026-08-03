"""
Shared signal contract for the microstructure signal modules
(sweep_reclaim, fvg_retest, orb_vwap, l2_absorption — see
docs/microstructure_pivot_plan.md §1 "Signal Engine". l2_absorption is a
bar-only proxy for its full spec — see that module's docstring — and is
observe-only: it is intentionally excluded from
scripts/run_intraday_backtest.py's WFO/promotion pipeline).

No-lookahead contract (enforced by construction, verified by
tests/test_intraday_signals.py): every `evaluate_*` function in this
package receives `bars` already sliced up to and including "now"
(`bars.index[-1]`) and decides only whether a signal fires AT that last
bar. None of these functions ever accept or peek at bars beyond "now" —
python/backtest/intraday_engine.py enforces the actual fill-on-next-bar
timing; these modules only ever emit the DECISION, at the close of an
already-known bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MicroSignal:
    symbol: str
    strategy: str              # "sweep_reclaim" | "fvg_retest" | "orb_vwap" | "l2_absorption"
    direction: str             # "long" | "short"
    signal_time: pd.Timestamp  # the already-closed bar the decision was made on
    entry_price: float         # limit/reference price; intraday_engine decides actual fill on NEXT bar
    stop_price: float
    target_price: float | None = None
    order_type: str = "next_open"   # "next_open" (market-style, fills at next bar open) | "limit"
    expiry_time: pd.Timestamp | None = None  # None = no time-based expiry (order_type == "next_open" cases)
    context: dict = field(default_factory=dict)  # triggering context for signal_journal / reports
