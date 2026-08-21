"""
Paper-forward-test allowlists (2026-08-15).

This is NOT a WFO GO promotion. `absorption_breakout` remains a frozen
NO-GO-but-closest candidate under a 30–60 trading-day forward experiment;
`pairs_trading` is regime-conditional automation based on the 2022 GO
evidence in `backtests/reports/regime_generalization_report.md`. See
`backtests/reports/absorption_breakout_paper_protocol.md` and
`backtests/reports/pairs_regime_live_protocol.md`.

Retired / confirmed-losing microstructure signals must never appear in
`LIVE_SIGNALS` and must never be granted `auto_execute` at the gateway,
even if a human later flips their yaml flag.
"""
from __future__ import annotations

# Frozen TIGHT6 universe from absorption_breakout_investigation_report.md §4
# (B5 winner). Do not expand without a new pre-declared protocol.
ABSORPTION_BREAKOUT_UNIVERSE: tuple[str, ...] = (
    "AAPL", "GOOGL", "NVDA", "MSFT", "PLTR", "INTC",
)

# The only two names that may be armed for paper auto-execution.
PAPER_AUTO_ALLOWLIST: frozenset[str] = frozenset({
    "absorption_breakout",
    "pairs_trading",
})

# Confirmed-losing / RETIRED microstructure signals — live footguns.
# Must never be in LIVE_SIGNALS and must never pass the paper allowlist.
RETIRED_MICRO_SIGNALS: frozenset[str] = frozenset({
    "sweep_reclaim",
    "fvg_retest",
    "orb_vwap",
    "orb_vwap_regime",
    "vwap_band_fade",
    "vp_breakout",
    "l2_absorption",
})
