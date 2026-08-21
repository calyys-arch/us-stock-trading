"""
Causal tick replay helper — slice captured prints up to "now".

No lookahead: a signal evaluated at bar close `now` may only see prints
with timestamp <= now. This is the tick analogue of
`bars.iloc[:i+1]` in python/backtest/intraday_engine.py. It does not
reconstruct an L2 book (that would be depth_replay.py, still unbuilt).
"""
from __future__ import annotations

from ..data.tick_cache import ticks_up_to

__all__ = ["ticks_up_to"]
