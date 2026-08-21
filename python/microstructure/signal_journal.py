"""
Append-only signal journal (docs/microstructure_pivot_plan.md §5/§7/§9,
"訊號日誌（signal journal）"): records the FULL triggering context for
every LIVE microstructure `MicroSignal` that fires — regardless of whether
`RiskEngine.qualify_microstructure_order` goes on to approve or reject it
— to `data/signal_journal/<YYYY-MM-DD>.jsonl` (one JSON object per line,
append-only). Wired into `dashboard/live_microstructure_scheduler.py`'s
`MicrostructureScheduler._qualify_and_publish` so a journal entry is
written at the exact point a signal is about to be published to the bus —
see that module for the hook.

Why this exists (plan §7): "把訊號迴路真正接上 dashboard" — Phase 2's exit
criterion needs an honest, continuously-accumulating record of every
signal actually seen live, GO-signal or not, to eventually compare against
`scripts/run_intraday_backtest.py`'s own backtested trade log ("紙上實測 vs
回測" — the gap between what was live-observed and what the backtest
predicted). This module only guarantees the record exists; the comparison
report itself is future work once enough days have accumulated.

File rotation: one file per SESSION date, derived from the signal's own
`signal_time` (an ET-naive `pd.Timestamp` by this pipeline's existing
convention — see `dashboard/live_microstructure_scheduler.py`'s
`_to_et_naive` docstring) — NOT wall-clock "now" at write time. This keeps
rotation deterministic under test and correctly files a signal under the
trading session it actually belongs to.

Outcome tracking is a DELIBERATE placeholder in this pass
(`outcome: {"status": "pending", ...}` on every recorded entry). Every
entry also carries `risk_passed` (bool) — whether
`RiskEngine.qualify_microstructure_order` approved this specific signal —
so a reader can separate "the signal fired AND would have been sized/
approved" from "the signal fired but got filtered out by risk gates"
without waiting for outcome data.

Why outcome isn't wired up yet: `MicrostructureScheduler` only tracks
*submitted-and-accepted* order acceptance for its own open-position count
(see that module's `_on_execution_report` docstring's "KNOWN LIMITATION"
note) — and because this whole feature is strictly observe-only,
`ExecutionGateway._on_microstructure_order` always rejects with
`reason="observe_mode"` before ever calling the broker. So there is
currently no accepted entry for ANY observe-mode signal to eventually
resolve into a real fill/exit to record here. Building genuine
hypothetical (never-submitted) paper-fill simulation — replaying
subsequent bars against entry/stop/target the way
`python/backtest/intraday_engine.py` already does offline, but live — is
real, non-trivial follow-up work, explicitly left as such rather than
half-built. `update_outcome()` below is the seam a future patch hooks into
once that simulation exists.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_JOURNAL_DIR = Path("data/signal_journal")
_ET = ZoneInfo("America/New_York")


def today_et_date() -> date:
    """Current date in US/Eastern — the same trading-session date
    convention every entry's file rotation uses (see module docstring)."""
    return datetime.now(timezone.utc).astimezone(_ET).date()


class SignalJournal:
    """One instance is stateless/cheap to construct — no open file handles
    are held between calls (append-open-close per `record()`), matching
    this being called at most a few times a minute from live signal
    evaluation, never in a hot per-tick loop."""

    def __init__(self, output_dir: str | Path = DEFAULT_JOURNAL_DIR) -> None:
        self._dir = Path(output_dir)

    def _file_path(self, session_date) -> Path:
        d = pd.Timestamp(session_date).date()
        return self._dir / f"{d.isoformat()}.jsonl"

    def record(self, signal, order) -> dict:
        """`signal`: a `python.microstructure.signals.MicroSignal` (or
        anything with the same attributes). `order`: the
        `python.core.types.QualifiedMicroOrder` RiskEngine already
        produced for THIS signal — `order.approved` is exactly the
        "passed RiskEngine's gate" flag recorded as `risk_passed` below.
        Never raises: a write failure is logged and the (still-returned)
        entry is simply not persisted, so a disk problem here can never
        take down the live signal pipeline that calls this."""
        entry = {
            "signal_time": pd.Timestamp(signal.signal_time).isoformat(),
            "symbol": signal.symbol,
            "strategy": signal.strategy,
            "direction": signal.direction,
            "entry_price": signal.entry_price,
            "stop_price": signal.stop_price,
            "target_price": signal.target_price,
            "order_type": signal.order_type,
            "expiry_time": signal.expiry_time.isoformat() if signal.expiry_time is not None else None,
            "context": dict(signal.context or {}),
            "risk_passed": bool(order.approved),
            "rejection_reason": order.rejection_reason,
            "qualified_order": {
                "qty": order.qty,
                "entry_limit_price": order.entry_limit_price,
                "stop_price": order.stop_price,
                "stop_limit_price": order.stop_limit_price,
                "target_price": order.target_price,
                "gross_notional": order.gross_notional,
            },
            # Deliberate placeholder — see module docstring's "Outcome
            # tracking" section for exactly why this isn't filled in yet.
            "outcome": {
                "status": "pending",  # "pending" | "closed"
                "exit_price": None,
                "exit_time": None,
                "pnl": None,
                "pnl_pct": None,
            },
        }
        self._append(signal.signal_time, entry)
        return entry

    def _append(self, session_date, entry: dict) -> None:
        path = self._file_path(session_date)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            log.exception("SignalJournal: failed writing to %s — entry dropped", path)

    def read_day(self, session_date) -> list[dict]:
        """Every recorded entry for one session date, in the order they
        were written. Returns [] (not an error) if nothing was ever
        recorded that day — same "absence is not an error" convention as
        the rest of this diagnostic layer (e.g.
        `intraday_cache.get_cached_intraday_panel`)."""
        path = self._file_path(session_date)
        if not path.exists():
            return []
        rows: list[dict] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        log.warning("SignalJournal: skipping malformed line in %s", path)
        except OSError:
            log.exception("SignalJournal: failed reading %s", path)
        return rows

    def read_today(self) -> list[dict]:
        return self.read_day(today_et_date())
