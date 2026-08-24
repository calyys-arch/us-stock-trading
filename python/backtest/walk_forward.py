"""
Walk-Forward Optimizer (WFO) — ported from
forex-trading/python/backtest/walk_forward.py, generalized from tick-count
windows to DATE-based windows (daily bars, not intraday ticks, are this
repo's primary backtest granularity for both strategies), and decoupled
from any specific engine: the caller supplies a `backtest_fn(start, end,
params) -> dict` callable that runs whichever engine is appropriate
(vector_engine for cross-sectional, event-driven engine for pairs) and
returns a metrics dict with at least a "sharpe_ratio" key.

    ┌────────────────────────────────────────────────────────────────┐
    │  Fold 1  │  IS (train)      │  OOS (validate)  │                │
    │  Fold 2  │                  │   IS (train)     │  OOS (validate)│
    │  ...
    └────────────────────────────────────────────────────────────────┘

Each fold:
  1. Backtest the IS window once per candidate in `param_grid`.
  2. Pick the best candidate by IS Sharpe ratio (per-fold re-optimization —
     an honest simulation of "optimize on the past, trade the future").
  3. Backtest the winning candidate on the OOS window.
  4. Record OOS metrics as the fold result.

GO if >= `min_pass_folds_ratio` of the EVALUABLE OOS folds individually pass
(OOS Sharpe does not decay more than `max_sharpe_decay` vs the fold's IS
Sharpe, and OOS Sharpe clears `min_oos_sharpe_abs`).

"Evaluable" means the fold actually traded. A fold where the signal never
fired is excluded from the ratio on both sides rather than scored, because it
carries no information about whether the edge extrapolates — and because the
pass test would otherwise award it a PASS: with no trades, both Sharpes are
0.0, so the decay test degenerates to `0.0 >= 0` and the absolute test to
`0.0 >= 0.0`. If fewer than `min_evaluable_folds_ratio` of folds traded, the
verdict is INCONCLUSIVE rather than GO or NO-GO, since a ratio computed over
two folds out of eight is a stricter numerator bought with a meaningless
sample. Callers all test `decision == "GO"`, so INCONCLUSIVE fails closed.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

BacktestFn = Callable[[datetime, datetime, dict], dict]


@dataclass
class WFOConfig:
    is_days: int = 504     # ~2 trading years in-sample
    oos_days: int = 126    # ~6 trading months out-of-sample
    step_days: int = 126
    min_pass_folds_ratio: float = 0.60
    min_oos_sharpe_abs: float = 0.0
    max_sharpe_decay: float = 0.5   # OOS sharpe must be >= IS sharpe * (1 - max_sharpe_decay)
    # Fraction of folds that must actually have traded for the pass ratio to
    # mean anything. Folds with no fills are excluded from the ratio entirely
    # (see FoldResult.is_evaluable), which on its own would let 2 traded folds
    # out of 8 produce a ratio of 1.0 — a stricter numerator bought with a
    # sample small enough to be meaningless. Below this fraction the run is
    # INCONCLUSIVE, which is reported separately from NO-GO because "we could
    # not tell" and "we looked and it failed" are different findings, and only
    # the second is evidence about the signal.
    min_evaluable_folds_ratio: float = 0.60
    # False (default): ROLLING fixed-width IS window that SLIDES forward by
    # step_days each fold — the original behavior, byte-identical for every
    # existing caller (backtests/reports/regime_gate_robustness_report.md
    # left this default untouched; it only reads it via a NEW, opt-in `True`
    # value). True: ANCHORED/EXPANDING IS window — is_start stays fixed at
    # the study's start date and is_end grows by step_days each fold instead
    # of sliding. Both are standard, textbook walk-forward conventions
    # (rolling vs. anchored/expanding walk-forward optimization — e.g. Pardo,
    # "The Evaluation and Optimization of Trading Strategies", ch. 5), not a
    # bespoke fold rule invented for one result.
    anchored: bool = False


@dataclass
class FoldResult:
    fold_idx: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    is_sharpe: float
    oos_sharpe: float
    oos_pass: bool
    best_params: dict = field(default_factory=dict)
    oos_metrics: dict = field(default_factory=dict)

    @property
    def is_evaluable(self) -> bool:
        """Did this fold produce any out-of-sample fills to judge?

        A fold where the signal never fired carries no information about
        whether the edge extrapolates. It used to be scored as a PASS: with no
        trades on either side both Sharpes are 0.0, so `is_sharpe > 0` is
        false, the decay test degenerates to `0.0 >= 0` and the absolute test
        to `0.0 >= 0.0`. Both hold, and the empty fold counts toward the pass
        ratio.

        That was harmless while `wfo_go` was only a warning. Promoting it to a
        hard gate made it decision-flipping, and absorption_breakout_15m
        promptly exercised it: folds 1-3 fired nothing, and had folds 5-7 done
        the same the ratio would have read 6/8 = 0.75 against a 0.60 floor —
        a GO for a strategy that traded in two folds out of eight and lost
        10.2 Sharpe in one of them. `has_oos_trades` does not catch it, since
        it only asks whether ANY fold traded.
        """
        return int(self.oos_metrics.get("n_trades", 0)) > 0


@dataclass
class WFOResult:
    folds: list
    total_folds: int
    passing_folds: int
    pass_ratio: float
    decision: str
    config: dict
    # Folds that actually traded. `pass_ratio` is over THESE, not over
    # total_folds; the two differ whenever a signal went quiet for a whole OOS
    # window. Defaulted so that older callers constructing a WFOResult by hand
    # keep working.
    evaluable_folds: int = 0
    run_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def oos_sharpe_mean(self) -> float:
        """Mean OOS Sharpe over folds that traded.

        Empty folds are excluded rather than averaged in as 0.0, which
        otherwise drags a losing signal's mean toward zero and makes it look
        merely flat.
        """
        vals = [f.oos_sharpe for f in self.folds if f.is_evaluable]
        return sum(vals) / len(vals) if vals else 0.0

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "total_folds": self.total_folds,
            "evaluable_folds": self.evaluable_folds,
            "passing_folds": self.passing_folds,
            "pass_ratio": round(self.pass_ratio, 3),
            "oos_sharpe_mean": round(self.oos_sharpe_mean, 3),
            "config": self.config,
            "run_at": self.run_at,
            "folds": [asdict(f) for f in self.folds],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        log.info("WFO result saved to %s", path)
        return path

    def print_summary(self) -> None:
        d = self.to_dict()
        empty = d["total_folds"] - d["evaluable_folds"]
        print("\n== Walk-Forward Optimization ==")
        print(f"  Decision       : {d['decision']}")
        print(f"  Folds          : {d['passing_folds']} / {d['evaluable_folds']} passed "
              f"({d['pass_ratio']:.0%}) of {d['total_folds']} total"
              + (f", {empty} with no fills" if empty else ""))
        print(f"  OOS Sharpe mean: {d['oos_sharpe_mean']:.3f}  (traded folds only)")
        for f in d["folds"]:
            traded = int((f.get("oos_metrics") or {}).get("n_trades", 0)) > 0
            status = "PASS" if f["oos_pass"] else "FAIL"
            if not traded:
                status = "EMPTY"
            print(f"  Fold {f['fold_idx']:2d}  IS={f['is_sharpe']:+.3f}  OOS={f['oos_sharpe']:+.3f}  {status}")
        print("=" * 55 + "\n")


def _decide(folds: list, cfg: WFOConfig) -> WFOResult:
    """Score folds into a GO / NO-GO / INCONCLUSIVE verdict.

    Split out from the optimizer loop so that a completed run's stored folds
    can be rescored without repeating the candidate search — the expensive
    part, at roughly half an hour per fold.
    """
    evaluable = [f for f in folds if f.is_evaluable]
    passing = sum(1 for f in evaluable if f.oos_pass)
    pass_ratio = (passing / len(evaluable)) if evaluable else 0.0

    if not folds:
        # Kept as NO-GO, matching the "date range too short" early return in
        # the optimizer, rather than INCONCLUSIVE: no folds at all is a caller
        # error about the window, not a finding about the signal.
        decision = "NO-GO"
    elif (len(evaluable) / len(folds)) < cfg.min_evaluable_folds_ratio:
        decision = "INCONCLUSIVE"
        log.warning(
            "WFO: only %d of %d folds produced fills (need %.0f%%) — the "
            "pass ratio is not a verdict on this signal",
            len(evaluable), len(folds), cfg.min_evaluable_folds_ratio * 100,
        )
    else:
        decision = "GO" if pass_ratio >= cfg.min_pass_folds_ratio else "NO-GO"

    return WFOResult(
        folds=folds, total_folds=len(folds), evaluable_folds=len(evaluable),
        passing_folds=passing, pass_ratio=pass_ratio, decision=decision,
        config=asdict(cfg),
    )


def rescore_folds(payload: dict, cfg: WFOConfig | None = None) -> WFOResult:
    """Rebuild a verdict from a saved WFO payload's folds.

    Reads the config back out of the payload where present, so rescoring an
    old run does not silently re-judge it under today's fold geometry.
    """
    stored = dict(payload.get("config") or {})
    if cfg is None:
        known = {f.name for f in fields(WFOConfig)}
        cfg = WFOConfig(**{k: v for k, v in stored.items() if k in known})
    folds = [
        FoldResult(**{k: v for k, v in f.items()
                      if k in {fl.name for fl in fields(FoldResult)}})
        for f in payload.get("folds") or []
    ]
    return _decide(folds, cfg)


class WalkForwardOptimizer:
    def __init__(
        self,
        backtest_fn: BacktestFn,
        config: WFOConfig | None = None,
        param_grid: list | None = None,
    ) -> None:
        self._backtest_fn = backtest_fn
        self._cfg = config or WFOConfig()
        self._param_grid: list = list(param_grid) if param_grid else [{}]

    def run(self, start: datetime, end: datetime) -> WFOResult:
        cfg = self._cfg
        folds: list[FoldResult] = []
        fold_idx = 0

        window_start = start
        while True:
            if cfg.anchored:
                # Anchored/expanding: IS always starts at the study's start;
                # IS length grows by one step_days per fold (fold 0's IS is
                # still exactly is_days long, matching the rolling case, so
                # the two conventions are directly comparable fold-for-fold).
                is_start = start
                is_end = is_start + timedelta(days=cfg.is_days + fold_idx * cfg.step_days)
            else:
                is_start = window_start
                is_end = is_start + timedelta(days=cfg.is_days)
            oos_end = is_end + timedelta(days=cfg.oos_days)
            if oos_end > end:
                break

            best_params: dict = {}
            best_sharpe = -math.inf
            is_metrics: dict = {}
            for candidate in self._param_grid:
                metrics = self._backtest_fn(is_start, is_end, candidate)
                sharpe = metrics.get("sharpe_ratio", 0.0)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = candidate
                    is_metrics = metrics

            oos_metrics = self._backtest_fn(is_end, oos_end, best_params)
            oos_sharpe = oos_metrics.get("sharpe_ratio", 0.0)
            is_sharpe = is_metrics.get("sharpe_ratio", 0.0)

            decay_ok = oos_sharpe >= is_sharpe * (1 - cfg.max_sharpe_decay) if is_sharpe > 0 else oos_sharpe >= 0
            abs_ok = oos_sharpe >= cfg.min_oos_sharpe_abs
            # A fold that never traded has demonstrated nothing, so it cannot
            # pass. Both tests above would otherwise hold on its all-zero
            # series. `_decide` drops such folds from the ratio entirely, but
            # the flag is also set false here because it is read directly
            # elsewhere (scripts/_regime_gate_robustness_pairs.py bootstraps
            # the per-fold pass sequence), and there it must fail closed.
            traded = int(oos_metrics.get("n_trades", 0)) > 0
            oos_pass = traded and decay_ok and abs_ok

            fold = FoldResult(
                fold_idx=fold_idx,
                is_start=is_start.isoformat(), is_end=is_end.isoformat(),
                oos_start=is_end.isoformat(), oos_end=oos_end.isoformat(),
                is_sharpe=is_sharpe, oos_sharpe=oos_sharpe, oos_pass=oos_pass,
                best_params=best_params, oos_metrics=oos_metrics,
            )
            folds.append(fold)
            log.info("WFO fold %d: IS=%.3f OOS=%.3f pass=%s", fold_idx, is_sharpe, oos_sharpe, oos_pass)

            fold_idx += 1
            window_start = window_start + timedelta(days=cfg.step_days)

        if not folds:
            log.warning("WFO: date range too short for a single fold")
            return WFOResult(folds=[], total_folds=0, passing_folds=0, pass_ratio=0.0,
                              decision="NO-GO", config=asdict(cfg))

        return _decide(folds, cfg)
