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

GO if >= `min_pass_folds_ratio` of OOS folds individually pass (OOS Sharpe
does not decay more than `max_sharpe_decay` vs the fold's IS Sharpe, and OOS
Sharpe clears `min_oos_sharpe_abs`).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
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


@dataclass
class WFOResult:
    folds: list
    total_folds: int
    passing_folds: int
    pass_ratio: float
    decision: str
    config: dict
    run_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def oos_sharpe_mean(self) -> float:
        vals = [f.oos_sharpe for f in self.folds]
        return sum(vals) / len(vals) if vals else 0.0

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "total_folds": self.total_folds,
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
        print("\n== Walk-Forward Optimization ==")
        print(f"  Decision       : {d['decision']}")
        print(f"  Folds          : {d['passing_folds']} / {d['total_folds']} passed ({d['pass_ratio']:.0%})")
        print(f"  OOS Sharpe mean: {d['oos_sharpe_mean']:.3f}")
        for f in d["folds"]:
            status = "PASS" if f["oos_pass"] else "FAIL"
            print(f"  Fold {f['fold_idx']:2d}  IS={f['is_sharpe']:+.3f}  OOS={f['oos_sharpe']:+.3f}  {status}")
        print("=" * 55 + "\n")


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
            oos_pass = decay_ok and abs_ok

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

        passing = sum(1 for f in folds if f.oos_pass)
        pass_ratio = passing / len(folds)
        decision = "GO" if pass_ratio >= cfg.min_pass_folds_ratio else "NO-GO"

        return WFOResult(
            folds=folds, total_folds=len(folds), passing_folds=passing,
            pass_ratio=pass_ratio, decision=decision, config=asdict(cfg),
        )
