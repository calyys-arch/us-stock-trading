"""
Reality Check — White's Reality Check, ported from
forex-trading/python/backtest/reality_check.py.

Generalized from a single-instrument tick-price randomizer to a PRICE PANEL
randomizer (columns = instrument codes, index = date/time, values = close
price), since both this repo's strategies need multiple instruments at once
(a pair, or the whole cross-sectional universe) — a single-series randomizer
cannot exercise either strategy's actual decision logic.

Algorithm (unchanged from forex-trading):
  1. Run the caller-supplied `backtest_fn(real_panel) -> sharpe` once on the
     REAL price panel.
  2. Phase-randomize (FFT-based) each column's log-return series
     INDEPENDENTLY. This preserves each instrument's own autocorrelation
     structure and volatility, but destroys any genuine cross-instrument
     cointegration/mean-reversion relationship the strategy might be
     exploiting — exactly the null hypothesis we want ("this Sharpe could
     have come from statistically-similar-but-unrelated random walks").
  3. Rerun `backtest_fn` on `n_sims` such randomized panels.
  4. p_value = fraction of randomized runs whose Sharpe >= the real Sharpe.
     p_value < 0.05 -> PASS (likely a genuine, non-random edge).

`backtest_fn` is supplied by the caller and can wrap either
backtest/vector_engine.run_vector_backtest (cross-sectional) or
backtest/engine.run_pairs_backtest (pairs) — this module has zero
dependency on either engine's internals, keeping it strategy-agnostic.

Multiple comparisons — two methods, one cheap, one exact:

scripts/self_improve_loop.py runs `RealityCheck` on ONE candidate: the
walk-forward winner picked out of a grid of 27-32 candidates (the fixed
param_grids.yaml grid plus any UHAI suggestions). Comparing only the winner
against a null built from single-model randomization understates how likely
a spuriously good Sharpe was to turn up SOMEWHERE in a search that wide —
this is the classic data-snooping/multiple-comparisons problem White's
Reality Check (White 2000) was actually designed to correct for, by testing
the BEST of many models against the null distribution of the best-of-many
statistic, not one model's p-value in isolation.

Two ways to get there, offered as two separate entry points because their
cost differs by roughly the number of candidates:

  - `run(panel, n_candidates_searched=N)` (cheap, default path): keeps the
    existing single-model simulation (`N=1` behaves exactly as before — every
    other caller of this class, e.g. scripts/run_backtest.py's single fixed
    parameterization, is unaffected) and applies a Bonferroni correction to
    the resulting p-value: `p_adjusted = min(1, p_raw * N)`. This costs
    nothing extra (same n_sims backtest_fn calls as before) and is
    provably conservative — the true best-of-N p-value can only be smaller,
    never larger, so this cannot let through a candidate that a full
    multi-model test would have failed.
  - `run_multi(panel, backtest_fns)` (exact, expensive): actually resamples
    the max-of-N statistic by evaluating every candidate against the SAME
    shared randomized panel in each simulation and comparing the real best
    Sharpe to the distribution of randomized-best Sharpes. This is the
    textbook procedure, and less conservative than Bonferroni when the
    candidates are correlated (they usually are — adjacent grid points share
    most of their trades) — but it costs N times as many backtest_fn calls.
    Measured on this repo's pairs backtest (~0.11s/call): ~1 minute for
    n_sims=500 at N=1, ~30 minutes at N=32. Not wired into the default
    per-iteration loop for that reason; available for a slower, periodic,
    stricter validation pass.
"""
from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class RealityCheckConfig:
    n_sims: int = 500
    pass_threshold: float = 0.05
    marginal_threshold: float = 0.10
    seed: int | None = 42


@dataclass
class RealityCheckResult:
    real_sharpe: float
    random_sharpes: list = field(default_factory=list)
    n_sims: int = 0
    p_value: float = 1.0
    percentile_rank: float = 0.0
    verdict: str = "FAIL"
    mean_random_sharpe: float = 0.0
    std_random_sharpe: float = 0.0
    # Multiple-comparisons bookkeeping (see module docstring). Defaults keep
    # every pre-existing caller's output identical: n_candidates_searched=1
    # makes raw_p_value == p_value and method == "single".
    raw_p_value: float = 1.0
    n_candidates_searched: int = 1
    method: str = "single"
    run_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "real_sharpe": round(self.real_sharpe, 4),
            "mean_random_sharpe": round(self.mean_random_sharpe, 4),
            "std_random_sharpe": round(self.std_random_sharpe, 4),
            "p_value": round(self.p_value, 4),
            "raw_p_value": round(self.raw_p_value, 4),
            "n_candidates_searched": self.n_candidates_searched,
            "method": self.method,
            "percentile_rank": round(self.percentile_rank, 1),
            "verdict": self.verdict,
            "n_sims": self.n_sims,
            "run_at": self.run_at,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def print_summary(self) -> None:
        d = self.to_dict()
        print("\n== Reality Check (White's) ==")
        print(f"  Real Sharpe      : {d['real_sharpe']:+.4f}")
        print(f"  Random mean+-std : {d['mean_random_sharpe']:+.4f} +- {d['std_random_sharpe']:.4f}")
        if d["n_candidates_searched"] > 1:
            print(f"  Raw p-value      : {d['raw_p_value']:.4f}  "
                  f"({d['method']}-adjusted over {d['n_candidates_searched']} candidates)")
        print(f"  p-value          : {d['p_value']:.4f}")
        print(f"  Percentile rank  : {d['percentile_rank']:.1f}th")
        print(f"  Verdict          : {d['verdict']}")
        print("=" * 55 + "\n")


class RealityCheck:
    def __init__(
        self,
        backtest_fn: Callable[[pd.DataFrame], float],
        config: RealityCheckConfig | None = None,
    ) -> None:
        self._backtest_fn = backtest_fn
        self._cfg = config or RealityCheckConfig()
        self._rng = random.Random(self._cfg.seed)

    def run(self, price_panel: pd.DataFrame, n_candidates_searched: int = 1) -> RealityCheckResult:
        """Single-model Reality Check, optionally Bonferroni-adjusted.

        Args:
            price_panel: real price panel for the ONE candidate `backtest_fn`
                was built around (e.g. the WFO winner).
            n_candidates_searched: how many candidates competed to produce
                that winner (e.g. `len(param_grid)` including any UHAI
                suggestions). Default 1 reproduces the exact pre-existing
                behavior — every caller that does not pass this gets an
                unadjusted p-value, identical output to before this
                parameter existed.

        See the module docstring for why this Bonferroni correction, rather
        than the full `run_multi` resampling, is the default: it costs
        nothing extra and is provably conservative.
        """
        cfg = self._cfg
        real_sharpe = self._backtest_fn(price_panel)
        log.info("RealityCheck real Sharpe = %.4f", real_sharpe)

        random_sharpes: list[float] = []
        for i in range(cfg.n_sims):
            rand_panel = _phase_randomize_panel(price_panel, self._rng)
            random_sharpes.append(self._backtest_fn(rand_panel))
            if (i + 1) % max(1, cfg.n_sims // 20) == 0:
                log.info("RealityCheck: %d/%d simulations done", i + 1, cfg.n_sims)

        n = len(random_sharpes)
        beats = sum(1 for s in random_sharpes if s >= real_sharpe)
        raw_p_value = beats / n if n > 0 else 1.0
        pct_rank = sum(1 for s in random_sharpes if s < real_sharpe) / n * 100 if n else 0.0
        mean_r = sum(random_sharpes) / n if n else 0.0
        var_r = sum((s - mean_r) ** 2 for s in random_sharpes) / max(n - 1, 1)
        std_r = math.sqrt(var_r)

        n_candidates_searched = max(1, int(n_candidates_searched))
        p_value = min(1.0, raw_p_value * n_candidates_searched)
        method = "bonferroni" if n_candidates_searched > 1 else "single"

        verdict = "PASS" if p_value < cfg.pass_threshold else ("MARGINAL" if p_value < cfg.marginal_threshold else "FAIL")

        return RealityCheckResult(
            real_sharpe=real_sharpe,
            random_sharpes=random_sharpes,
            n_sims=n,
            p_value=p_value,
            raw_p_value=raw_p_value,
            n_candidates_searched=n_candidates_searched,
            method=method,
            percentile_rank=pct_rank,
            verdict=verdict,
            mean_random_sharpe=mean_r,
            std_random_sharpe=std_r,
        )

    def run_multi(
        self, price_panel: pd.DataFrame, backtest_fns: list[Callable[[pd.DataFrame], float]]
    ) -> RealityCheckResult:
        """Exact best-of-N Reality Check over every searched candidate.

        Each simulation's randomized panel is SHARED across all candidates
        (paired resampling) — every candidate is scored against the exact
        same random world in a given simulation, not its own independent
        one. This is what makes "compare the real best to the distribution
        of the randomized best" a valid test: it isolates the effect of
        picking the best of N choices from the effect of which random panel
        happened to be drawn.

        `self._backtest_fn` (constructor arg) is ignored here; every
        candidate comes from `backtest_fns` instead, including the winner —
        callers should pass a closure per grid candidate, not the winner
        alone. Cost is `len(backtest_fns)` times `run()`'s — see the module
        docstring for measured wall-clock numbers before pointing this at a
        large grid inside a tight loop.
        """
        cfg = self._cfg
        n_candidates = len(backtest_fns)
        if n_candidates == 0:
            raise ValueError("run_multi needs at least one candidate backtest_fn")

        real_sharpes = [fn(price_panel) for fn in backtest_fns]
        real_best_sharpe = max(real_sharpes)
        log.info("RealityCheck (multi, N=%d) real best Sharpe = %.4f",
                 n_candidates, real_best_sharpe)

        random_best_sharpes: list[float] = []
        for i in range(cfg.n_sims):
            rand_panel = _phase_randomize_panel(price_panel, self._rng)
            random_best_sharpes.append(max(fn(rand_panel) for fn in backtest_fns))
            if (i + 1) % max(1, cfg.n_sims // 20) == 0:
                log.info("RealityCheck (multi): %d/%d simulations done", i + 1, cfg.n_sims)

        n = len(random_best_sharpes)
        beats = sum(1 for s in random_best_sharpes if s >= real_best_sharpe)
        p_value = beats / n if n > 0 else 1.0
        pct_rank = sum(1 for s in random_best_sharpes if s < real_best_sharpe) / n * 100 if n else 0.0
        mean_r = sum(random_best_sharpes) / n if n else 0.0
        var_r = sum((s - mean_r) ** 2 for s in random_best_sharpes) / max(n - 1, 1)
        std_r = math.sqrt(var_r)

        verdict = "PASS" if p_value < cfg.pass_threshold else ("MARGINAL" if p_value < cfg.marginal_threshold else "FAIL")

        return RealityCheckResult(
            real_sharpe=real_best_sharpe,
            random_sharpes=random_best_sharpes,
            n_sims=n,
            p_value=p_value,
            raw_p_value=p_value,
            n_candidates_searched=n_candidates,
            method="full_bootstrap",
            percentile_rank=pct_rank,
            verdict=verdict,
            mean_random_sharpe=mean_r,
            std_random_sharpe=std_r,
        )


def _phase_randomize_series(prices: np.ndarray, rng: random.Random) -> np.ndarray:
    n = len(prices)
    if n < 4:
        return prices.copy()
    p = np.where(prices <= 0, 1e-10, prices)
    returns = np.log(p[1:] / p[:-1])
    x = np.fft.rfft(returns)
    amp = np.abs(x)
    phases = np.array([rng.uniform(-math.pi, math.pi) for _ in range(len(x))])
    rand_x = amp * np.exp(1j * phases)
    rand_returns = np.fft.irfft(rand_x, n=len(returns))
    rand_prices = np.empty(n)
    rand_prices[0] = prices[0]
    rand_prices[1:] = prices[0] * np.exp(np.cumsum(rand_returns))
    return rand_prices


def _phase_randomize_panel(panel: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    """Randomize each column independently (see module docstring)."""
    out = panel.copy()
    for col in panel.columns:
        series = panel[col].to_numpy(dtype=float)
        out[col] = _phase_randomize_series(series, rng)
    return out
