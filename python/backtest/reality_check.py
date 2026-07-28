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
    run_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "real_sharpe": round(self.real_sharpe, 4),
            "mean_random_sharpe": round(self.mean_random_sharpe, 4),
            "std_random_sharpe": round(self.std_random_sharpe, 4),
            "p_value": round(self.p_value, 4),
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

    def run(self, price_panel: pd.DataFrame) -> RealityCheckResult:
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
        p_value = beats / n if n > 0 else 1.0
        pct_rank = sum(1 for s in random_sharpes if s < real_sharpe) / n * 100 if n else 0.0
        mean_r = sum(random_sharpes) / n if n else 0.0
        var_r = sum((s - mean_r) ** 2 for s in random_sharpes) / max(n - 1, 1)
        std_r = math.sqrt(var_r)

        verdict = "PASS" if p_value < cfg.pass_threshold else ("MARGINAL" if p_value < cfg.marginal_threshold else "FAIL")

        return RealityCheckResult(
            real_sharpe=real_sharpe,
            random_sharpes=random_sharpes,
            n_sims=n,
            p_value=p_value,
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
