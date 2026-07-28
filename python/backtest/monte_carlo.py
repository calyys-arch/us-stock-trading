"""
Monte Carlo Validator — ported from forex-trading/python/backtest/monte_carlo.py.

Generalized to accept a plain sequence of trade P&Ls or period returns
(rather than only a forex BacktestReport.trades list), since this repo has
TWO very different backtest engines (event-driven pairs, vectorized daily
cross-sectional) that produce results in different shapes. Both feed the
exact same statistical core here — bootstrap-resample with replacement N
times to estimate the distribution of Sharpe / Calmar / drawdown / win-rate
/ profit-factor, answering "how much of this backtest result is luck?".

The 5th-percentile Sharpe is a conservative lower bound — if it remains
positive the strategy edge is likely not due to lucky sequencing alone
(same interpretation as the forex version).
"""
from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class PercentileSummary:
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float

    def to_dict(self) -> dict:
        return {"p5": round(self.p5, 4), "p25": round(self.p25, 4),
                "p50": round(self.p50, 4), "p75": round(self.p75, 4),
                "p95": round(self.p95, 4)}


@dataclass
class MonteCarloResult:
    n_trades: int
    n_sims: int
    run_at: str
    sharpe: PercentileSummary
    calmar: PercentileSummary
    max_drawdown: PercentileSummary
    win_rate: PercentileSummary
    total_pnl: PercentileSummary
    profit_factor: PercentileSummary
    prob_profitable: float

    def to_dict(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "n_sims": self.n_sims,
            "run_at": self.run_at,
            "prob_profitable": round(self.prob_profitable, 3),
            "sharpe": self.sharpe.to_dict(),
            "calmar": self.calmar.to_dict(),
            "max_drawdown": self.max_drawdown.to_dict(),
            "win_rate": self.win_rate.to_dict(),
            "total_pnl": self.total_pnl.to_dict(),
            "profit_factor": self.profit_factor.to_dict(),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        log.info("Monte Carlo result saved to %s", path)
        return path

    def print_summary(self) -> None:
        d = self.to_dict()
        print("\n== Monte Carlo Validation ==")
        print(f"  Trades/Periods  : {d['n_trades']}  Sims: {d['n_sims']}")
        print(f"  Prob profitable : {d['prob_profitable']:.1%}")
        print(f"  {'Metric':<18} {'p5':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8}")
        for key in ("sharpe", "calmar", "max_drawdown", "win_rate", "total_pnl", "profit_factor"):
            v = d[key]
            print(f"  {key:<18} {v['p5']:>8.3f} {v['p25']:>8.3f} {v['p50']:>8.3f} {v['p75']:>8.3f} {v['p95']:>8.3f}")
        print("=" * 55 + "\n")


class MonteCarloValidator:
    """Bootstrap Monte Carlo validator over a plain list of P&L/return values.

    Usage (event-driven pairs backtest):
        mc.run([t.net_pnl for t in pair_trades])

    Usage (vectorized cross-sectional backtest):
        mc.run(vector_result.daily_returns.tolist())
    """

    def __init__(self, n_sims: int = 1000, seed: int | None = 42) -> None:
        self._n_sims = n_sims
        self._rng = random.Random(seed)

    def run(self, pnl_series: list[float]) -> MonteCarloResult:
        if not pnl_series:
            log.warning("MonteCarloValidator: empty pnl_series — returning empty result")
            return _empty_result(self._n_sims)

        n = len(pnl_series)
        log.info("MonteCarloValidator: %d observations, n_sims=%d", n, self._n_sims)

        sharpes, calmars, drawdowns, win_rates, totals, pfs = [], [], [], [], [], []
        profitable_count = 0

        for _ in range(self._n_sims):
            sample = [self._rng.choice(pnl_series) for _ in range(n)]
            total = sum(sample)
            if total > 0:
                profitable_count += 1
            sharpes.append(_sharpe(sample))
            drawdowns.append(_max_drawdown(sample))
            calmars.append(_calmar(sample))
            win_rates.append(sum(1 for p in sample if p > 0) / n)
            totals.append(total)
            pfs.append(_profit_factor(sample))

        return MonteCarloResult(
            n_trades=n,
            n_sims=self._n_sims,
            run_at=datetime.utcnow().isoformat(),
            sharpe=_pctile(sharpes),
            calmar=_pctile(calmars),
            max_drawdown=_pctile(drawdowns),
            win_rate=_pctile(win_rates),
            total_pnl=_pctile(totals),
            profit_factor=_pctile(pfs),
            prob_profitable=profitable_count / self._n_sims,
        )


def _sharpe(pnls: list[float]) -> float:
    n = len(pnls)
    if n < 2:
        return 0.0
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    return (mean / std * math.sqrt(252)) if std > 0 else 0.0


def _max_drawdown(pnls: list[float]) -> float:
    equity, peak, dd = 0.0, 0.0, 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    return dd


def _calmar(pnls: list[float]) -> float:
    total = sum(pnls)
    dd = _max_drawdown(pnls)
    return total / dd if dd > 0 else (float("inf") if total > 0 else 0.0)


def _profit_factor(pnls: list[float]) -> float:
    wins = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    return wins / losses if losses > 0 else (float("inf") if wins > 0 else 0.0)


def _pctile(values: list[float]) -> PercentileSummary:
    s = sorted(values)
    n = len(s)

    def _p(frac: float) -> float:
        idx = frac * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        t = idx - lo
        return s[lo] * (1 - t) + s[hi] * t

    return PercentileSummary(p5=_p(0.05), p25=_p(0.25), p50=_p(0.50), p75=_p(0.75), p95=_p(0.95))


def _empty_result(n_sims: int) -> MonteCarloResult:
    z = PercentileSummary(0.0, 0.0, 0.0, 0.0, 0.0)
    return MonteCarloResult(
        n_trades=0, n_sims=n_sims, run_at=datetime.utcnow().isoformat(),
        sharpe=z, calmar=z, max_drawdown=z, win_rate=z, total_pnl=z, profit_factor=z,
        prob_profitable=0.0,
    )
