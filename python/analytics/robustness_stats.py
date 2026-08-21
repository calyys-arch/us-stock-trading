"""
Fold-resampling / block-bootstrap helpers for
`backtests/reports/regime_gate_robustness_report.md`.

WHY THIS IS NOT THE SAME AS `python/backtest/monte_carlo.py`
--------------------------------------------------------------
`MonteCarloValidator` resamples INDIVIDUAL daily returns (or trade P&Ls)
with replacement — an i.i.d. bootstrap that answers "how much of this
Sharpe ratio could be luck in the SEQUENCING of otherwise-fixed
observations". It does not answer, and was never meant to answer, a
different question this robustness round needs: "how sensitive is the WFO
fold-pass-ratio VERDICT to the SPECIFIC, arbitrary historical boundary that
happened to split IS from OOS?" Every distinct historical window in this
system has already been used at least once in this campaign
(regime_gate_robustness_report.md §0), so there is no fresh holdout window
available to answer that with new data. The two functions below are a
SUBSTITUTE, not an equivalent, for a true out-of-time holdout — see that
report for the explicit caveat.

  - `bootstrap_fold_pass_ratio`: resamples WHICH of the already-computed,
    already-causal WFO fold results count toward the pass ratio (sampling
    folds WITH REPLACEMENT), producing a distribution of the discrete
    pass-ratio metric under many alternative "which folds mattered" draws.
    This tests sensitivity to fold ASSIGNMENT, not to unseen data.
  - `moving_block_bootstrap_sharpe`: resamples contiguous BLOCKS (not single
    days) of an already-computed daily-return series, preserving each
    block's internal autocorrelation / volatility-clustering structure
    (which the i.i.d. Monte Carlo bootstrap destroys by construction),
    producing a Sharpe distribution under many alternative block orderings
    of the same underlying data.

Neither function manufactures new information about the future; both are
resampling checks on data this campaign has already spent, reported here
with that limitation stated plainly rather than implied to be more than
it is.
"""
from __future__ import annotations

import math
import random


def bootstrap_fold_pass_ratio(
    fold_passes: list[bool],
    min_pass_ratio: float,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """Resample `fold_passes` (one bool per already-computed WFO fold) WITH
    REPLACEMENT `n_boot` times, each time drawing `len(fold_passes)` folds
    and recomputing the pass ratio. Returns percentiles of the resampled
    pass-ratio distribution plus the fraction of draws that would clear
    `min_pass_ratio` (`configs/goal.yaml`'s `wfo.min_pass_folds_ratio`) —
    i.e. "if the folds that happened to pass/fail had come out in a
    different, equally-plausible combination, how often would the GO bar
    still be cleared?"

    Raises ValueError on empty input (never silently fabricates a
    distribution from nothing)."""
    if not fold_passes:
        raise ValueError("fold_passes must be non-empty")
    n = len(fold_passes)
    rng = random.Random(seed)
    ratios = []
    for _ in range(n_boot):
        draw = [rng.choice(fold_passes) for _ in range(n)]
        ratios.append(sum(draw) / n)
    ratios.sort()
    return {
        "n_folds": n,
        "observed_pass_ratio": sum(fold_passes) / n,
        "n_boot": n_boot,
        "pass_ratio_p5": _pctile(ratios, 0.05),
        "pass_ratio_p25": _pctile(ratios, 0.25),
        "pass_ratio_p50": _pctile(ratios, 0.50),
        "pass_ratio_p75": _pctile(ratios, 0.75),
        "pass_ratio_p95": _pctile(ratios, 0.95),
        "frac_boot_draws_clearing_bar": sum(1 for r in ratios if r >= min_pass_ratio) / n_boot,
        "min_pass_ratio_bar": min_pass_ratio,
    }


def moving_block_bootstrap_sharpe(
    daily_returns: list[float],
    block_size: int = 21,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Moving-block bootstrap (Kunsch 1989): assemble a synthetic series of
    the same total length as `daily_returns` by repeatedly drawing
    overlapping contiguous blocks of length `block_size` (default 21 trading
    days ~= one calendar month — the same monthly convention
    `trend_efficiency_gate.py`'s own `window` default uses) WITH
    REPLACEMENT, then compute its annualized Sharpe. Repeated `n_boot`
    times. Unlike an i.i.d. bootstrap (`MonteCarloValidator`), this
    preserves each block's internal autocorrelation structure, so it is a
    meaningfully different robustness check, not a duplicate of the
    existing Monte Carlo gate.

    Raises ValueError if there are fewer than `2 * block_size` observations
    (too few blocks for resampling to mean anything)."""
    n = len(daily_returns)
    if n < 2 * block_size:
        raise ValueError(f"need >= {2 * block_size} daily returns for block_size={block_size}, got {n}")
    rng = random.Random(seed)
    n_blocks_needed = math.ceil(n / block_size)
    max_start = n - block_size
    sharpes = []
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in range(n_blocks_needed):
            block_start = rng.randint(0, max_start)
            sample.extend(daily_returns[block_start:block_start + block_size])
        sample = sample[:n]
        sharpes.append(_annualized_sharpe(sample))
    sharpes.sort()
    return {
        "n_obs": n,
        "block_size": block_size,
        "n_boot": n_boot,
        "observed_sharpe": _annualized_sharpe(daily_returns),
        "sharpe_p5": _pctile(sharpes, 0.05),
        "sharpe_p25": _pctile(sharpes, 0.25),
        "sharpe_p50": _pctile(sharpes, 0.50),
        "sharpe_p75": _pctile(sharpes, 0.75),
        "sharpe_p95": _pctile(sharpes, 0.95),
        "frac_boot_draws_positive": sum(1 for s in sharpes if s > 0) / n_boot,
    }


def _annualized_sharpe(returns: list[float]) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    return (mean / std * math.sqrt(252)) if std > 0 else 0.0


def _pctile(sorted_values: list[float], frac: float) -> float:
    n = len(sorted_values)
    idx = frac * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    t = idx - lo
    return sorted_values[lo] * (1 - t) + sorted_values[hi] * t
