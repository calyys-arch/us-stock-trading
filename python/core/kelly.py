"""
Kelly Criterion capital allocation (Chan Ch.4, "Money and Risk Management").

Chan's core formula for a single bet with a Gaussian return assumption:

    f* = m / s^2

where m = expected excess return of the strategy (per period) and
s^2 = variance of that return. For MULTIPLE simultaneous strategies/bets
with a return covariance matrix, the vector form (Chan p.68, eq. 4.5) is:

    F* = C^-1 * M

where M is the vector of expected excess returns and C is the covariance
matrix of returns. F* gives the fraction of capital to allocate to each
strategy/pair SIMULTANEOUSLY (not one at a time), which is what makes it
different from just running the scalar formula per-strategy independently
— it accounts for cross-strategy correlation.

Chan's practical caveats, all implemented below as explicit parameters
(never silently skipped):
  - Full Kelly is extremely aggressive and assumes the estimated m, C are
    exactly correct; in practice estimation error means full Kelly typically
    OVER-bets. Chan recommends half-Kelly (f = F*/2) as a standard haircut
    (p.70).
  - Kelly can suggest leverage > 1.0 for a high-Sharpe strategy; this must
    still be capped by Reg T (2:1 intraday margin for a pattern day trader,
    4:1 with special approval) and by the RiskEngine's own hard leverage cap
    — Kelly is a capital-allocation optimum, not a risk-limit override.
  - If a strategy's estimated Sharpe was computed on a short/noisy sample,
    the resulting f* is unreliable; a max-single-strategy-allocation cap is
    applied to prevent one noisy edge estimate from dominating the book.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KellyResult:
    full_kelly_fractions: dict          # {name: raw f*_i}
    applied_fractions: dict             # {name: post-haircut, post-cap fraction actually used}
    kelly_multiplier: float             # e.g. 0.5 for half-Kelly
    max_single_fraction_cap: float
    gross_leverage: float               # sum(|applied_fractions|)


def kelly_fraction_single(
    expected_return: float,
    variance: float,
) -> float:
    """Scalar Kelly fraction f* = m / s^2 for one strategy (Chan eq. 4.2)."""
    if variance <= 0:
        return 0.0
    return expected_return / variance


def kelly_fractions_multi(
    expected_returns: dict,          # {name: m_i}, per-period excess return
    covariance: np.ndarray,          # covariance matrix, ordered per `names`
    names: list[str],
    kelly_multiplier: float = 0.5,   # half-Kelly by default (Chan p.70)
    max_single_fraction_cap: float = 0.25,
    max_gross_leverage: float = 2.0,  # Reg T day-trading buying-power cap
) -> KellyResult:
    """Vector Kelly F* = C^-1 M (Chan eq. 4.5), with half-Kelly haircut and
    two explicit safety caps: per-strategy allocation cap, and gross
    leverage cap (Reg T). Both caps are applied AFTER the haircut, and the
    gross-leverage cap scales ALL fractions down proportionally (preserves
    the relative Kelly-optimal mix) rather than truncating arbitrarily."""
    m = np.array([expected_returns[n] for n in names], dtype=float)
    c = np.asarray(covariance, dtype=float)

    if c.shape != (len(names), len(names)):
        raise ValueError(
            f"kelly_fractions_multi: covariance shape {c.shape} does not match "
            f"len(names)={len(names)}"
        )

    try:
        f_star = np.linalg.solve(c, m)
    except np.linalg.LinAlgError:
        # Singular covariance (e.g. two perfectly correlated strategies) —
        # fall back to the pseudo-inverse rather than crashing the allocator.
        f_star = np.linalg.pinv(c) @ m

    full = {n: float(f_star[i]) for i, n in enumerate(names)}

    haircut = {n: v * kelly_multiplier for n, v in full.items()}
    capped = {
        n: float(np.clip(v, -max_single_fraction_cap, max_single_fraction_cap))
        for n, v in haircut.items()
    }

    gross = sum(abs(v) for v in capped.values())
    if gross > max_gross_leverage and gross > 0:
        scale = max_gross_leverage / gross
        capped = {n: v * scale for n, v in capped.items()}
        gross = max_gross_leverage

    return KellyResult(
        full_kelly_fractions=full,
        applied_fractions=capped,
        kelly_multiplier=kelly_multiplier,
        max_single_fraction_cap=max_single_fraction_cap,
        gross_leverage=gross,
    )


def half_kelly_drawdown_capped(
    f_star: float,
    max_acceptable_drawdown: float = 0.20,
    kelly_implied_drawdown: float | None = None,
) -> float:
    """Chan p.71: full Kelly's expected max drawdown is approximately equal
    to the Kelly fraction itself for a Gaussian-return bet (i.e. betting full
    Kelly implies you should expect a drawdown roughly as large as f* at some
    point). If the caller supplies an estimated `kelly_implied_drawdown`
    (e.g. from a Monte Carlo simulation of the strategy's actual return
    distribution — see backtest/monte_carlo.py), scale `f_star` down so the
    implied drawdown does not exceed `max_acceptable_drawdown`."""
    implied_dd = kelly_implied_drawdown if kelly_implied_drawdown is not None else abs(f_star)
    if implied_dd <= 0:
        return f_star
    scale = min(1.0, max_acceptable_drawdown / implied_dd)
    return f_star * scale
