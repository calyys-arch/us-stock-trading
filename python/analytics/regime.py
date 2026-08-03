"""
Report-only market-regime diagnostic: a discrete-state Markov chain over a
rolling-return Bull/Bear/Sideways label.

Provenance: evaluated from an external quant write-up
(github.com/jackson-video-resources/markov-hedge-fund-method, reviewed
2026-07-29 on user request). That repo's own "install" doc is actually an
AI-agent prompt-injection script (auto-installs a Claude Code skill and
prints a promotional link) — NONE of its install/execution instructions
were followed. Only the underlying quant technique (observable Markov
regime chain) was evaluated and is reimplemented here from scratch against
this repo's own data pipeline and conventions.

Honesty contract (same standard as python/signals/trap_detector.py):
  - This is a well-known, simple regime-detection heuristic (rolling-return
    threshold -> 3-state label -> MLE transition-count matrix), NOT a
    validated trading edge. It is REPORT-ONLY: nothing in this module is
    wired to any strategy, order, or configs/strategy.yaml auto_execute
    flag, and calling it never places or filters a trade.
  - `illustrative_naive_backtest`'s Sharpe/drawdown are a NAIVE, COST-FREE,
    single-day sign-of-transition-matrix illustration — no slippage, no
    commission, no WFO/Monte Carlo/param_guard gating. It exists so a
    human reviewing the regime panel can see "is this even worth pursuing
    further" before anyone considers wiring regime labels into an actual
    strategy's entry filter. If that ever happens, the filter's parameters
    count against python/backtest/param_guard.py's Chan-discipline budget
    and must earn a GO through the same WFO/Monte Carlo pipeline as every
    other strategy in this repo — this module does not pre-judge that.
  - No lookahead: the transition matrix at "as of" day t is built ONLY
    from labels[0..t] (see `compute_regime_report`); `label_regimes` itself
    uses only trailing data (`close.pct_change(window)`).

Free parameters (2): `window` (rolling-return lookback, trading days),
`threshold` (symmetric Bull/Bear cutoff on that rolling return). Chan's
discipline does not apply to a report-only diagnostic with no
enabled/auto_execute strategy block — see the contract above for when it
WOULD start to apply.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Index 0/1/2 — matches the source write-up's internal ordering so anyone
# cross-checking against that repo doesn't hit a transposed matrix.
STATES = ["Bear", "Sideways", "Bull"]
_BEAR, _SIDEWAYS, _BULL = 0, 1, 2


def label_regimes(close: pd.Series, window: int = 20, threshold: float = 0.02) -> pd.Series:
    """Label each day Bull(2) / Bear(0) / Sideways(1) from the trailing
    `window`-day return. Bull: return > +threshold. Bear: return <
    -threshold. Sideways: otherwise. Leading `window` days (no full
    trailing return yet) are dropped, not labeled Sideways-by-default."""
    rolling_return = close.pct_change(window)
    labels = pd.Series(_SIDEWAYS, index=close.index, dtype="Int64")
    labels[rolling_return > threshold] = _BULL
    labels[rolling_return < -threshold] = _BEAR
    labels[rolling_return.isna()] = pd.NA
    return labels.dropna().astype(int)


def build_transition_matrix(labels: pd.Series) -> np.ndarray:
    """MLE (count + row-normalize) estimate of the 3x3 transition matrix.
    A state that never occurred in `labels` gets a uniform 1/3 row instead
    of an all-zero row, so the result always stays a valid row-stochastic
    matrix (matters for `stationary_distribution` / `n_step_forecast`,
    which assume that)."""
    n = len(STATES)
    counts = np.zeros((n, n), dtype=float)
    arr = labels.to_numpy()
    for i in range(len(arr) - 1):
        counts[arr[i], arr[i + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    P = np.full((n, n), 1.0 / n)
    nonzero = row_sums.flatten() > 0
    P[nonzero] = counts[nonzero] / row_sums[nonzero]
    return P


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Left eigenvector of P for eigenvalue 1, normalized to sum to 1
    (long-run regime mix under this transition matrix). Falls back to a
    uniform distribution if no eigenvalue is close to 1 (should not happen
    for a well-formed row-stochastic matrix, but a diagnostic tool must
    degrade honestly rather than return NaNs to a dashboard)."""
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    if abs(eigvals[idx] - 1.0) > 1e-6:
        return np.full(len(STATES), 1.0 / len(STATES))
    vec = np.real(eigvecs[:, idx])
    vec = np.abs(vec)
    total = vec.sum()
    if total <= 0:
        return np.full(len(STATES), 1.0 / len(STATES))
    return vec / total


def n_step_forecast(P: np.ndarray, n: int) -> np.ndarray:
    """Chapman-Kolmogorov: P^n is the n-step transition matrix."""
    return np.linalg.matrix_power(P, n)


def _signal_from_matrix(P: np.ndarray, current_state: int) -> float:
    """P(next=Bull|current) - P(next=Bear|current) — positive leans long,
    negative leans short, magnitude is NOT a validated conviction measure
    (see module docstring)."""
    return float(P[current_state, _BULL] - P[current_state, _BEAR])


def illustrative_naive_backtest(close: pd.Series, labels: pd.Series, min_train: int = 252) -> dict | None:
    """NAIVE illustration only — see module docstring. At each day t (t >=
    min_train): fit the transition matrix on labels[0..t-1] (no lookahead),
    take sign(signal) as a next-day position, hold one day, no costs.
    Returns None if there isn't enough history to say anything ({} would
    look like "zero trades happened" instead of "not enough data")."""
    daily_returns = close.pct_change().dropna()
    common_index = labels.index.intersection(daily_returns.index)
    labels = labels.loc[common_index]
    daily_returns = daily_returns.loc[common_index]

    if len(labels) < min_train + 30:
        return None

    strategy_returns = []
    for t in range(min_train, len(labels) - 1):
        p_t = build_transition_matrix(labels.iloc[:t])
        current_state = int(labels.iloc[t])
        position = float(np.sign(_signal_from_matrix(p_t, current_state)))
        strategy_returns.append(position * float(daily_returns.iloc[t + 1]))

    sr = np.array(strategy_returns, dtype=float)
    std = sr.std(ddof=1) if len(sr) > 1 else 0.0
    sharpe = float(sr.mean() / std * np.sqrt(252)) if std > 0 and np.isfinite(std) else float("nan")

    equity = (1.0 + sr).cumprod()
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min()) if len(drawdown) else float("nan")

    return {
        "sharpe_naive_no_cost": sharpe,
        "max_drawdown_naive_no_cost": max_dd,
        "n_days": int(len(sr)),
        "note": "Illustrative only: no slippage/commission, no WFO/Monte Carlo gating, "
                "single-day sign-of-transition-matrix position. NOT a validated strategy.",
    }


@dataclass
class RegimeReport:
    symbol: str
    as_of: str
    window: int
    threshold: float
    n_days_labeled: int
    current_state: str
    transition_matrix: dict[str, dict[str, float]]
    stationary_distribution: dict[str, float]
    recent_history: list[dict] = field(default_factory=list)
    naive_backtest: dict | None = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "as_of": self.as_of,
            "window": self.window,
            "threshold": self.threshold,
            "n_days_labeled": self.n_days_labeled,
            "current_state": self.current_state,
            "transition_matrix": self.transition_matrix,
            "stationary_distribution": self.stationary_distribution,
            "recent_history": self.recent_history,
            "naive_backtest": self.naive_backtest,
        }


def compute_regime_report(
    close: pd.Series,
    symbol: str,
    window: int = 20,
    threshold: float = 0.02,
    min_train: int = 252,
    recent_days: int = 90,
) -> RegimeReport:
    """Full report-only regime snapshot for one symbol's close series, as
    of the LAST bar in `close`. See module docstring for the no-lookahead
    / naive-backtest honesty contract."""
    labels = label_regimes(close, window=window, threshold=threshold)
    if labels.empty:
        raise ValueError(f"regime.compute_regime_report: no labelable history for {symbol} "
                          f"(need > {window} bars, got {len(close)})")

    P = build_transition_matrix(labels)
    pi = stationary_distribution(P)
    current_state = int(labels.iloc[-1])

    transition_matrix = {
        STATES[i]: {STATES[j]: round(float(P[i, j]), 6) for j in range(len(STATES))}
        for i in range(len(STATES))
    }
    stationary = {STATES[i]: round(float(pi[i]), 6) for i in range(len(STATES))}

    recent = labels.tail(recent_days)
    recent_history = [
        {"date": ts.strftime("%Y-%m-%d"), "state": STATES[int(v)]}
        for ts, v in recent.items()
    ]

    naive_backtest = illustrative_naive_backtest(close, labels, min_train=min_train)

    as_of = close.index[-1]
    as_of_str = as_of.strftime("%Y-%m-%d") if hasattr(as_of, "strftime") else str(as_of)

    return RegimeReport(
        symbol=symbol,
        as_of=as_of_str,
        window=window,
        threshold=threshold,
        n_days_labeled=int(len(labels)),
        current_state=STATES[current_state],
        transition_matrix=transition_matrix,
        stationary_distribution=stationary,
        recent_history=recent_history,
        naive_backtest=naive_backtest,
    )
