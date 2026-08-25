"""
Find a walk-forward gate that actually discriminates, and show what it costs.

The problem, measured in scripts/diagnose_gate_power.py: the shipped `wfo_go`
rule (per-fold `oos_sharpe >= is_sharpe * 0.5` and `>= 0.0`, then
`pass_ratio >= 0.50`) fires 42-58% of the time on a strategy with NO EDGE, at
every sample size tried. A fold's OOS Sharpe landing above zero is a coin
flip, and asking for half of eight coin flips is asking for nothing.

This script scores candidate replacements on the same two axes any test has:
false positive rate at true Sharpe 0, and power at a true Sharpe worth having.
Candidates are scored by simulation against the same fold geometries the repo
actually uses, so the answer is a measured trade-off rather than a preference.

The candidates, and why each exists:

  shipped          the current rule, as the baseline to beat.

  calibrated_      keep the count-the-folds shape, but set the required count
  breadth          from the BINOMIAL NULL instead of at 50%: the smallest k
                   where P(Binom(n_folds, 0.5) >= k) <= alpha. This is the
                   minimal honest fix — it makes the false positive rate a
                   design parameter rather than an accident.

  pooled_decay     stop testing folds one at a time. Concatenate every fold's
                   OOS days into one series and every fold's IS days into
                   another, then ask whether pooled OOS Sharpe retains a
                   fraction of pooled IS Sharpe. Same question the decay test
                   was asking, with n_folds times the sample behind it.

  breadth_and_     both of the above. Breadth catches "one fold carried the
  decay            whole result"; pooled decay catches "the edge was fitted".

A NOTE ON WHAT THIS GATE IS FOR. It should not become a significance test on
OOS returns, because `monte_carlo_p5_sharpe` already is one and two gates
asking the same question is one gate plus noise. Its distinct job is
PERSISTENCE: did the edge survive being taken out of the window its
parameters were chosen in. So the candidates test IS-vs-OOS consistency and
breadth across folds, not whether OOS beat zero significantly.

Three scenarios, because two of them cannot judge the decay test:

  null      true Sharpe 0 in both IS and OOS. Measures FALSE POSITIVES, and
            measures them exactly — at zero edge there is nothing to fit.

  real      true Sharpe POWER_AT in both. Measures POWER, as an UPPER BOUND:
            nothing here pays for parameter fitting, so a real strategy that
            had its parameters chosen on IS will do worse.

  overfit   IS inflated to OVERFIT_IS_SHARPE while OOS is truly 0 — what
            picking the best of K parameter sets on in-sample data produces,
            and the exact failure mode this project kept hitting (in-sample
            cost gates improving while out-of-sample ones did not). Measures
            the CATCH RATE. Without this scenario `pooled_decay` cannot be
            evaluated at all: the null and real scenarios contain no
            overfitting for it to detect, so it could only ever look like
            dead weight there.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_JSON = ROOT / "backtests/reports/wfo_gate_design.json"
OUT_MD = ROOT / "backtests/reports/wfo_gate_design.md"

TRADING_DAYS_PER_YEAR = 252

# (label, oos_days_per_fold, n_folds) — the two geometries in this repo.
GEOMETRIES = (
    ("intraday: 8 折 × 21 日 OOS", 21, 8),
    ("daily 10 年: 24 折 × 90 日 OOS", 90, 24),
)

# Power is quoted at this true annualized Sharpe: a realistic target for a
# real strategy, and the level the daily-horizon pivot is aiming at.
POWER_AT = 1.0
NULL_SHARPE = 0.0
# The overfit scenario: in-sample looks this good, out-of-sample is worth zero.
OVERFIT_IS_SHARPE = 2.0

# Allowed false positive rate for the calibrated candidates.
ALPHA = 0.10
# Pooled OOS Sharpe must retain this share of pooled IS Sharpe.
DECAY_KEEP = 0.5


@dataclass
class Fold:
    is_returns: np.ndarray
    oos_returns: np.ndarray


def _sharpe_ann(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    sd = float(x.std(ddof=1))
    return float(x.mean()) / sd * math.sqrt(TRADING_DAYS_PER_YEAR) if sd > 0 else 0.0


def _make_folds(sharpe_ann: float, oos_days: int, n_folds: int,
                rng: np.random.Generator,
                is_sharpe_ann: float | None = None) -> list[Fold]:
    """Folds whose IS and OOS Sharpe can differ, to model overfitting.

    `is_sharpe_ann=None` means honest folds (same distribution both sides).
    Setting it above `sharpe_ann` is what selecting parameters on in-sample
    data looks like from the outside.
    """
    scale = 1.0 / math.sqrt(TRADING_DAYS_PER_YEAR)
    mu_oos = sharpe_ann * scale
    mu_is = (sharpe_ann if is_sharpe_ann is None else is_sharpe_ann) * scale
    return [
        Fold(rng.normal(mu_is, 1.0, oos_days * 3), rng.normal(mu_oos, 1.0, oos_days))
        for _ in range(n_folds)
    ]


def min_folds_for_alpha(n_folds: int, alpha: float = ALPHA) -> int:
    """Smallest k with P(Binom(n_folds, 0.5) >= k) <= alpha.

    This is the whole fix in one line: the bar is set by what chance alone
    delivers, so the false positive rate becomes something chosen rather than
    something discovered afterwards.
    """
    for k in range(n_folds, -1, -1):
        if stats.binom.sf(k - 1, n_folds, 0.5) > alpha:
            return k + 1
    return 0


# ---- candidate gates: each takes folds, returns pass/fail ----

def gate_shipped(folds: list[Fold]) -> bool:
    passed = 0
    for f in folds:
        is_s, oos_s = _sharpe_ann(f.is_returns), _sharpe_ann(f.oos_returns)
        decay_ok = oos_s >= is_s * 0.5 if is_s > 0 else oos_s >= 0
        if decay_ok and oos_s >= 0.0:
            passed += 1
    return (passed / len(folds)) >= 0.50


def gate_calibrated_breadth(folds: list[Fold]) -> bool:
    k = min_folds_for_alpha(len(folds))
    positive = sum(1 for f in folds if f.oos_returns.sum() > 0)
    return positive >= k


def gate_pooled_decay(folds: list[Fold]) -> bool:
    pooled_is = np.concatenate([f.is_returns for f in folds])
    pooled_oos = np.concatenate([f.oos_returns for f in folds])
    is_s, oos_s = _sharpe_ann(pooled_is), _sharpe_ann(pooled_oos)
    if is_s <= 0:
        return oos_s > 0
    return oos_s >= is_s * DECAY_KEEP


def gate_breadth_and_decay(folds: list[Fold]) -> bool:
    return gate_calibrated_breadth(folds) and gate_pooled_decay(folds)


CANDIDATES = {
    "shipped": gate_shipped,
    "calibrated_breadth": gate_calibrated_breadth,
    "pooled_decay": gate_pooled_decay,
    "breadth_and_decay": gate_breadth_and_decay,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=3000)
    args = ap.parse_args()

    rng = np.random.default_rng(11)
    results = []
    for label, oos_days, n_folds in GEOMETRIES:
        k = min_folds_for_alpha(n_folds)
        print(f"== {label} ==")
        print(f"   校準門檻：{n_folds} 折要求 >= {k} 折為正 "
              f"(P(Binom({n_folds},0.5) >= {k}) = "
              f"{stats.binom.sf(k-1, n_folds, 0.5):.3f} <= {ALPHA})", flush=True)
        rows = {}
        for name, fn in CANDIDATES.items():
            fp = sum(
                fn(_make_folds(NULL_SHARPE, oos_days, n_folds, rng))
                for _ in range(args.reps)
            ) / args.reps
            power = sum(
                fn(_make_folds(POWER_AT, oos_days, n_folds, rng))
                for _ in range(args.reps)
            ) / args.reps
            # Overfit folds: IS looks great, OOS is worthless. A good gate
            # rejects these, so the figure of merit is 1 - pass rate.
            overfit_pass = sum(
                fn(_make_folds(NULL_SHARPE, oos_days, n_folds, rng,
                               is_sharpe_ann=OVERFIT_IS_SHARPE))
                for _ in range(args.reps)
            ) / args.reps
            rows[name] = {"false_positive": fp, "power": power,
                          "overfit_caught": 1.0 - overfit_pass,
                          "discrimination": power - fp}
            print(f"   {name:22s} 假陽性 {fp:5.1%}   "
                  f"檢定力@Sharpe{POWER_AT} {power:5.1%}   "
                  f"抓到配適 {1.0-overfit_pass:5.1%}   "
                  f"分辨力 {power-fp:+5.1%}", flush=True)
        results.append({
            "geometry": label, "oos_days_per_fold": oos_days,
            "n_folds": n_folds, "calibrated_k": k, "candidates": rows,
        })
        print(flush=True)

    payload = {
        "method": (
            "simulated folds of known true Sharpe; IS and OOS drawn from the "
            "same distribution, so power is an upper bound while the false "
            "positive column is exact"
        ),
        "reps": args.reps, "power_at_sharpe": POWER_AT, "alpha": ALPHA,
        "overfit_is_sharpe": OVERFIT_IS_SHARPE,
        "decay_keep": DECAY_KEEP, "geometries": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_render(payload), encoding="utf-8")
    print(f"Wrote {OUT_JSON}\nWrote {OUT_MD}")
    return 0


def _render(payload: dict) -> str:
    L = [
        "# 重新設計 `wfo_go`：要一道真的能分辨的閘門",
        "",
        f"現行判準在真實 Sharpe 0 時會亮 42–58%（見 `gate_power_diagnosis.md`）。",
        f"下表用模擬比較候選設計，兩個軸是任何檢定都要看的：",
        f"真實 Sharpe 0 的**假陽性率**，與真實 Sharpe {payload['power_at_sharpe']} 的**檢定力**。",
        f"每格 {payload['reps']} 次模擬。",
        "",
        "**這道閘門的職責。** 它不該變成 OOS 報酬的顯著性檢定 —— "
        "`monte_carlo_p5_sharpe` 已經是了，兩道問同一件事等於一道加噪音。"
        "它獨有的工作是**持續性**：邊緣有沒有在離開挑參數的那個視窗後存活。"
        "所以候選都在測 IS 對 OOS 的一致性與跨折的廣度，而不是 OOS 有沒有顯著大於零。",
        "",
        "**檢定力那一欄是上界**（模擬的 IS/OOS 同分佈，沒有為挑參數付代價）。"
        "假陽性那一欄是精確的 —— 真實 Sharpe 0 時沒有東西可以配適。",
        "",
    ]
    for g in payload["geometries"]:
        L.append(f"## {g['geometry']}")
        L.append("")
        L.append(f"校準後的折數門檻：{g['n_folds']} 折要求 ≥ **{g['calibrated_k']}** 折為正"
                 f"（由二項式虛無分佈定，α ≤ {payload['alpha']}）")
        L.append("")
        L.append("| 候選 | 假陽性率 | 檢定力 @ Sharpe 1.0 | 抓到過度配適 | 分辨力 |")
        L.append("|---|---:|---:|---:|---:|")
        for name, r in g["candidates"].items():
            L.append(f"| `{name}` | {r['false_positive']:.1%} | "
                     f"{r['power']:.1%} | {r['overfit_caught']:.1%} | "
                     f"{r['discrimination']:+.1%} |")
        L.append("")
    L.append("「抓到過度配適」是在 IS 年化 Sharpe "
             f"{payload['overfit_is_sharpe']}、OOS 真值 0 的折上拒絕的比例 —— "
             "也就是在 IS 挑最好參數所產生的樣子。這一欄是唯一能評價 "
             "`pooled_decay` 的欄位：null 與 real 兩個情境裡沒有配適可抓。")
    L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
