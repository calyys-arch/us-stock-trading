"""
How much true edge does each sample-limited gate actually DEMAND, at the
sample sizes this project has run them at?

Why this exists. 19 of 19 entry-hypothesis cells came back NO-GO, and the
report reads as a verdict on the signals. Part of it is not: three of the
seven gates are statistical tests, and a statistical test on a short sample
demands a large effect before it will fire. If the demanded effect is larger
than any real strategy of that class produces, the gate is not measuring the
strategy — it is measuring the window. That is a claim about arithmetic, so it
should be measured rather than argued.

Method. Simulate a strategy whose true annualized Sharpe is KNOWN, hand its
daily series to the REAL gate code (python.backtest.monte_carlo for p5, the
fold rule from python.backtest.walk_forward for wfo_go), and record how often
each gate passes. Sweep true Sharpe and sample size. The answer is a pass
probability, i.e. the gate's power curve.

Two regimes are compared, both taken from what this repo actually ran:

  intraday  239 trading days total; `configs/goal.yaml` gives the
            microstructure signals is_days=90 / oos_days=30 CALENDAR days, so
            each OOS fold holds ~21 trading days. Monte Carlo runs on days
            with ACTIVITY, and the observed counts in the run logs were far
            below 239 (5 for auction_reclaim_15m, 23 for auction_reclaim_5m
            on the expensive tier, 66 for vsa_no_demand_5m on the cheap
            tier), so several values are swept.

  daily     2,556 trading days (the 15 symbols in data/history that carry
            2016-06..2026-07), WFOConfig defaults is_days=504 /
            oos_days=126 calendar days, so ~90 trading days per OOS fold and
            ~24 folds. A position-holding strategy has P&L on essentially
            every day, so Monte Carlo sees the full 2,556.

WHAT THIS DELIBERATELY DOES NOT CLAIM. The simulation draws IS and OOS from
the SAME distribution, i.e. it models a strategy whose edge is genuinely
stable and whose parameters were not overfit. Real walk-forward picks
parameters on IS and pays for it in OOS. So every pass probability here is an
UPPER BOUND on what a real strategy of that true Sharpe achieves. It answers
"could this gate ever have passed", not "will my strategy pass".

It also says nothing about the two COST gates
(`cost_adjusted_profit_factor`, `stress_slippage_1.5x_pf_ge_1`). Those depend
on edge per trade against cost per trade, not on sample size, and no
simulation of returns can speak to them. They need the measured edge/cost
ratios in backtests/reports/slippage_decomposition.json.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.backtest.monte_carlo import MonteCarloValidator  # noqa: E402
from python.backtest.walk_forward import WFOConfig  # noqa: E402

OUT_JSON = ROOT / "backtests/reports/gate_power_diagnosis.json"
OUT_MD = ROOT / "backtests/reports/gate_power_diagnosis.md"

TRADING_DAYS_PER_YEAR = 252

# True annualized Sharpe values to sweep. 0.0 is the null (a gate that fires
# here is a false positive); 3.0 is roughly what the intraday runs would have
# needed by the back-of-envelope this script is checking.
SHARPE_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)

# (label, total_trading_days, oos_days_per_fold, n_folds, mc_observations)
REGIMES = (
    ("intraday (MC 在 23 個活躍日)", 239, 21, 8, 23),
    ("intraday (MC 在 66 個活躍日)", 239, 21, 8, 66),
    ("intraday (MC 在全部 239 日)", 239, 21, 8, 239),
    ("daily 10 年 (MC 在 2556 日)", 2556, 90, 24, 2556),
)


def _returns(n: int, sharpe_ann: float, rng: np.random.Generator) -> np.ndarray:
    """`n` iid daily returns whose TRUE annualized Sharpe is `sharpe_ann`.

    Sharpe is scale-free, so sigma is fixed at 1.0 and only the mean moves.
    """
    mu = sharpe_ann / math.sqrt(TRADING_DAYS_PER_YEAR)
    return rng.normal(mu, 1.0, size=n)


def _sharpe_ann(x: np.ndarray) -> float:
    """Same estimator as monte_carlo._sharpe, vectorized."""
    if x.size < 2:
        return 0.0
    sd = float(x.std(ddof=1))
    if sd <= 0:
        return 0.0
    return float(x.mean()) / sd * math.sqrt(TRADING_DAYS_PER_YEAR)


def _wfo_passes(
    sharpe_ann_true: float, oos_days: int, n_folds: int, cfg: WFOConfig,
    rng: np.random.Generator,
) -> bool:
    """Apply the real fold rule to simulated folds and return the verdict.

    Mirrors walk_forward._decide's per-fold test:
        decay_ok = oos >= is * (1 - max_sharpe_decay) if is > 0 else oos >= 0
        abs_ok   = oos >= min_oos_sharpe_abs
    IS is drawn 3x longer than OOS, matching is_days=90 vs oos_days=30.
    """
    passed = 0
    for _ in range(n_folds):
        is_sharpe = _sharpe_ann(_returns(oos_days * 3, sharpe_ann_true, rng))
        oos_sharpe = _sharpe_ann(_returns(oos_days, sharpe_ann_true, rng))
        if is_sharpe > 0:
            decay_ok = oos_sharpe >= is_sharpe * (1 - cfg.max_sharpe_decay)
        else:
            decay_ok = oos_sharpe >= 0
        if decay_ok and oos_sharpe >= cfg.min_oos_sharpe_abs:
            passed += 1
    return (passed / n_folds) >= cfg.min_pass_folds_ratio


def _mc_p5_passes(
    sharpe_ann_true: float, n_obs: int, n_sims: int, rng: np.random.Generator,
) -> tuple[bool, float]:
    """Run the REAL MonteCarloValidator and test its p5 Sharpe against 0."""
    series = _returns(n_obs, sharpe_ann_true, rng).tolist()
    # seed=None so repeated reps are not identical draws of the bootstrap
    result = MonteCarloValidator(n_sims=n_sims, seed=None).run(series)
    p5 = float(result.sharpe.p5)
    return p5 >= 0.0, p5


def _pass_rate(flags: list[bool]) -> float:
    return sum(1 for f in flags if f) / len(flags) if flags else 0.0


def _needed_sharpe(curve: dict[float, float], target: float) -> str:
    """Lowest swept true Sharpe whose pass rate reaches `target`."""
    for s in sorted(curve):
        if curve[s] >= target:
            return f"{s:.1f}"
    return f">{max(curve):.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=60,
                    help="simulated strategies per (regime, Sharpe) cell")
    ap.add_argument("--sims", type=int, default=300,
                    help="bootstrap draws inside each Monte Carlo run")
    args = ap.parse_args()

    cfg_defaults = WFOConfig()
    # goal.yaml relaxes the pass ratio for every strategy; use what actually ran.
    import yaml
    goal = yaml.safe_load((ROOT / "configs/goal.yaml").read_text(encoding="utf-8"))
    wfo_yaml = goal.get("wfo") or {}
    cfg = WFOConfig(
        min_pass_folds_ratio=float(wfo_yaml.get("min_pass_folds_ratio",
                                                cfg_defaults.min_pass_folds_ratio)),
        min_oos_sharpe_abs=float(wfo_yaml.get("min_oos_sharpe",
                                              cfg_defaults.min_oos_sharpe_abs)),
        max_sharpe_decay=float(wfo_yaml.get("max_sharpe_decay",
                                            cfg_defaults.max_sharpe_decay)),
    )
    print(f"fold rule: pass_ratio >= {cfg.min_pass_folds_ratio}, "
          f"oos_sharpe >= {cfg.min_oos_sharpe_abs}, "
          f"decay <= {cfg.max_sharpe_decay}", flush=True)

    rng = np.random.default_rng(7)
    results = []
    for label, total_days, oos_days, n_folds, mc_obs in REGIMES:
        wfo_curve, mc_curve, p5_medians = {}, {}, {}
        for sharpe in SHARPE_GRID:
            wfo_flags = [
                _wfo_passes(sharpe, oos_days, n_folds, cfg, rng)
                for _ in range(args.reps)
            ]
            mc_pairs = [
                _mc_p5_passes(sharpe, mc_obs, args.sims, rng)
                for _ in range(args.reps)
            ]
            wfo_curve[sharpe] = _pass_rate(wfo_flags)
            mc_curve[sharpe] = _pass_rate([f for f, _ in mc_pairs])
            p5_medians[sharpe] = statistics.median(p5 for _, p5 in mc_pairs)
            print(f"  {label} | true Sharpe {sharpe:.1f}: "
                  f"wfo_go {wfo_curve[sharpe]:.0%}  "
                  f"mc_p5 {mc_curve[sharpe]:.0%}  "
                  f"(p5 中位數 {p5_medians[sharpe]:+.2f})", flush=True)
        results.append({
            "regime": label,
            "total_trading_days": total_days,
            "oos_days_per_fold": oos_days,
            "n_folds": n_folds,
            "mc_observations": mc_obs,
            "wfo_go_pass_rate": wfo_curve,
            "mc_p5_pass_rate": mc_curve,
            "mc_p5_median": p5_medians,
            "sharpe_for_80pct_wfo": _needed_sharpe(wfo_curve, 0.80),
            "sharpe_for_80pct_mc": _needed_sharpe(mc_curve, 0.80),
        })
        print(flush=True)

    payload = {
        "method": (
            "simulated strategies of known true Sharpe scored by the real gate "
            "code; IS and OOS drawn from the same distribution, so every pass "
            "rate is an upper bound on a real (parameter-fitted) strategy"
        ),
        "reps_per_cell": args.reps,
        "bootstrap_sims": args.sims,
        "fold_rule": {
            "min_pass_folds_ratio": cfg.min_pass_folds_ratio,
            "min_oos_sharpe_abs": cfg.min_oos_sharpe_abs,
            "max_sharpe_decay": cfg.max_sharpe_decay,
        },
        "sharpe_grid": list(SHARPE_GRID),
        "regimes": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_render(payload), encoding="utf-8")
    print(f"Wrote {OUT_JSON}\nWrote {OUT_MD}")
    return 0


def _render(payload: dict) -> str:
    grid = payload["sharpe_grid"]
    L = [
        "# 閘門的檢定力：它們實際上要求多大的真實邊緣",
        "",
        "問題：19 格全數 NO-GO，其中有多少是在講訊號，有多少只是在講樣本量。",
        "",
        f"做法：模擬**真實年化 Sharpe 已知**的策略，把日報酬交給**真正的閘門程式碼**"
        f"（`python.backtest.monte_carlo` 算 p5、`walk_forward` 的折判準算 wfo_go），"
        f"記錄各閘門通過的頻率。每格 {payload['reps_per_cell']} 次模擬、"
        f"bootstrap {payload['bootstrap_sims']} 抽。",
        "",
        f"折判準（取自 `configs/goal.yaml`，即實際跑過的設定）："
        f"通過率 ≥ {payload['fold_rule']['min_pass_folds_ratio']}、"
        f"OOS Sharpe ≥ {payload['fold_rule']['min_oos_sharpe_abs']}、"
        f"衰減 ≤ {payload['fold_rule']['max_sharpe_decay']}。",
        "",
        "**這些數字是上界。** 模擬的 IS 與 OOS 抽自同一個分佈，也就是假設邊緣完全穩定、",
        "參數沒有過度配適。真實走步會在 IS 挑參數並在 OOS 付出代價，所以真實策略只會更低。",
        "本表回答的是「這道閘門有沒有可能通過」，不是「我的策略會不會通過」。",
        "",
        "本表也**完全不涉及兩道成本閘門**（`cost_adjusted_profit_factor`、",
        "`stress_slippage_1.5x_pf_ge_1`）。那兩道取決於每筆邊緣對每筆成本，與樣本量無關，",
        "報酬模擬對它們無話可說 —— 那要看 `slippage_decomposition.json` 的實測倍數。",
        "",
    ]
    for r in payload["regimes"]:
        L.append(f"## {r['regime']}")
        L.append("")
        L.append(f"- 總交易日 {r['total_trading_days']}；每折 OOS {r['oos_days_per_fold']} 日 × "
                 f"{r['n_folds']} 折；MC 觀測值 {r['mc_observations']}")
        L.append("")
        L.append("| 真實年化 Sharpe | " + " | ".join(f"{s:.1f}" for s in grid) + " |")
        L.append("|---|" + "---|" * len(grid))
        L.append("| `wfo_go` 通過率 | " + " | ".join(
            f"{r['wfo_go_pass_rate'][str(s)] if str(s) in r['wfo_go_pass_rate'] else r['wfo_go_pass_rate'][s]:.0%}"
            for s in grid) + " |")
        L.append("| `monte_carlo_p5_sharpe` 通過率 | " + " | ".join(
            f"{r['mc_p5_pass_rate'][str(s)] if str(s) in r['mc_p5_pass_rate'] else r['mc_p5_pass_rate'][s]:.0%}"
            for s in grid) + " |")
        L.append("| MC p5 中位數 | " + " | ".join(
            f"{(r['mc_p5_median'][str(s)] if str(s) in r['mc_p5_median'] else r['mc_p5_median'][s]):+.2f}"
            for s in grid) + " |")
        L.append("")
        L.append(f"- 要讓 `wfo_go` 有 80% 機會通過，真實 Sharpe 需 ≥ **{r['sharpe_for_80pct_wfo']}**")
        L.append(f"- 要讓 `monte_carlo_p5_sharpe` 有 80% 機會通過，真實 Sharpe 需 ≥ **{r['sharpe_for_80pct_mc']}**")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
