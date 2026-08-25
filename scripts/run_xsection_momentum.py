"""
Cross-sectional momentum, long-short: the first non-beta return source tried.

Everything this project has passed a gate with so far is long equity beta in
some form. Buy-and-hold, the 200-day overlay and the vol-targeted portfolio all
earned well through 2018-2026 and all collapsed together through 2006-2010,
because they are the same exposure with different brakes. A brake does not
make an engine.

A long-short cross-sectional strategy is structurally different: it buys the
strongest names and shorts the weakest in the SAME universe, so the market
move common to both sides cancels. Whether it cancels in practice is an
empirical question, and it is the question this script is really asking. The
headline number here is not Sharpe -- it is BETA AND CORRELATION against the
long-only portfolio. A market-neutral strategy that turns out to be 0.7
correlated to the market is just leveraged beta wearing a hedge, and its
attractive Sharpe over a bull decade would mean nothing.

Signal: 12-1 momentum, the standard formulation (Jegadeesh & Titman 1993;
Fama & French 2012 use the same skip). Rank each name by its return over the
trailing 12 months EXCLUDING the most recent month. The skip is not
decoration: the most recent month carries short-term reversal, which points
the opposite way and dilutes the signal if left in.

Construction: rebalance monthly, long the top quantile and short the bottom,
equal weight within each side, dollar neutral. Only names with a full lookback
of history at that date are ranked, so a name is never selected on a partial
window.

Costs are charged on realized turnover with the same one-way figure as the
other daily studies. Momentum rebalances monthly rather than daily, so cost
is small but not negligible -- and unlike the long-only work, BOTH sides trade.

Shorting is charged borrow via the shared fee model's rate. Hard-to-borrow
names would cost more; this universe is large-cap and liquid, so the easy-to
-borrow rate is the right default, and the stress arm doubles costs to show
what a worse assumption does.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.backtest.monte_carlo import MonteCarloValidator  # noqa: E402
from python.backtest.walk_forward import (  # noqa: E402
    WalkForwardOptimizer, WFOConfig,
)
from python.core.fees_equity import DEFAULT_ANNUAL_BORROW_RATE_EASY_TO_BORROW  # noqa: E402
from scripts.run_daily_baseline_gates import (  # noqa: E402
    _max_dd, _profit_factor, _sharpe, load_panel, TRADING_DAYS,
)
from scripts.run_vol_target_study import gross_returns  # noqa: E402

OUT_JSON = ROOT / "backtests/reports/xsection_momentum.json"
OUT_MD = ROOT / "backtests/reports/xsection_momentum.md"

WINDOW_START = "2018-06-01"
MAX_DD_LIMIT = -0.25
ONE_WAY_COST_BPS = 4.0
# Minimum names needed to form two meaningful sides. Below this the quantiles
# are one or two names and the result is stock picking, not a factor.
MIN_NAMES = 20


def momentum_scores(panel: pd.DataFrame, lookback: int, skip: int
                    ) -> pd.DataFrame:
    """Trailing `lookback`-day return, ending `skip` days ago.

    Using prices shifted by `skip` on BOTH ends leaves the most recent `skip`
    days out of the signal entirely, which is the point of the skip.
    """
    ref = panel.shift(skip)
    return ref / ref.shift(lookback) - 1.0


def momentum_returns(panel: pd.DataFrame, lookback: int = 252, skip: int = 21,
                     quantile: float = 0.2, rebalance_days: int = 21,
                     cost_bps: float = ONE_WAY_COST_BPS,
                     ) -> tuple[pd.Series, pd.DataFrame]:
    """Dollar-neutral long-short daily returns, and the weights used.

    Weights for day t are built from scores known at t-1 and applied to day
    t's return, so no position is informed by the move it captures.
    """
    scores = momentum_scores(panel, lookback, skip)
    rets = panel.pct_change(fill_method=None)
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)

    current = pd.Series(0.0, index=panel.columns)
    for i, date in enumerate(panel.index):
        if i % rebalance_days == 0:
            row = scores.loc[date].dropna()
            # A name needs a price today to be traded, and a score to be ranked.
            row = row[panel.loc[date, row.index].notna()]
            if len(row) >= MIN_NAMES:
                k = max(1, int(len(row) * quantile))
                ranked = row.sort_values()
                shorts, longs = ranked.index[:k], ranked.index[-k:]
                current = pd.Series(0.0, index=panel.columns)
                current[longs] = 0.5 / k
                current[shorts] = -0.5 / k
        weights.loc[date] = current.to_numpy()

    held = weights.shift(1).fillna(0.0)
    gross = (held * rets).sum(axis=1)

    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    trade_cost = turnover * (cost_bps / 10_000.0)
    # Borrow on the short leg, accrued daily on the shorted notional.
    short_notional = held.clip(upper=0.0).abs().sum(axis=1)
    borrow = short_notional * (DEFAULT_ANNUAL_BORROW_RATE_EASY_TO_BORROW
                               / TRADING_DAYS)
    return (gross - trade_cost - borrow).dropna(), held


def _beta_and_corr(strategy: pd.Series, market: pd.Series) -> tuple[float, float]:
    """Beta and correlation of the strategy against the long-only portfolio.

    This is the load-bearing measurement of the whole script: a long-short
    construction is only market neutral if it MEASURES as market neutral.
    """
    joined = pd.concat([strategy, market], axis=1).dropna()
    if len(joined) < 30:
        return float("nan"), float("nan")
    s, m = joined.iloc[:, 0], joined.iloc[:, 1]
    var = float(m.var(ddof=1))
    beta = float(s.cov(m) / var) if var > 0 else float("nan")
    return beta, float(s.corr(m))


def run_wfo(panel: pd.DataFrame, cost_bps: float = ONE_WAY_COST_BPS) -> dict:
    grid = [
        {"lookback": lb, "quantile": q}
        for lb in (126, 252) for q in (0.2, 0.3)
    ]

    def backtest_fn(start: datetime, end: datetime, params: dict) -> dict:
        lb = int(params["lookback"])
        # Momentum needs lookback + skip of history before it can rank at all,
        # so a fold sliced without warmup would score an empty strategy as 0.0.
        need = lb + 40
        warmup = panel.loc[panel.index < start].tail(need)
        window = panel.loc[(panel.index >= start) & (panel.index < end)]
        if len(window) < 30 or len(warmup) < need // 2:
            return {"sharpe_ratio": 0.0, "n_trades": 0, "daily_returns": []}
        net_all, held = momentum_returns(
            pd.concat([warmup, window]), lookback=lb,
            quantile=float(params["quantile"]), cost_bps=cost_bps)
        net = net_all.loc[net_all.index >= window.index[0]]
        if net.empty or (held.loc[net.index] != 0).to_numpy().sum() == 0:
            return {"sharpe_ratio": 0.0, "n_trades": 0, "daily_returns": []}
        rebalances = int((held.loc[net.index].diff().abs().sum(axis=1) > 1e-12).sum())
        return {
            "sharpe_ratio": _sharpe(net),
            "n_trades": max(rebalances, 1),
            "total_net_pnl": float(net.sum()),
            "profit_factor": _profit_factor(net),
            "max_drawdown": _max_dd(net),
            "daily_returns": [float(v) for v in net.tolist()],
        }

    cfg = WFOConfig(is_days=504, oos_days=126, step_days=126)
    result = WalkForwardOptimizer(backtest_fn, cfg, grid).run(
        panel.index[0].to_pydatetime(), panel.index[-1].to_pydatetime())

    pooled, dds, trades = [], [], []
    for f in result.folds:
        if not f.is_evaluable:
            continue
        pooled.extend((f.oos_metrics or {}).get("daily_returns") or [])
        dds.append(float((f.oos_metrics or {}).get("max_drawdown") or 0.0))
        trades.append(int((f.oos_metrics or {}).get("n_trades") or 0))
    series = pd.Series(pooled)
    return {
        "decision": result.decision,
        "total_folds": result.total_folds,
        "evaluable_folds": result.evaluable_folds,
        "positive_folds": result.positive_folds,
        "required_positive_folds": result.required_positive_folds,
        "chosen_params": [f.best_params for f in result.folds if f.is_evaluable],
        "pooled_oos_days": int(len(series)),
        "pooled_oos_sharpe": _sharpe(series) if len(series) > 1 else 0.0,
        "pooled_oos_profit_factor": _profit_factor(series) if len(series) else 0.0,
        "pooled_oos_returns": [float(v) for v in series.tolist()],
        "worst_fold_drawdown": min(dds) if dds else 0.0,
        "min_trades_in_a_fold": min(trades) if trades else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, default=2000)
    args = ap.parse_args()

    full = load_panel(min_rows=250)
    panel = full.loc[full.index >= pd.Timestamp("2016-06-01")]
    scored_from = pd.Timestamp(WINDOW_START)

    net, held = momentum_returns(panel)
    net = net.loc[net.index >= scored_from]
    market = gross_returns(panel).loc[net.index]

    beta, corr = _beta_and_corr(net, market)
    mc = MonteCarloValidator(n_sims=args.sims, seed=42).run(
        [float(v) for v in net.tolist()])
    n_names = int((held.loc[net.index] != 0).sum(axis=1).median())

    print(f"universe {panel.shape[1]} 檔，{len(net)} 個交易日 "
          f"({net.index[0].date()} -> {net.index[-1].date()})")
    print(f"每次持有中位數 {n_names} 檔（多空各半）\n")
    print("== 全窗 12-1 動量（多空、金額中性）==")
    print(f"   Sharpe {_sharpe(net):.2f}  "
          f"CAGR {(1+net).prod()**(TRADING_DAYS/len(net))-1:.1%}  "
          f"回撤 {_max_dd(net):.1%}  PF {_profit_factor(net):.2f}")
    print(f"   MC p5 Sharpe {float(mc.sharpe.p5):+.2f}")
    print(f"   對多頭組合: beta {beta:+.3f}  相關性 {corr:+.3f}   <-- 決定性數字")

    print("\n== 走步最佳化（新閘門規則）==")
    wfo = run_wfo(panel)
    stress = run_wfo(panel, cost_bps=ONE_WAY_COST_BPS * 1.5)
    pooled = pd.Series(wfo["pooled_oos_returns"])
    pooled_mc = MonteCarloValidator(n_sims=args.sims, seed=42).run(
        [float(v) for v in pooled.tolist()]) if len(pooled) > 1 else None
    print(f"   判決 {wfo['decision']}  {wfo['positive_folds']}/"
          f"{wfo['evaluable_folds']} 折為正（需 {wfo['required_positive_folds']}）")
    print(f"   彙總 OOS {wfo['pooled_oos_days']} 日  "
          f"Sharpe {wfo['pooled_oos_sharpe']:.2f}  "
          f"PF {wfo['pooled_oos_profit_factor']:.2f}")
    print(f"   最差單折回撤 {wfo['worst_fold_drawdown']:.1%}")
    print(f"   1.5x 成本 PF {stress['pooled_oos_profit_factor']:.2f}")

    gates = {
        "wfo_go": wfo["decision"] == "GO",
        "has_oos_trades": wfo["min_trades_in_a_fold"] > 0,
        "min_trades_per_oos_fold": wfo["min_trades_in_a_fold"] >= 40,
        "oos_drawdown_within_limit": wfo["worst_fold_drawdown"] >= MAX_DD_LIMIT,
        "cost_adjusted_profit_factor": wfo["pooled_oos_profit_factor"] >= 1.0,
        "monte_carlo_p5_sharpe": (
            float(pooled_mc.sharpe.p5) > 0.0 if pooled_mc else False),
        "stress_slippage_1.5x_pf_ge_1": stress["pooled_oos_profit_factor"] >= 1.0,
    }
    print("\n== 七道閘門（評在彙總樣本外報酬上）==")
    for g, ok in gates.items():
        print(f"   {'PASS' if ok else 'FAIL'}  {g}")
    print(f"\n   通過 {sum(gates.values())}/7")

    payload = {
        "run_at": datetime.utcnow().isoformat(),
        "n_symbols": int(panel.shape[1]),
        "window": [str(net.index[0].date()), str(net.index[-1].date())],
        "median_names_held": n_names,
        "full_window": {
            "sharpe": _sharpe(net),
            "cagr": float((1 + net).prod() ** (TRADING_DAYS / len(net)) - 1),
            "max_drawdown": _max_dd(net),
            "profit_factor": _profit_factor(net),
            "mc_p5_sharpe": float(mc.sharpe.p5),
            "beta_to_long_only": beta,
            "correlation_to_long_only": corr,
        },
        "wfo": {k: v for k, v in wfo.items() if k != "pooled_oos_returns"},
        "wfo_stress_1.5x_pf": stress["pooled_oos_profit_factor"],
        "pooled_oos_mc_p5": float(pooled_mc.sharpe.p5) if pooled_mc else None,
        "gates": gates,
        "gates_passed": sum(gates.values()),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_render(payload), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")
    return 0


def _render(p: dict) -> str:
    f, w = p["full_window"], p["wfo"]
    L = [
        "# 橫截面動量（12-1、多空、金額中性）",
        "",
        f"{p['n_symbols']} 檔宇宙，{p['window'][0]} → {p['window'][1]}，"
        f"每次持有中位數 {p['median_names_held']} 檔。",
        "",
        "**先看最後兩列。** 這個專案先前通過閘門的每一個東西都是多頭 beta 的變形，"
        "在 2018–2026 賺錢、在 2006–2010 一起垮。多空結構的意義在於對消市場走勢，"
        "而它有沒有真的對消是實證問題 —— 一個對市場 beta 0.7 的「中性」策略"
        "只是戴著避險名義的槓桿 beta，它在牛市十年的漂亮 Sharpe 不代表任何事。",
        "",
        "| 指標 | 值 |",
        "|---|---:|",
        f"| 年化 Sharpe | {f['sharpe']:.2f} |",
        f"| CAGR | {f['cagr']:.1%} |",
        f"| 最大回撤 | {f['max_drawdown']:.1%} |",
        f"| 獲利因子 | {f['profit_factor']:.2f} |",
        f"| MC p5 Sharpe | {f['mc_p5_sharpe']:+.2f} |",
        f"| **對多頭組合的 beta** | **{f['beta_to_long_only']:+.3f}** |",
        f"| **對多頭組合的相關性** | **{f['correlation_to_long_only']:+.3f}** |",
        "",
        "## 走步最佳化與七道閘門",
        "",
        f"判決 **{w['decision']}**，{w['positive_folds']}/{w['evaluable_folds']} "
        f"折為正（需 {w['required_positive_folds']}）。彙總樣本外 "
        f"{w['pooled_oos_days']} 日，Sharpe {w['pooled_oos_sharpe']:.2f}，"
        f"PF {w['pooled_oos_profit_factor']:.2f}；1.5x 成本下 PF "
        f"{p['wfo_stress_1.5x_pf']:.2f}。",
        "",
        "| 閘門 | 結果 |",
        "|---|---|",
    ]
    for g, ok in p["gates"].items():
        L.append(f"| `{g}` | {'PASS' if ok else 'FAIL'} |")
    L += [
        "",
        f"通過 **{p['gates_passed']}/7**。",
        "",
        "宇宙仍帶倖存者偏誤（2026 年的流動性快照），這對多空策略的影響尤其要留意："
        "被選進宇宙的條件是活到 2026 年，而動量策略做空的正是弱勢股，"
        "真實世界裡那些名字更可能退市或被併購。",
        "",
    ]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
