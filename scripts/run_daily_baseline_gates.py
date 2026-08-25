"""
Score the dumbest possible daily strategies against the real gates.

Why this runs BEFORE any momentum strategy is built. 19 intraday cells came
back NO-GO and it was never established that the gate set can say GO to
anything at all on a daily horizon. Building a strategy first and reading
another NO-GO would leave two explanations tangled together — a bad strategy,
or an apparatus that cannot pass anything — and that ambiguity is the actual
complaint about this whole study. So measure the floor first:

  buy_hold      equal-weight the universe, hold for ten years, rebalance
                never. Close to zero cost and no parameters to fit. If the
                gates cannot pass THIS, the gates are the finding.

  ma200_overlay hold each name only while it is above its own 200-day moving
                average, flat otherwise. The oldest trend rule there is, one
                parameter, a few dozen round trips per name per decade.

Neither is a proposal. They are a floor and a benchmark: anything built later
has to beat them after costs, or it is elaborate packaging for a moving
average.

Costs are charged with the same `round_trip_cost` the intraday study used, so
the edge-to-cost arithmetic is comparable across horizons rather than being
re-based on a friendlier model.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.backtest.monte_carlo import MonteCarloValidator  # noqa: E402
from python.backtest.walk_forward import WalkForwardOptimizer, WFOConfig  # noqa: E402
from python.core.fees_equity import round_trip_cost  # noqa: E402

HISTORY = ROOT / "data/history"
OUT_JSON = ROOT / "backtests/reports/daily_baseline_gates.json"
OUT_MD = ROOT / "backtests/reports/daily_baseline_gates.md"

CAPITAL = 1_000_000.0
TRADING_DAYS = 252
# Only names with the full 2016-06..2026-07 span, so no strategy is credited
# for a symbol that merely started late.
MIN_ROWS = 2500


def load_panel(min_rows: int = MIN_ROWS, directory: Path | None = None
               ) -> pd.DataFrame:
    """Wide close-price frame (index=date, columns=symbol) of full-span names.

    `directory` defaults to data/history; pass another to score the same
    machinery on a different dataset (e.g. data/history_gfc2008).
    """
    source = directory or HISTORY
    series = {}
    for path in sorted(source.glob("*.csv")):
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        if "date" not in cols or "close" not in cols:
            continue
        if len(df) < min_rows:
            continue
        s = pd.Series(
            df[cols["close"]].to_numpy(dtype=float),
            index=pd.to_datetime(df[cols["date"]]),
            name=path.stem,
        )
        series[path.stem] = s[~s.index.duplicated(keep="last")].sort_index()
    if not series:
        raise SystemExit(f"no symbols in {source} with >= {min_rows} rows")
    panel = pd.DataFrame(series).sort_index()
    return panel.dropna(how="all")


def _cost_bps_per_round_trip(price: float, adv_dollars: float,
                             notional: float, holding_days: float) -> float:
    """Round-trip cost in bps of notional, via the shared fee model."""
    shares = max(1, int(notional / max(price, 0.01)))
    bd = round_trip_cost(
        shares=shares, entry_price=price, exit_price=price, is_short=False,
        holding_days=holding_days, adv_dollars=adv_dollars,
        half_spread_bps=2.0,
    )
    return (bd.total / notional) * 10_000.0 if notional > 0 else 0.0


def _adv_dollars(panel: pd.DataFrame) -> float:
    """A single representative ADV, used only to size market impact.

    data/history CSVs carry volume, but this baseline holds so little turnover
    that impact is second-order; a fixed, deliberately CONSERVATIVE (small)
    ADV keeps impact from being silently zeroed, which is what passing
    adv_dollars=0 into round_trip_cost would do.
    """
    return 200_000_000.0


def strategy_returns(panel: pd.DataFrame, kind: str, ma_window: int = 200
                     ) -> tuple[pd.Series, int]:
    """Net daily portfolio returns and a round-trip count.

    Positions are equal-weight across whatever is held that day, and both
    strategies act on information available at the prior close: the position
    for day t is decided by data through t-1, so `shift(1)` is applied to the
    holding mask before it multiplies day t's return.
    """
    rets = panel.pct_change(fill_method=None)
    if kind == "buy_hold":
        hold = panel.notna().astype(float)
    elif kind == "ma200_overlay":
        ma = panel.rolling(ma_window, min_periods=ma_window).mean()
        hold = (panel > ma).astype(float).where(panel.notna(), 0.0)
    else:
        raise ValueError(kind)

    hold = hold.shift(1).fillna(0.0)
    n_held = hold.sum(axis=1)
    weights = hold.div(n_held.where(n_held > 0), axis=0).fillna(0.0)
    gross = (weights * rets).sum(axis=1)

    # Cost: a round trip is charged whenever a name's weight goes 0 -> held.
    entries = ((hold == 1) & (hold.shift(1) == 0)).sum().sum()
    n_round_trips = int(entries)
    held_days = float((hold == 1).sum().sum())
    avg_hold = held_days / max(n_round_trips, 1)
    price = float(panel.iloc[-1].dropna().median())
    per_trip_bps = _cost_bps_per_round_trip(
        price, _adv_dollars(panel), CAPITAL / max(int(n_held.max() or 1), 1),
        avg_hold,
    )
    # Charge each round trip against the whole portfolio on its entry day,
    # spread evenly rather than dated, which is enough for a floor estimate.
    total_cost_frac = (per_trip_bps / 10_000.0) * n_round_trips / max(
        int(n_held.max() or 1), 1)
    net = gross - (total_cost_frac / len(gross))
    return net.dropna(), n_round_trips


def _sharpe(x: pd.Series) -> float:
    sd = float(x.std(ddof=1))
    return float(x.mean()) / sd * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0


def _profit_factor(x: pd.Series) -> float:
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    return gains / losses if losses > 0 else float("inf")


def _max_dd(x: pd.Series) -> float:
    eq = (1.0 + x).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def run_wfo(panel: pd.DataFrame, kind: str) -> dict:
    """Walk-forward the one parameter there is, under the new gate rule."""
    grid = ([{"ma_window": w} for w in (100, 150, 200, 250)]
            if kind == "ma200_overlay" else [{}])

    def backtest_fn(start: datetime, end: datetime, params: dict) -> dict:
        ma_window = int(params.get("ma_window", 200))
        # Feed the moving average its warmup from BEFORE the window, then keep
        # only in-window days. Slicing first and computing second leaves a
        # 200-day average undefined across a 90-day OOS fold, so the mask is
        # all-zero, the fold returns exactly 0.0, and every lookback longer
        # than a fold silently scores NO-GO. Warmup is read-only history
        # ending before `start`, so it grants no look-ahead.
        warmup = panel.loc[panel.index < start].tail(ma_window * 2)
        window = panel.loc[(panel.index >= start) & (panel.index < end)]
        if len(window) < 30:
            return {"sharpe_ratio": 0.0, "n_trades": 0, "daily_returns": []}
        combined = pd.concat([warmup, window])
        net_all, n_trips = strategy_returns(combined, kind, ma_window)
        net = net_all.loc[net_all.index >= window.index[0]]
        if net.empty:
            return {"sharpe_ratio": 0.0, "n_trades": 0, "daily_returns": []}
        return {
            "sharpe_ratio": _sharpe(net),
            "n_trades": max(n_trips, 1),
            "total_net_pnl": float(net.sum() * CAPITAL),
            "daily_returns": [float(v) for v in net.tolist()],
        }

    cfg = WFOConfig(is_days=504, oos_days=126, step_days=126)
    result = WalkForwardOptimizer(backtest_fn, cfg, grid).run(
        panel.index[0].to_pydatetime(), panel.index[-1].to_pydatetime())
    return {
        "decision": result.decision,
        "total_folds": result.total_folds,
        "evaluable_folds": result.evaluable_folds,
        "positive_folds": result.positive_folds,
        "required_positive_folds": result.required_positive_folds,
        "pass_ratio_legacy_view": result.pass_ratio,
        "oos_sharpe_mean": result.oos_sharpe_mean,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, default=2000)
    args = ap.parse_args()

    panel = load_panel()
    print(f"universe: {len(panel.columns)} 檔  "
          f"{panel.index[0].date()} -> {panel.index[-1].date()}  "
          f"{len(panel)} 個交易日", flush=True)
    print("  " + ", ".join(panel.columns), flush=True)

    cells = {}
    for kind in ("buy_hold", "ma200_overlay"):
        net, n_trips = strategy_returns(panel, kind)
        mc = MonteCarloValidator(n_sims=args.sims, seed=42).run(
            [float(v) for v in net.tolist()])
        wfo = run_wfo(panel, kind)
        pf = _profit_factor(net)
        cells[kind] = {
            "n_days": int(len(net)),
            "n_round_trips": n_trips,
            "sharpe_annualized": _sharpe(net),
            "cagr": float((1 + net).prod() ** (TRADING_DAYS / len(net)) - 1),
            "max_drawdown": _max_dd(net),
            "profit_factor": pf,
            "mc_p5_sharpe": float(mc.sharpe.p5),
            "mc_median_sharpe": float(mc.sharpe.p50),
            "wfo": wfo,
            "gates": {
                "wfo_go": wfo["decision"] == "GO",
                "monte_carlo_p5_sharpe": float(mc.sharpe.p5) > 0.0,
                "cost_adjusted_profit_factor": pf >= 1.0,
                "oos_drawdown_within_limit": _max_dd(net) >= -0.25,
            },
        }
        c = cells[kind]
        print(f"\n== {kind} ==", flush=True)
        print(f"   Sharpe {c['sharpe_annualized']:.2f}  CAGR {c['cagr']:.1%}  "
              f"MaxDD {c['max_drawdown']:.1%}  PF {pf:.2f}  "
              f"往返次數 {n_trips}", flush=True)
        print(f"   MC p5 Sharpe {c['mc_p5_sharpe']:+.2f} "
              f"(中位數 {c['mc_median_sharpe']:+.2f})", flush=True)
        print(f"   WFO {wfo['decision']}: {wfo['positive_folds']}/"
              f"{wfo['evaluable_folds']} 折為正，需要 "
              f"{wfo['required_positive_folds']}；平均 OOS Sharpe "
              f"{wfo['oos_sharpe_mean']:.2f}", flush=True)
        print(f"   閘門 {c['gates']}", flush=True)

    payload = {
        "run_at": datetime.utcnow().isoformat(),
        "universe": list(panel.columns),
        "window": [str(panel.index[0].date()), str(panel.index[-1].date())],
        "n_trading_days": int(len(panel)),
        "capital": CAPITAL,
        "cells": cells,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_render(payload), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")
    return 0


def _render(p: dict) -> str:
    L = [
        "# 日線基準：閘門在這個視野上能不能通過任何東西",
        "",
        f"宇宙：{len(p['universe'])} 檔（{p['window'][0]} → {p['window'][1]}，"
        f"{p['n_trading_days']} 個交易日）",
        "",
        "`" + "`, `".join(p["universe"]) + "`",
        "",
        "這兩個策略都不是提案，是**地板和基準**：後面建的任何東西扣完成本後要打贏它們，",
        "否則就只是把移動平均包裝得比較複雜。",
        "",
        "| | `buy_hold` | `ma200_overlay` |",
        "|---|---:|---:|",
    ]
    a, b = p["cells"]["buy_hold"], p["cells"]["ma200_overlay"]
    rows = [
        ("年化 Sharpe", f"{a['sharpe_annualized']:.2f}", f"{b['sharpe_annualized']:.2f}"),
        ("CAGR", f"{a['cagr']:.1%}", f"{b['cagr']:.1%}"),
        ("最大回撤", f"{a['max_drawdown']:.1%}", f"{b['max_drawdown']:.1%}"),
        ("獲利因子", f"{a['profit_factor']:.2f}", f"{b['profit_factor']:.2f}"),
        ("往返次數（10 年）", f"{a['n_round_trips']}", f"{b['n_round_trips']}"),
        ("MC p5 Sharpe", f"{a['mc_p5_sharpe']:+.2f}", f"{b['mc_p5_sharpe']:+.2f}"),
        ("WFO 判決", a["wfo"]["decision"], b["wfo"]["decision"]),
        ("為正的折數",
         f"{a['wfo']['positive_folds']}/{a['wfo']['evaluable_folds']}"
         f"（需 {a['wfo']['required_positive_folds']}）",
         f"{b['wfo']['positive_folds']}/{b['wfo']['evaluable_folds']}"
         f"（需 {b['wfo']['required_positive_folds']}）"),
    ]
    for name, x, y in rows:
        L.append(f"| {name} | {x} | {y} |")
    L += [
        "",
        "## 必須連帶讀的警告",
        "",
        "這個宇宙是「在 2016 年就存在、且到 2026 年仍有十年完整日線」的名單，",
        "等於挑了倖存者，而且成分幾乎全是半導體與科技股 —— 2016–2026 是該類股史上",
        "最強的多頭之一。所以上表的絕對報酬**大部分是類股 beta 加上倖存者偏誤**，",
        "不是策略技巧。它們的用途是當基準，不是當結論。",
        "",
    ]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
