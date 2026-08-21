"""
Splits every replayable intraday cell's cost drag into SLIPPAGE vs COMMISSION,
and reports the genuinely pre-cost profit factor.

Why this exists: `python/backtest/intraday_engine.py`'s `metrics_from_report`
exposes `profit_factor_gross` and `total_costs`, and both have been read as
"pre-cost edge" and "what costs took". Neither is that.
`IntradayTrade.gross_pnl` is computed in `_close_position` from
`position.entry_price` and `exit_price`, and both of those are
`_slippage_price` outputs — so half-spread and participation impact are
already inside `gross_pnl`. `IntradayTrade.costs` is
`entry_commission + exit_commission` alone. The two published profit factors
therefore differ only by commission, which is why they track each other
within a few percent on every signal in the corpus.

That matters for retirement decisions. Several signals were retired on the
reasoning "profit_factor_gross < 1, so there is no edge to recover and cost
work cannot help". Measured on vsa_no_demand 5m over 2026-07, the truly
pre-cost profit factor was 1.471 while `profit_factor_gross` read 1.026 —
slippage had taken $1,950 of a $2,104 pre-cost result and commission $116.
A signal can therefore show `profit_factor_gross` well below 1 and still
have a real pre-cost edge. This script measures that for the whole corpus
instead of extrapolating one month onto it.

Method — two replays per cell, frozen params, no WFO and no
re-optimization:
  1. NORMAL: the engine's default cost model, i.e. whatever the cell's
     published numbers were produced under. Its net PnL and profit factor
     are compared against the stored values as a self-check; a mismatch is
     recorded rather than silently tolerated, since it would mean the
     replay is not reproducing the published run.
  2. ZERO-COST: `half_spread_bps=0`, `impact_bps_per_participation=0`,
     `commission_per_share=0`, `min_commission=0`.

Slippage is then `net_zero_cost - net_normal - commission_normal`, using the
NORMAL run's own commission figure (`total_costs`). This is exact rather
than modelled: verified against the 2026-07 vsa_no_demand numbers above
($2,104 - $37 - $116 = $1,951).

CAVEAT on the zero-cost replay: removing slippage changes fill PRICES, so
it can change which bars fill and which stops trigger. The two runs are
therefore not guaranteed to hold the same trade set, and `n_trades` is
reported for both so a divergence is visible instead of assumed away. The
decomposition is a comparison of two achievable paths, not an accounting
identity over one fixed path.

Cells excluded and why:
  * l2_absorption_1m, absorption_breakout_1m — their source reports
    (backtests/reports/_*_validation/A0_grid_full20.json) are per-fold
    re-optimized grid baselines with `params: {}`, so there is no single
    frozen parameter set to replay. Recovering one requires re-running
    their WFO.
  * the three daily cells — different engines with their own cost models;
    this decomposition is specific to intraday_engine's slippage path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from python.analytics.volume_route_policy import time_stop_for  # noqa: E402
from python.backtest.intraday_engine import IntradayBacktestConfig  # noqa: E402
from python.backtest.optimize import build_intraday_backtest_fn  # noqa: E402

GATE_REPORT = Path("backtests/reports/entry_hypothesis_gate_report.json")
STRATEGY_PATH = Path("configs/strategy.yaml")
OUT_JSON = Path("backtests/reports/slippage_decomposition.json")
OUT_MD = Path("backtests/reports/slippage_decomposition.md")

ZERO_COST = {
    "half_spread_bps": 0.0,
    "impact_bps_per_participation": 0.0,
    "commission_per_share": 0.0,
    "min_commission": 0.0,
}

# Cheapest first, so partial progress is still useful if the tail is slow.
# sweep_reclaim_1m (104k fills) is deliberately last.
CELL_ORDER = (
    "auction_reclaim_15m",
    "vsa_effort_15m",
    "auction_reclaim_5m",
    "vsa_no_demand_5m",
    "vp_breakout_1m",
    "obv_divergence_5m",
    "vwap_band_fade_1m",
    "fvg_retest_1m",
    "orb_vwap_regime_1m",
    "orb_vwap_1m",
    "sweep_reclaim_1m",
)

# Widest warmup any listed signal needs (orb_vwap_regime: 35), used only to
# decide how much history to load; each cell still passes its own value.
_MAX_WARMUP_DAYS = 40


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _run(bars, signal_name, base_cfg, params, minutes, warmup_days, start_ts, end_ts, **cost_kw):
    cfg = IntradayBacktestConfig(
        chart_minutes=max(minutes, 1),
        time_stop_minutes=time_stop_for(max(minutes, 1)),
        **cost_kw,
    )
    fn = build_intraday_backtest_fn(
        bars, signal_name, base_cfg, engine_cfg=cfg, warmup_days=warmup_days,
    )
    metrics = fn(start_ts.to_pydatetime(), end_ts.to_pydatetime(), params)
    return {k: v for k, v in metrics.items() if k != "daily_returns"}


def _fmt(value, spec: str = ".3f", dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _render_md(payload: dict) -> str:
    L: list[str] = []
    L.append("# 滑價 vs 佣金拆解 — 真正的稅前邊緣")
    L.append("")
    L.append(f"- 產生時間：{payload['generated_at']}")
    L.append(f"- 視窗：{payload['window']}")
    L.append(f"- 方法：凍結參數、單路徑重播兩次（正常成本 / 完全零成本），無 WFO、無重新最佳化")
    L.append("")
    L.append("`profit_factor_gross` 是 **pre-commission**，不是 pre-cost："
             "`IntradayTrade.gross_pnl` 由 `_slippage_price` 過的成交價算出，"
             "`costs` 只含佣金。所以下表的「稅前 PF」才是真正關掉滑價後的數字。")
    L.append("")
    L.append("| 格子 | 筆數（正常／零成本） | published PF | 稅前 PF | 正常淨額 | 滑價 | 佣金 | 稅前淨額 | 每筆邊緣 | 每筆成本 | 倍數 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in payload["results"]:
        if r.get("error"):
            L.append(f"| `{r['cell']}` | 失敗 | | | | | | | | | |")
            continue
        L.append(
            f"| `{r['cell']}` | {r['n_trades_normal']} / {r['n_trades_zero']} | "
            f"{_fmt(r['pf_normal'])} | **{_fmt(r['pf_pre_cost'])}** | "
            f"{_fmt(r['net_normal'], ',.0f')} | {_fmt(r['slippage'], ',.0f')} | "
            f"{_fmt(r['commission'], ',.0f')} | {_fmt(r['net_pre_cost'], ',.0f')} | "
            f"{_fmt(r['edge_per_trade'], ',.2f')} | {_fmt(r['cost_per_trade'], ',.2f')} | "
            f"{_fmt(r['edge_cost_ratio'], '.2f')} |"
        )
    L.append("")
    L.append("「倍數」= 每筆稅前邊緣 ÷ 每筆總成本。壓力閘門把滑價乘 1.5，"
             "所以倍數需要在滑價那一塊上留餘裕才可能存活。")
    L.append("")
    for r in payload["results"]:
        L.append(f"## `{r['cell']}`")
        L.append("")
        if r.get("error"):
            L.append(f"- 執行失敗：`{r['error']}`")
            L.append("")
            continue
        L.append(f"- 訊號 `{r['signal']}`，{r['chart_minutes']}m 圖，warmup {r['warmup_days']} 天")
        L.append(f"- 凍結參數：`{json.dumps(r['frozen_params'], sort_keys=True)}`")
        chk = r["reproduction_check"]
        if chk.get("matches") is False:
            L.append(f"- **重現不一致**：published net {_fmt(chk.get('published_net'), ',.0f')} / "
                     f"PF {_fmt(chk.get('published_pf'))} vs 重播 net {_fmt(r['net_normal'], ',.0f')} / "
                     f"PF {_fmt(r['pf_normal'])} — 此格結論需先解釋差異")
        elif chk.get("matches") is True:
            L.append(f"- 重現核對：與 published 一致（net、PF 皆在容差內）")
        else:
            L.append("- 重現核對：published 無可比數字")
        L.append(f"- 滑價佔總成本 {_fmt(r['slippage_share_of_cost'], '.1%')}；"
                 f"總成本佔稅前淨額 {_fmt(r['cost_share_of_pre_cost'], '.1%')}")
        verdict = r["verdict"]
        L.append(f"- **判讀**：{verdict}")
        L.append("")
    return "\n".join(L) + "\n"


def _verdict(pf_pre_cost, pf_normal, ratio) -> str:
    if pf_pre_cost is None:
        return "無法判讀（零成本重播沒有成交）"
    if pf_pre_cost < 1.0:
        return ("稅前就沒有邊緣 — 關掉全部滑價與佣金仍然虧錢。"
                "退役結論成立，成本與執行的槓桿救不了這格。")
    if pf_normal is not None and pf_normal >= 1.0:
        return "稅前與成本後都有邊緣；問題不在成本。"
    if ratio is not None and ratio >= 1.5:
        return ("**稅前有邊緣、被成本吃掉，而且每筆邊緣對成本仍有餘裕** — "
                "原本以「沒有毛邊緣」為由的退役結論在這格不成立，執行成本是主要瓶頸。")
    return ("**稅前有邊緣但被成本吃掉**，且每筆邊緣與每筆成本相當 — "
            "退役理由需要改寫：這不是「沒有邊緣」，是「邊緣小於執行成本」。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default=",".join(CELL_ORDER))
    parser.add_argument("--start", default="2025-08-01")
    parser.add_argument("--end", default="2026-07-01")
    args = parser.parse_args()

    report = json.loads(GATE_REPORT.read_text(encoding="utf-8"))
    cells = report.get("cells") or {}
    strategy = _load_yaml(STRATEGY_PATH)

    from run_intraday_backtest import SIGNAL_WARMUP_DAYS, _load_real_bars
    from python.data.fixed_universe import load_universe_config

    universe = load_universe_config()
    load_start = str((pd.Timestamp(args.start) - pd.Timedelta(days=_MAX_WARMUP_DAYS)).date())
    print(f"loading 1m bars [{load_start}, {args.end}) for {len(universe['symbols'])} symbols...",
          flush=True)
    bars = _load_real_bars(universe["symbols"], load_start, args.end)
    if not bars:
        raise SystemExit("no cached 1m bars")
    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)

    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "window": f"{start_ts.date()} .. {end_ts.date()} (end-exclusive)",
        "method": (
            "frozen params, two single-path replays (normal cost vs zero cost); "
            "slippage = net_zero_cost - net_normal - commission_normal"
        ),
        "excluded": {
            "l2_absorption_1m / absorption_breakout_1m": (
                "source reports are per-fold re-optimized grid baselines with "
                "params: {} — no frozen parameter set to replay"
            ),
            "daily cells": "different engines with their own cost models",
        },
        "results": [],
    }

    for key in [c.strip() for c in args.cells.split(",") if c.strip()]:
        cell = cells.get(key)
        if not cell or not cell.get("candidate_params"):
            print(f"  !! {key}: no frozen params — skipped", flush=True)
            continue
        signal_name = cell["signal"]
        params = dict(cell["candidate_params"])
        minutes = int(cell["chart_minutes"])
        warmup = SIGNAL_WARMUP_DAYS.get(signal_name, 1)
        stored = cell.get("full_window_metrics") or {}

        print(f"\n=== {key} ({signal_name}, {minutes}m, warmup {warmup}d) ===", flush=True)
        row: dict = {
            "cell": key, "signal": signal_name, "chart_minutes": minutes,
            "warmup_days": warmup, "frozen_params": params,
        }
        try:
            normal = _run(bars, signal_name, strategy[signal_name], params,
                          minutes, warmup, start_ts, end_ts)
            print(f"    normal   n={normal['n_trades']:6d} PF={normal['profit_factor']:.3f} "
                  f"net=${normal['total_net_pnl']:>14,.0f} commission=${normal['total_costs']:>12,.0f}",
                  flush=True)
            zero = _run(bars, signal_name, strategy[signal_name], params,
                        minutes, warmup, start_ts, end_ts, **ZERO_COST)
            print(f"    zerocost n={zero['n_trades']:6d} PF={zero['profit_factor']:.3f} "
                  f"net=${zero['total_net_pnl']:>14,.0f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    !! {row['error']}", flush=True)
            payload["results"].append(row)
            _persist(payload)
            continue

        net_normal = float(normal["total_net_pnl"])
        commission = float(normal["total_costs"])
        net_pre_cost = float(zero["total_net_pnl"])
        slippage = net_pre_cost - net_normal - commission
        total_cost = slippage + commission
        n_zero = int(zero["n_trades"])
        pf_pre_cost = zero["profit_factor"] if n_zero else None

        published_net = stored.get("total_net_pnl")
        published_pf = stored.get("profit_factor")
        if published_net is None or published_pf is None:
            matches = None
        else:
            matches = (
                abs(net_normal - float(published_net)) <= max(1.0, 0.01 * abs(float(published_net)))
                and abs(float(normal["profit_factor"]) - float(published_pf)) <= 0.01
            )

        edge_per_trade = (net_pre_cost / n_zero) if n_zero else None
        cost_per_trade = (total_cost / n_zero) if n_zero else None
        ratio = (
            edge_per_trade / cost_per_trade
            if (edge_per_trade is not None and cost_per_trade not in (None, 0))
            else None
        )

        row.update({
            "n_trades_normal": int(normal["n_trades"]),
            "n_trades_zero": n_zero,
            "pf_normal": normal["profit_factor"],
            "pf_pre_commission_published": stored.get("profit_factor_gross"),
            "pf_pre_cost": pf_pre_cost,
            "net_normal": net_normal,
            "net_pre_cost": net_pre_cost,
            "slippage": slippage,
            "commission": commission,
            "total_cost": total_cost,
            "slippage_share_of_cost": (slippage / total_cost) if total_cost else None,
            "cost_share_of_pre_cost": (total_cost / net_pre_cost) if net_pre_cost else None,
            "edge_per_trade": edge_per_trade,
            "cost_per_trade": cost_per_trade,
            "edge_cost_ratio": ratio,
            "reproduction_check": {
                "published_net": published_net,
                "published_pf": published_pf,
                "matches": matches,
            },
            "metrics_normal": normal,
            "metrics_zero_cost": zero,
        })
        row["verdict"] = _verdict(pf_pre_cost, normal["profit_factor"], ratio)
        print(f"    -> 稅前 PF {_fmt(pf_pre_cost)} | 滑價 ${slippage:,.0f} | 佣金 ${commission:,.0f} | "
              f"每筆邊緣/成本 {_fmt(ratio, '.2f')}", flush=True)
        if matches is False:
            print(f"    !! 重現不一致: published net ${float(published_net):,.0f} "
                  f"PF {float(published_pf):.3f}", flush=True)
        payload["results"].append(row)
        _persist(payload)

    _persist(payload)
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")
    return 0


def _persist(payload: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_render_md(payload), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
