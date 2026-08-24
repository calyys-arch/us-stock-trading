"""
Does restricting the universe to tight-spread symbols raise the edge/cost ceiling?

The one cost term position size cannot escape. Execution cost splits into a
half-spread piece (linear in size) and a participation-impact piece (quadratic
in size). Shrinking the position shrinks the second and leaves the first, which
is why `ceiling = edge_bps / spread_round_trip_bps` is an asymptote: below 1.0
no position size clears PF 1.0. Every attempt so far has pushed on size, i.e.
on the escapable half.

The unexamined half. `IntradayBacktestConfig.half_spread_bps` defaults to a
FLAT 2.0 bps for every symbol, and the whole 16-cell matrix was run that way.
But backtests/reports/calibrated_spreads.json, measured from captured L2 depth,
puts the fixed universe's real half-spreads between 0.33 bps (AAPL) and 5.36
bps (STX) — a 16x spread, median 1.73. So the flat constant is roughly right on
AVERAGE while being wrong per symbol in both directions, and the spread term is
not a property of "the market" but of each market we chose to trade.

The hypothesis this tests. If per-trade edge is roughly independent of a
symbol's spread, then dropping the expensive names cuts the denominator of the
ceiling without cutting the numerator, and every cell's ceiling rises by the
ratio of the two universes' spreads. Restricting to the 8 tightest names should
take the measured round trip from ~2.9 bps toward ~1.1 bps.

The hypothesis that would kill it. Edge and spread may be the SAME thing seen
twice: a wide quote can be the compensation for providing liquidity, in which
case the expensive names carry the edge and dropping them drops it too. That is
why this runs the expensive tier as well as the cheap one. If edge_bps rises
with spread roughly in proportion, the ceiling is flat across tiers and the
lever does not exist.

Method, and why it is cheap. Two replays per tier — zero-cost (for edge) and
spread-only (for the spread denominator) — reusing
python/analytics/execution_ceiling.py so the arithmetic matches the published
figures exactly. Spread-only uses the CALIBRATED per-symbol spreads rather than
the flat constant, since the point is to measure what the real quotes cost.
Six replays total against the ~4.5 hours a WFO takes.

What this cannot settle. Trade count falls with the universe, and the Monte
Carlo p5 gate already needs an annualized Sharpe near 3.0 over a 239-day
window; a tier that wins on cost while losing 60% of its sample may trade a
cost problem for a power problem. Both numbers are reported side by side rather
than collapsed into one verdict. The depth calibration also rests on 8 days,
which is thin for a level even if spread RANKINGS are usually persistent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from python.analytics.execution_ceiling import (  # noqa: E402
    edge_bps_from_zero,
    spread_bps_from_runs,
)
from python.analytics.volume_route_policy import time_stop_for  # noqa: E402
from python.backtest.intraday_engine import (  # noqa: E402
    IntradayBacktestConfig,
    IntradayBacktestReport,
    metrics_from_report,
    run_intraday_backtest,
)

GATE_REPORT = Path("backtests/reports/entry_hypothesis_gate_report.json")
SPREADS = Path("backtests/reports/calibrated_spreads.json")
OUT_JSON = Path("backtests/reports/spread_tier_study.json")
OUT_MD = Path("backtests/reports/spread_tier_study.md")

# Cells with a published tuned ceiling to compare against. auction_reclaim_5m
# is the only cell in the matrix whose cost-adjusted PF exceeds 1.0 at full
# size, so it is the one where a cheaper universe could plausibly matter for a
# decision rather than only for a ranking.
CELLS = ("auction_reclaim_5m", "vsa_no_demand_5m")

TIER_SIZE = 8


def _run(bars, signal_name, sig_params, cfg, warmup_days, start_ts, end_ts) -> dict:
    """One single-path replay, mirroring optimize.build_intraday_backtest_fn's
    warmup slice and exit-time filter so figures stay comparable to the
    published full-window numbers, but keeping trades so notional is available."""
    warmup_start = start_ts - pd.Timedelta(days=warmup_days)
    sliced = {
        symbol: window
        for symbol, bars_df in bars.items()
        if not (window := bars_df.loc[
            (bars_df.index >= warmup_start) & (bars_df.index < end_ts)
        ]).empty
    }
    report = run_intraday_backtest(sliced, signal_name, sig_params, cfg)
    in_window = [t for t in report.trades if start_ts <= t.exit_time < end_ts]
    metrics = metrics_from_report(IntradayBacktestReport(trades=in_window), cfg.capital)
    notionals = [t.shares * t.entry_price for t in in_window]
    return {
        "n_trades": int(metrics["n_trades"]),
        "profit_factor": metrics["profit_factor"],
        "total_net_pnl": float(metrics["total_net_pnl"]),
        "commission": float(metrics["total_costs"]),
        "avg_notional": (sum(notionals) / len(notionals)) if notionals else None,
    }


def _fmt(value, spec: str = ".2f", dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _tiers(spreads: dict) -> dict[str, list[str]]:
    ranked = sorted(spreads.items(), key=lambda kv: kv[1])
    return {
        "cheap": [s for s, _ in ranked[:TIER_SIZE]],
        "expensive": [s for s, _ in ranked[-TIER_SIZE:]],
        "all": [s for s, _ in ranked],
    }


def _render(payload: dict) -> str:
    L = [
        "# 價差分層研究：便宜的標的能不能抬高天花板",
        "",
        f"- 產生於：{payload['generated_at']}",
        f"- 視窗：{payload['window']}",
        "- 方法：每層兩次單路徑重播（zero 取邊緣、spread_only 取價差），"
        "價差用實測的逐檔校準值而非 2.0 常數",
        "",
        "價差是部位大小唯一無法規避的成本項，而整張矩陣都是用每檔一律 2.0 bps "
        "的常數跑的。實測 L2 depth 顯示真實值介於 0.33 到 5.36 bps，離散 16 倍。",
        "如果每筆邊緣和價差無關，砍掉貴的標的就只砍分母不砍分子。",
        "如果邊緣其實來自寬價差（提供流動性的補償），三層的天花板會一樣平，槓桿就不存在。",
        "",
    ]

    tiers = payload["tiers"]
    L += ["## 分層", "",
          "| 層 | 檔數 | 平均 half-spread bps | 標的 |", "|---|---:|---:|---|"]
    for name in ("cheap", "all", "expensive"):
        syms = tiers[name]["symbols"]
        L.append(f"| {name} | {len(syms)} | "
                 f"{_fmt(tiers[name]['mean_half_spread_bps'])} | "
                 f"{', '.join(syms)} |")
    L.append("")

    for cell, res in payload["results"].items():
        L += [f"## `{cell}`", "",
              "| 層 | 筆數 | 每筆邊緣 bps | 來回價差 bps | 天花板 | 天花板/全體 |",
              "|---|---:|---:|---:|---:|---:|"]
        base = (res.get("all") or {}).get("ceiling")
        for name in ("cheap", "all", "expensive"):
            r = res.get(name) or {}
            ratio = (r.get("ceiling") / base) if (base and r.get("ceiling")) else None
            L.append(
                f"| {name} | {r.get('n_trades', '—')} | "
                f"{_fmt(r.get('edge_bps'))} | {_fmt(r.get('spread_bps'))} | "
                f"**{_fmt(r.get('ceiling'))}** | {_fmt(ratio)} |"
            )
        L += ["", res.get("verdict", ""), ""]

    L += ["## 保留", "",
          "- depth 校準只有 8 天。價差的**排序**通常很持久（由價格水準、跳動單位、"
          "流動性決定），但**水準**用 8 天定就薄了。",
          "- 縮小標的池會砍掉筆數，而 MC p5 閘門在 239 個交易日下需要年化 Sharpe 近 3.0。"
          "贏了成本卻輸掉樣本，是把成本問題換成檢定力問題，所以兩個數字並列而不合成一個判定。",
          "- 這裡完全沒有檢驗樣本外一致性。天花板高只代表值得花算力。",
          ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--start", default="2025-08-01")
    ap.add_argument("--end", default="2026-07-01")
    args = ap.parse_args()

    spread_payload = json.loads(SPREADS.read_text(encoding="utf-8"))
    spreads = {s: v["median_bps"] for s, v in spread_payload["symbols"].items()}
    tiers = _tiers(spreads)

    published = json.loads(GATE_REPORT.read_text(encoding="utf-8"))["cells"]

    from run_intraday_backtest import SIGNAL_WARMUP_DAYS, _load_real_bars
    from python.backtest.optimize import SIGNAL_PARAM_KEYS

    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)
    load_start = str((start_ts - pd.Timedelta(days=10)).date())
    print(f"loading 1m bars [{load_start}, {args.end}) for {len(spreads)} symbols...",
          flush=True)
    all_bars = _load_real_bars(sorted(spreads), load_start, args.end)
    if not all_bars:
        raise SystemExit("no cached 1m bars")

    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "window": f"{start_ts.date()} .. {end_ts.date()} (end-exclusive)",
        "tiers": {
            name: {
                "symbols": syms,
                "mean_half_spread_bps": sum(spreads[s] for s in syms) / len(syms),
            }
            for name, syms in tiers.items()
        },
        "results": {},
    }

    for cell in [c.strip() for c in args.cells.split(",") if c.strip()]:
        info = published.get(cell) or {}
        signal_name = info.get("signal")
        minutes = int(info.get("chart_minutes") or 5)
        # The published tuned params, so the comparison is against the cell as
        # it was actually measured rather than against yaml defaults.
        tuned = dict(info.get("candidate_params") or {})
        sig_keys = SIGNAL_PARAM_KEYS.get(signal_name, ())
        sig_params = {k: v for k, v in tuned.items() if k in sig_keys}
        warmup = SIGNAL_WARMUP_DAYS.get(signal_name, 1)
        print(f"\n=== {cell} ({signal_name}, {minutes}m, params={sig_params}) ===",
              flush=True)

        res: dict[str, dict] = {}
        for name in ("cheap", "all", "expensive"):
            syms = tiers[name]
            bars = {s: df for s, df in all_bars.items() if s in set(syms)}
            common = dict(chart_minutes=max(minutes, 1),
                          time_stop_minutes=time_stop_for(max(minutes, 1)))
            zero = _run(bars, signal_name, sig_params,
                        IntradayBacktestConfig(**common, half_spread_bps=0.0,
                                               impact_bps_per_participation=0.0),
                        warmup, start_ts, end_ts)
            spread_only = _run(
                bars, signal_name, sig_params,
                IntradayBacktestConfig(
                    **common, half_spread_bps=2.0,
                    half_spread_bps_by_symbol={s: spreads[s] for s in syms},
                    impact_bps_per_participation=0.0),
                warmup, start_ts, end_ts)
            edge = edge_bps_from_zero(zero)
            sp = spread_bps_from_runs(zero, spread_only)
            res[name] = {
                "n_trades": zero["n_trades"],
                "edge_bps": edge,
                "spread_bps": sp,
                "ceiling": (edge / sp) if (edge is not None and sp) else None,
            }
            print(f"    {name:10s} n={zero['n_trades']:6d} "
                  f"edge={_fmt(edge):>7} bps  spread={_fmt(sp):>6} bps  "
                  f"ceiling={_fmt(res[name]['ceiling']):>6}", flush=True)

        res["verdict"] = _verdict(res)
        payload["results"][cell] = res
        print(f"    -> {res['verdict']}", flush=True)
        OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        OUT_MD.write_text(_render(payload), encoding="utf-8")

    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")
    return 0


def _verdict(res: dict) -> str:
    """Is the lever real, and is it real for a reason that survives?"""
    cheap, base, exp = res.get("cheap"), res.get("all"), res.get("expensive")
    if not (cheap and base and cheap.get("ceiling") and base.get("ceiling")):
        return "無法判定：某一層沒有成交或沒有名目金額。"
    lift = cheap["ceiling"] / base["ceiling"]
    parts = [f"便宜層天花板是全體的 {lift:.2f} 倍"]

    # Edge is what must NOT move. If it tracks spread, the ceiling gain is an
    # accounting artifact of a smaller denominator on a smaller numerator.
    if cheap.get("edge_bps") and base.get("edge_bps"):
        edge_ratio = cheap["edge_bps"] / base["edge_bps"]
        parts.append(f"每筆邊緣是全體的 {edge_ratio:.2f} 倍")
        if exp and exp.get("edge_bps") and exp["edge_bps"] > 0:
            exp_ratio = exp["edge_bps"] / base["edge_bps"]
            parts.append(f"昂貴層邊緣是全體的 {exp_ratio:.2f} 倍")

    lost = 1 - (cheap["n_trades"] / base["n_trades"]) if base["n_trades"] else None
    if lost is not None:
        parts.append(f"筆數少了 {lost:.0%}")

    if lift >= 1.5 and (not cheap.get("edge_bps") or not base.get("edge_bps")
                        or cheap["edge_bps"] >= base["edge_bps"] * 0.8):
        parts.append(
            "**槓桿看起來是真的** — 分母降了而分子沒有跟著降。"
            "下一步是在便宜層上重跑完整 WFO，因為天花板完全沒有檢驗樣本外一致性。"
        )
    elif lift >= 1.5:
        parts.append(
            "**天花板升了，但每筆邊緣也一起降了** — 收益部分是分子變小的會計效果，"
            "要看淨額而不是比值。"
        )
    else:
        parts.append(
            "**槓桿不存在** — 換掉標的池並沒有實質改變邊緣與價差的比值。"
        )
    return "；".join(parts) + "。"


if __name__ == "__main__":
    raise SystemExit(main())
