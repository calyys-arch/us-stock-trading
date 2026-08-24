"""
Full walk-forward on the two spread tiers the ceiling study picked out.

What the ceiling study found, and why it needs a WFO. Restricting the universe
by quoted spread moved the edge/cost ceiling hard, in OPPOSITE directions for
the two signals measured (backtests/reports/spread_tier_study.md):

  vsa_no_demand_5m   cheap 8 -> ceiling 10.60 (edge 6.45 bps / spread 0.61)
                     all 20  -> ceiling  2.05 (edge 7.96 / 3.89)
                     dear 8  -> ceiling  1.48 (edge 9.60 / 6.49)

  auction_reclaim_5m cheap 8 -> ceiling  1.82 (edge  1.28 / 0.70)
                     all 20  -> ceiling  2.33 (edge 12.00 / 5.14)
                     dear 8  -> ceiling  3.62 (edge 39.59 / 10.94)

vsa_no_demand's per-trade edge barely moves across a 10.6x range of spread, so
tightening the universe cuts the denominator and leaves the numerator.
auction_reclaim's edge grows 30.9x against a 15.6x spread range — its edge IS
the quote width, consistent with a reclaim-after-sweep being paid for
providing liquidity at an extreme. So each signal gets the tier that suits it.

The ceiling is a necessary condition and nothing more: it says nothing about
out-of-sample consistency, which is where every cell that reached a WFO
actually died (wfo_go, Monte Carlo p5). This script supplies the missing half.

Cost model. Both runs use the CALIBRATED per-symbol half-spreads measured from
captured L2 depth, not the flat 2.0 bps the published matrix assumed. For
auction_reclaim's expensive tier that is a substantial handicap versus the
published baseline (10.94 bps round trip against 2.92), so an improvement there
is being bought under a stricter cost model than the number it is compared to.
For vsa_no_demand's cheap tier the calibrated spread is far lower — but that is
the treatment itself, not a thumb on the scale: those symbols really are cheap
to trade, and the flat constant was hiding it.

What a pass here would and would not mean. The published all-20 baselines were
run at flat 2.0 bps, which the calibration shows was too optimistic by 1.76x
for auction_reclaim and 1.28x for vsa_no_demand. The baselines are therefore
FLATTERED, and beating them is a stronger result than it looks. Trade count
also falls with the universe (84 and 44 in the ceiling study, against 248 and
111), so a tier that wins on cost may lose on statistical power — the Monte
Carlo p5 gate needs an annualized Sharpe near 3.0 over this 239-day window.
Both are reported rather than collapsed into one number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from python.analytics.volume_route_policy import time_stop_for  # noqa: E402
from python.backtest.intraday_engine import IntradayBacktestConfig  # noqa: E402

GATE_REPORT = Path("backtests/reports/entry_hypothesis_gate_report.json")
SPREADS = Path("backtests/reports/calibrated_spreads.json")
TIER_STUDY = Path("backtests/reports/spread_tier_study.json")
OUT_JSON = Path("backtests/reports/spread_tier_wfo.json")
OUT_MD = Path("backtests/reports/spread_tier_wfo.md")

# (cell, tier) — each signal gets the tier its ceiling favored.
RUNS = (
    ("vsa_no_demand_5m", "cheap"),
    ("auction_reclaim_5m", "expensive"),
)


def _fmt(value, spec: str = ".3f", dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _verdict(cell: str, tier: str, base: dict, new: dict) -> str:
    """Did the OUT-OF-SAMPLE evidence move, or only the in-sample cost math?

    Written to resist the failure this project already walked into once: at
    0.75x size auction_reclaim_5m cleared the then-hard gates purely on an
    in-sample stress replay while its walk-forward and Monte Carlo numbers sat
    still (backtests/reports/sizing_wfo_auction_reclaim_5m.md). A cheaper
    universe is a different lever, but the test for whether it is real is the
    same one: per-fold out-of-sample metrics, not the pooled cost arithmetic.
    """
    lines = []
    hard = new.get("gates") or {}
    failed = [k for k, v in hard.items() if not v]
    lines.append(
        f"`{cell}` @ {tier}：{new.get('decision')}"
        + (f"（未過：{', '.join(failed)}）" if failed else "（全數通過）")
    )

    pairs = [
        ("WFO 通過率", base.get("wfo_pass_ratio"), new.get("wfo_pass_ratio"), ".3f"),
        ("OOS Sharpe 平均", base.get("oos_sharpe_mean"), new.get("oos_sharpe_mean"), "+.3f"),
        ("MC p5 Sharpe", base.get("mc_p5_sharpe"), new.get("mc_p5_sharpe"), "+.3f"),
        ("成本後 PF", (base.get("full_window_metrics") or {}).get("profit_factor"),
         (new.get("full_window_metrics") or {}).get("profit_factor"), ".3f"),
        ("壓力 PF", (base.get("stress_metrics") or {}).get("profit_factor"),
         (new.get("stress_metrics") or {}).get("profit_factor"), ".3f"),
        ("筆數", (base.get("full_window_metrics") or {}).get("n_trades"),
         (new.get("full_window_metrics") or {}).get("n_trades"), ".0f"),
    ]
    for label, b, n, spec in pairs:
        lines.append(f"  - {label}：{_fmt(b, spec)} -> {_fmt(n, spec)}")

    # The two gates that test consistency rather than survival.
    oos_moved = [
        n > b for b, n in (
            (base.get("wfo_pass_ratio") or 0.0, new.get("wfo_pass_ratio") or 0.0),
            (base.get("oos_sharpe_mean") or 0.0, new.get("oos_sharpe_mean") or 0.0),
            (base.get("mc_p5_sharpe") or 0.0, new.get("mc_p5_sharpe") or 0.0),
        )
    ]
    if new.get("decision") == "GO":
        lines.append(
            "  - **通過**。天花板預測的方向成立，且撐過了走步與 Monte Carlo。"
            "下一步是七月 holdout，不是直接相信。"
        )
    elif all(oos_moved):
        lines.append(
            "  - 三個樣本外指標全數改善但仍未過閘門。**方向對，幅度不夠** — "
            "值得往同方向再推（用價差門檻而非固定 8 檔以補回樣本）。"
        )
    elif any(oos_moved):
        lines.append(
            "  - 樣本外指標部分改善。要看是不是只有成本算術在動 — "
            "如果 WFO 通過率與 MC p5 沒動，那就和先前縮小部位的假陽性同一類。"
        )
    else:
        lines.append(
            "  - **樣本外指標沒有改善。**天花板升高沒有轉化成逐折的一致性，"
            "所以這個槓桿對這格是無效的。"
        )
    return "\n".join(lines)


def _render(payload: dict) -> str:
    L = [
        "# 價差分層的走步驗證",
        "",
        f"- 產生於：{payload['generated_at']}",
        f"- 視窗：{payload['window']}",
        "- 成本：實測逐檔 half-spread（非 2.0 常數）",
        "",
        "天花板研究顯示，限制標的池會大幅改變邊緣與成本的比值，而且兩個訊號的**方向相反**：",
        "`vsa_no_demand` 的每筆邊緣在 10.6 倍的價差範圍內幾乎不動，所以緊價差標的只砍分母；",
        "`auction_reclaim` 的邊緣隨價差超比例成長，它的邊緣就是報價寬度。",
        "所以每個訊號各配它天花板偏好的那一層。",
        "",
        "但天花板只是必要條件，它完全沒有檢驗樣本外一致性——而那正是所有跑到 WFO 的格子",
        "真正陣亡的地方。這份報告補上那一半。",
        "",
    ]
    for row in payload["results"]:
        L += [f"## `{row['cell']}` @ {row['tier']}", "",
              f"- 標的（{len(row['symbols'])} 檔）：{', '.join(row['symbols'])}",
              f"- 平均 half-spread：{_fmt(row['mean_half_spread_bps'], '.2f')} bps",
              f"- 天花板（來自分層研究）：{_fmt(row.get('ceiling'), '.2f')}",
              ""]
        if row.get("error"):
            L += [f"執行失敗：`{row['error']}`", ""]
            continue
        L += [row["verdict"], ""]

    L += ["## 讀這份報告時要記得的事", "",
          "- 已發布的全 20 檔基準是用 2.0 bps 常數跑的，而校準顯示那對 `auction_reclaim` "
          "低估成本 1.76 倍、對 `vsa_no_demand` 低估 1.28 倍。基準被美化了，所以贏過它比看起來更強。",
          "- 縮小標的池會砍掉筆數（分層研究裡是 84 與 44，對比 248 與 111）。"
          "MC p5 閘門在 239 個交易日下需要年化 Sharpe 近 3.0，所以贏了成本可能輸掉檢定力。",
          "- depth 校準只有 8 天。價差的排序通常持久，水準則不然。",
          ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2025-08-01")
    ap.add_argument("--end", default="2026-07-01")
    args = ap.parse_args()

    spread_payload = json.loads(SPREADS.read_text(encoding="utf-8"))
    spreads = {s: v["median_bps"] for s, v in spread_payload["symbols"].items()}
    tier_study = json.loads(TIER_STUDY.read_text(encoding="utf-8"))
    tiers = {name: t["symbols"] for name, t in tier_study["tiers"].items()}
    published = json.loads(GATE_REPORT.read_text(encoding="utf-8"))["cells"]

    from run_intraday_backtest import _load_real_bars, run_signal

    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)
    load_start = str((start_ts - pd.Timedelta(days=10)).date())
    print(f"loading 1m bars [{load_start}, {args.end})...", flush=True)
    all_bars = _load_real_bars(sorted(spreads), load_start, args.end)
    if not all_bars:
        raise SystemExit("no cached 1m bars")

    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "window": f"{start_ts.date()} .. {end_ts.date()} (end-exclusive)",
        "results": [],
    }

    for cell, tier in RUNS:
        info = published[cell]
        signal_name = info["signal"]
        minutes = int(info.get("chart_minutes") or 5)
        syms = tiers[tier]
        bars = {s: df for s, df in all_bars.items() if s in set(syms)}
        tier_spreads = {s: spreads[s] for s in syms}
        ceiling = ((tier_study["results"].get(cell) or {}).get(tier) or {}).get("ceiling")

        row = {
            "cell": cell,
            "tier": tier,
            "symbols": syms,
            "mean_half_spread_bps": sum(tier_spreads.values()) / len(tier_spreads),
            "ceiling": ceiling,
        }
        print(f"\n=== {cell} @ {tier} ({len(syms)} symbols, "
              f"mean half-spread {row['mean_half_spread_bps']:.2f} bps) ===", flush=True)

        try:
            result = run_signal(
                signal_name,
                SimpleNamespace(demo=False, start=args.start, end=args.end),
                bars,
                f"spread tier '{tier}' ({len(syms)} symbols), calibrated half-spreads",
                start_ts, end_ts,
                half_spread_bps_by_symbol=tier_spreads,
                engine_cfg=IntradayBacktestConfig(
                    chart_minutes=max(minutes, 1),
                    time_stop_minutes=time_stop_for(max(minutes, 1)),
                    half_spread_bps_by_symbol=tier_spreads,
                ),
            )
            row["result"] = {k: v for k, v in result.items() if k != "daily_returns"}
            row["baseline"] = {
                "wfo_pass_ratio": info.get("wfo_pass_ratio"),
                "oos_sharpe_mean": info.get("oos_sharpe_mean"),
                "mc_p5_sharpe": info.get("mc_p5_sharpe"),
                "full_window_metrics": info.get("full_window_metrics"),
                "stress_metrics": info.get("stress_metrics"),
                "decision": info.get("decision"),
            }
            row["verdict"] = _verdict(cell, tier, row["baseline"], result)
            print(row["verdict"], flush=True)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    !! {row['error']}", flush=True)

        payload["results"].append(row)
        OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        OUT_MD.write_text(_render(payload), encoding="utf-8")

    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
