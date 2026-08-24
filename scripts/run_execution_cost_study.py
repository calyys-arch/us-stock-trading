"""
Measures WHERE the per-trade execution cost comes from, and whether any
execution lever moves the edge-to-cost ratio enough to matter.

Background. scripts/run_slippage_decomposition.py established that most
retired intraday cells do have a pre-cost edge, and that they die because
per-trade cost ($150-270) dwarfs per-trade edge ($3-27). It did not say what
that cost is MADE of, so it could not say which lever, if any, is worth
pulling. This script answers that.

The engine charges two slippage terms per leg (intraday_engine._slippage_price):

    total_bps = half_spread_bps + impact_bps_per_participation * participation
    participation = min(shares / bar_volume, max_participation)

Those two behave completely differently under a change of position size, and
that asymmetry is the whole reason this study exists:

  * HALF-SPREAD is linear in size. Dollar cost = notional * spread_bps, and
    pre-cost edge is also linear in notional, so halving size halves both.
    The ratio is unchanged. Size is not a lever against spread.
  * IMPACT is quadratic in size. impact_bps itself scales with `shares`, so
    dollar cost scales with notional * shares, i.e. size squared, while edge
    stays linear. Halving size cuts impact cost to a QUARTER while cutting
    edge only in half — the ratio doubles.

So "can execution work save any of these signals?" reduces to "how much of
the slippage is impact rather than spread?", and that is measurable by
re-running each cell with one term switched off at a time.

Method — per cell, frozen params, single-path replays, no WFO:

  1. Cost attribution, four runs:
       baseline     spread 2.0 bps, impact 20 bps/participation  (the published cost model)
       spread_only  spread 2.0 bps, impact 0
       impact_only  spread 0,       impact 20
       zero         spread 0,       impact 0
     Slippage under each is (net_zero - net_run - commission_run), the same
     exact identity run_slippage_decomposition.py uses. Attribution is then
     read off directly rather than modelled.

  2. Size sweep at baseline costs, scaling BOTH `risk_per_trade_pct` and
     `max_notional_pct` by f. Both must move together: sizing is
     min(risk-implied shares, notional-cap shares), and for these signals the
     notional cap is frequently the binding one, so scaling risk alone would
     leave many trades the same size and understate the effect. Each f needs
     its own zero-cost twin, because per-trade EDGE scales with size too and
     the ratio is only meaningful against a matched baseline.

CAVEAT, same as the decomposition script: changing cost changes fill prices,
so the trade set can shift between runs. n_trades is reported for every run
so divergence is visible rather than assumed away. The attribution is a
comparison of achievable paths, not an accounting identity over a fixed path.

What this script deliberately does NOT model: passive/limit entry. Setting
half_spread to 0 is the perfect-passive BOUND (you cross nothing and pay no
spread), not a realistic limit-order simulation — real passive fills buy that
saving with queue position and adverse selection, neither of which exists in
a bar-replay engine. The spread_only/impact_only split still bounds how much
a passive router could possibly recover, which is the decision-relevant
number here.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from python.analytics.execution_ceiling import ceiling_from_runs  # noqa: E402
from python.analytics.volume_route_policy import time_stop_for  # noqa: E402
from python.backtest.intraday_engine import (  # noqa: E402
    IntradayBacktestConfig,
    IntradayBacktestReport,
    metrics_from_report,
    run_intraday_backtest,
)

GATE_REPORT = Path("backtests/reports/entry_hypothesis_gate_report.json")
STRATEGY_PATH = Path("configs/strategy.yaml")
OUT_JSON = Path("backtests/reports/execution_cost_study.json")
OUT_MD = Path("backtests/reports/execution_cost_study.md")

# Cheapest first so partial progress is useful. sweep_reclaim_1m (104k fills)
# and the orb pair (35-day warmup) are excluded by default: this study needs
# ~8 replays per cell and those would dominate the runtime without changing
# the shape of the answer, which the decomposition already showed is uniform
# across the 1m cohort.
DEFAULT_CELLS = ("auction_reclaim_5m", "vsa_no_demand_5m", "vp_breakout_1m", "fvg_retest_1m")

COST_VARIANTS = {
    "baseline": {"half_spread_bps": 2.0, "impact_bps_per_participation": 20.0},
    "spread_only": {"half_spread_bps": 2.0, "impact_bps_per_participation": 0.0},
    "impact_only": {"half_spread_bps": 0.0, "impact_bps_per_participation": 20.0},
    "zero": {"half_spread_bps": 0.0, "impact_bps_per_participation": 0.0},
}
ZERO_COMMISSION = {"commission_per_share": 0.0, "min_commission": 0.0}
SIZE_FACTORS = (1.0, 0.5, 0.25, 0.1)

_MAX_WARMUP_DAYS = 40


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _run(bars, signal_name, base_cfg, params, sig_keys, cfg, warmup_days, start_ts, end_ts) -> dict:
    """One single-path replay. Mirrors optimize.build_intraday_backtest_fn's
    warmup slice and exit-time filter so numbers stay comparable to the WFO
    full-window figures, but keeps the trade objects so notional and share
    counts can be summarised."""
    merged = {**base_cfg, **params}
    sig_params = {k: merged[k] for k in sig_keys if k in merged}
    warmup_start = start_ts - pd.Timedelta(days=warmup_days)
    sliced = {
        symbol: window
        for symbol, bars_df in bars.items()
        if not (window := bars_df.loc[(bars_df.index >= warmup_start) & (bars_df.index < end_ts)]).empty
    }
    report = run_intraday_backtest(sliced, signal_name, sig_params, cfg)
    in_window = [t for t in report.trades if start_ts <= t.exit_time < end_ts]
    metrics = metrics_from_report(
        IntradayBacktestReport(trades=in_window), cfg.capital,
    )
    notionals = [t.shares * t.entry_price for t in in_window]
    return {
        "n_trades": int(metrics["n_trades"]),
        "profit_factor": metrics["profit_factor"],
        "total_net_pnl": float(metrics["total_net_pnl"]),
        "commission": float(metrics["total_costs"]),
        "avg_shares": (sum(t.shares for t in in_window) / len(in_window)) if in_window else None,
        "avg_notional": (sum(notionals) / len(notionals)) if notionals else None,
    }


def _fmt(value, spec: str = ".3f", dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _money(value, spec: str = ",.0f", dash: str = "—") -> str:
    if value is None:
        return dash
    amount = float(value)
    return f"-${abs(amount):{spec}}" if amount < 0 else f"${amount:{spec}}"


def _persist(payload: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_render_md(payload), encoding="utf-8")


def _render_md(payload: dict) -> str:
    L: list[str] = [
        "# 執行成本研究：成本由什麼構成，縮小部位能不能救",
        "",
        f"- 產生時間：{payload['generated_at']}",
        f"- 視窗：{payload['window']}",
        "- 方法：凍結參數、單路徑重播，無 WFO、無重新最佳化",
        "",
        "引擎每一腿收兩種滑價：`half_spread_bps` 與 "
        "`impact_bps_per_participation × min(shares/bar_volume, max_participation)`。",
        "兩者對部位大小的行為完全不同，這是本研究的全部重點：",
        "",
        "- **半價差對大小是線性的。** 成本 = 名目 × bps，稅前邊緣也是名目的線性函數，"
        "所以縮小部位會等比縮小兩者，比值不變。對半價差而言，部位大小不是槓桿。",
        "- **衝擊對大小是二次的。** `impact_bps` 本身隨 `shares` 增加，"
        "所以成本隨「名目 × 股數」變化，而邊緣只隨名目變化。部位減半，衝擊成本只剩四分之一，"
        "邊緣剩一半，比值變成兩倍。",
        "",
        "因此「執行工程救不救得了這些訊號」等價於「滑價裡有多少是衝擊而非價差」。",
        "",
    ]
    for row in payload.get("results", []):
        L.append(f"## `{row['cell']}`")
        L.append("")
        if row.get("error"):
            L.append(f"- 執行失敗：`{row['error']}`")
            L.append("")
            continue
        L.append(f"- 訊號 `{row['signal']}`，{row['chart_minutes']}m 圖，warmup {row['warmup_days']} 天")
        base = row["attribution"]["baseline"]
        L.append(
            f"- 基準每筆平均：{_fmt(base.get('avg_shares'), ',.0f')} 股、"
            f"名目 {_money(base.get('avg_notional'))}"
        )
        L.append("")
        L.append("### 成本歸因")
        L.append("")
        L.append("| 成本設定 | 筆數 | PF | 淨額 | 滑價 | 佣金 |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for name in COST_VARIANTS:
            v = row["attribution"].get(name) or {}
            L.append(
                f"| {name} | {v.get('n_trades', '—')} | {_fmt(v.get('profit_factor'))} | "
                f"{_money(v.get('total_net_pnl'))} | {_money(v.get('slippage'))} | "
                f"{_money(v.get('commission'))} |"
            )
        split = row.get("split") or {}
        L.append("")
        L.append(
            f"- 滑價拆分：價差 {_money(split.get('spread'))}"
            f"（{_fmt(split.get('spread_share'), '.1%')}）、"
            f"衝擊 {_money(split.get('impact'))}（{_fmt(split.get('impact_share'), '.1%')}）"
        )
        L.append(f"- **判讀**：{row.get('lever_verdict', '')}")
        if row.get("ceiling_verdict"):
            limit = row.get("limit") or {}
            L.append("")
            L.append("### 漸近極限（部位 → 0）")
            L.append("")
            L.append(
                f"| 每筆邊緣 | 實測價差成本 | 比值上限 |\n|---:|---:|---:|\n"
                f"| {_fmt(limit.get('edge_bps'), '.2f')} bps | "
                f"{_fmt(limit.get('spread_round_trip_bps'), '.2f')} bps | "
                f"{_fmt(limit.get('ceiling'), '.2f')} |"
            )
            L.append("")
            L.append(f"- **判讀**：{row['ceiling_verdict']}")
            L.append(
                f"- 限制：價差與衝擊分開重播不可加，聯合成本比兩項之和多 "
                f"{_money(split.get('additivity_gap'))}"
                f"（基準滑價的 {_fmt((split.get('additivity_gap') or 0) / (split.get('baseline_slippage') or 1), '.1%')}）。"
                "關掉一項會改變成交價、進而改變哪些棒觸發停損，所以這是路徑差分而非會計恆等式。"
            )
        L.append("")
        L.append("### 部位大小掃描（基準成本）")
        L.append("")
        L.append("| 倍率 | 筆數 | PF | 淨額 | 每筆邊緣 | 每筆成本 | 倍數 |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|")
        for entry in row.get("size_sweep", []):
            L.append(
                f"| {entry['factor']:.2f}x | {entry.get('n_trades', '—')} | "
                f"{_fmt(entry.get('profit_factor'))} | {_money(entry.get('total_net_pnl'))} | "
                f"{_money(entry.get('edge_per_trade'), ',.2f')} | "
                f"{_money(entry.get('cost_per_trade'), ',.2f')} | "
                f"{_fmt(entry.get('edge_cost_ratio'), '.2f')} |"
            )
        L.append("")
        L.append(f"- **判讀**：{row.get('size_verdict', '')}")
        L.append("")
    return "\n".join(L) + "\n"


def _lever_verdict(impact_share: float | None) -> str:
    if impact_share is None:
        return "無法判讀（重播沒有成交）"
    if impact_share >= 0.6:
        return (
            "**滑價以參與度衝擊為主** — 部位大小是有效槓桿，因為衝擊成本對大小是二次的。"
            "縮小部位或拆單應該會改善每筆邊緣對成本的比值，見下方掃描。"
        )
    if impact_share >= 0.25:
        return (
            "價差與衝擊各佔一部分。縮小部位只能吃掉衝擊那一塊，"
            "價差那一塊要靠被動掛單或換標的才動得了。"
        )
    return (
        "**滑價幾乎全是半價差** — 部位大小不是槓桿（價差成本與邊緣同為線性，比值不變）。"
        "唯一能動的是被動成交或換更窄價差的標的。"
    )


def _fix_baseline_row(row: dict) -> None:
    """Restate the 1.0x sweep row against a commission-free edge baseline.

    Runs produced before this correction took the 1.0x edge from the
    attribution "zero" variant, which still pays commission, while every
    other size factor used a commission-free twin. That biased the 1.0x edge
    (and therefore the ratio) low, overstating how much shrinking size helps.
    Commission never enters `_slippage_price`, so adding it back is exact.
    """
    sweep = row.get("size_sweep") or []
    base = next((e for e in sweep if e.get("factor") == 1.0), None)
    zero = (row.get("attribution") or {}).get("zero") or {}
    normal = (row.get("attribution") or {}).get("baseline") or {}
    if base is None or not zero or not normal:
        return
    n_zero = zero.get("n_trades") or 0
    if not n_zero:
        return
    net_zero = float(zero["total_net_pnl"]) + float(zero.get("commission") or 0.0)
    edge = net_zero / n_zero
    cost = (net_zero - float(normal["total_net_pnl"])) / n_zero
    base["edge_per_trade"] = edge
    base["cost_per_trade"] = cost
    base["edge_cost_ratio"] = (edge / cost) if cost else None


def _ceiling(row: dict) -> dict | None:
    """The edge-to-cost ratio this cell converges to as position size -> 0.

    Thin wrapper over python.analytics.execution_ceiling so this script and
    scripts/run_ceiling_prescreen.py cannot drift apart on the formula. See
    that module for why the limit exists and what it excludes.
    """
    attribution = row.get("attribution") or {}
    if not attribution:
        return None
    return ceiling_from_runs(
        baseline=attribution.get("baseline") or {},
        spread_only=attribution.get("spread_only") or {},
        zero=attribution.get("zero") or {},
    )


def _ceiling_verdict(limit: dict | None) -> str:
    if not limit or limit.get("ceiling") is None:
        return ""
    ceiling, edge, spread = limit["ceiling"], limit["edge_bps"], limit["spread_round_trip_bps"]
    head = (
        f"每筆稅前邊緣 {edge:.2f} bps，實測價差成本 {spread:.2f} bps"
        f"（名目應為 4.0 bps，差額見下方限制），部位趨零時比值收斂到 {ceiling:.2f}。"
    )
    if ceiling < 1.0:
        return (
            f"{head} **邊緣比價差還薄，任何部位大小都過不了 PF 1.0** —— "
            "這是訊號的性質，不是執行的問題。掃描看到的改善是衝擊在消失，"
            "但它停在這個上限之下。"
        )
    if ceiling < 1.5:
        return (
            f"{head} 上限勉強過 1，但佣金的每股下限會在小部位吃掉這點餘裕，"
            "實際掃描到的比值會低於上限。"
        )
    return (
        f"{head} 上限有實質餘裕，代表衝擊確實是可以工程掉的那一塊，"
        "縮小部位或拆單有意義。"
    )


def _size_verdict(sweep: list[dict]) -> str:
    usable = [e for e in sweep if e.get("edge_cost_ratio") is not None]
    if not usable:
        return "無法判讀（掃描沒有成交）"
    best = max(usable, key=lambda e: e["edge_cost_ratio"])
    base = next((e for e in usable if e["factor"] == 1.0), None)
    if base is None:
        return "無 1.0x 基準可比"
    gain = best["edge_cost_ratio"] - base["edge_cost_ratio"]
    shrink = (
        f"代價是絕對淨額由 {_money(base['total_net_pnl'])} 掉到 "
        f"{_money(best['total_net_pnl'])}，因為部位小了；比值改善不等於賺得更多。"
    )
    if (base.get("profit_factor") or 0) >= 1.0:
        # Nothing to "flip" — this cell already clears the cost-adjusted PF
        # gate at full size, so size is a margin question, not a rescue.
        return (
            f"這格在 1.0x 就已經是成本後 PF {base['profit_factor']:.3f}，不需要靠縮小部位翻正。"
            f"縮小到 {best['factor']:.2f}x 把比值由 {base['edge_cost_ratio']:.2f} 推到 "
            f"{best['edge_cost_ratio']:.2f}、PF 到 {best['profit_factor']:.3f}，"
            f"買到的是壓力測試的餘裕而不是正負號。{shrink}"
        )
    flipped = [e for e in usable if (e.get("profit_factor") or 0) >= 1.0]
    if flipped:
        largest = max(flipped, key=lambda e: e["factor"])
        return (
            f"**縮小部位讓這格在 {largest['factor']:.2f}x 由成本後 PF "
            f"{base['profit_factor']:.3f} 翻成 {largest['profit_factor']:.3f}**。{shrink}"
            " 值得做正式的 WFO 重驗，而不是只看單路徑。"
        )
    if gain >= 0.2:
        return (
            f"比值由 {base['edge_cost_ratio']:.2f} 改善到 {best['edge_cost_ratio']:.2f}"
            f"（{best['factor']:.2f}x），方向對但仍不足以讓成本後 PF 過 1。"
            "縮小部位單獨不夠，需要同時提高每筆邊緣。"
        )
    return (
        f"比值幾乎不動（{base['edge_cost_ratio']:.2f} → {best['edge_cost_ratio']:.2f}）。"
        "這格的成本結構對部位大小不敏感，縮小部位不是解方。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default=",".join(DEFAULT_CELLS))
    parser.add_argument("--start", default="2025-08-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument(
        "--rerender", action="store_true",
        help="recompute verdicts from the existing JSON and rewrite the "
             "markdown, without re-running any replay",
    )
    args = parser.parse_args()

    if args.rerender:
        payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        for row in payload.get("results", []):
            if row.get("error"):
                continue
            _fix_baseline_row(row)
            split = row.get("split") or {}
            row["lever_verdict"] = _lever_verdict(split.get("impact_share"))
            row["limit"] = _ceiling(row)
            row["ceiling_verdict"] = _ceiling_verdict(row["limit"])
            row["size_verdict"] = _size_verdict(row.get("size_sweep") or [])
            print(f"  {row['cell']}: {row['size_verdict']}", flush=True)
        _persist(payload)
        print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")
        return 0

    report = json.loads(GATE_REPORT.read_text(encoding="utf-8"))
    cells = report.get("cells") or {}
    strategy = _load_yaml(STRATEGY_PATH)

    from run_intraday_backtest import SIGNAL_WARMUP_DAYS, _load_real_bars
    from python.backtest.optimize import SIGNAL_PARAM_KEYS
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
            "frozen params, single-path replays; slippage = net_zero_cost - net_run "
            "- commission_run; size sweep scales risk_per_trade_pct and "
            "max_notional_pct together"
        ),
        "cost_variants": COST_VARIANTS,
        "size_factors": list(SIZE_FACTORS),
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
        sig_keys = SIGNAL_PARAM_KEYS[signal_name]
        base_cfg = strategy[signal_name]
        row: dict = {
            "cell": key, "signal": signal_name, "chart_minutes": minutes,
            "warmup_days": warmup, "frozen_params": params,
        }
        print(f"\n=== {key} ({signal_name}, {minutes}m, warmup {warmup}d) ===", flush=True)

        def make_cfg(size_factor: float = 1.0, **cost_kw) -> IntradayBacktestConfig:
            cfg = IntradayBacktestConfig(
                chart_minutes=max(minutes, 1),
                time_stop_minutes=time_stop_for(max(minutes, 1)),
                **cost_kw,
            )
            if size_factor == 1.0:
                return cfg
            return replace(
                cfg,
                risk_per_trade_pct=cfg.risk_per_trade_pct * size_factor,
                max_notional_pct=cfg.max_notional_pct * size_factor,
            )

        try:
            attribution: dict[str, dict] = {}
            for name, cost_kw in COST_VARIANTS.items():
                cfg = make_cfg(**cost_kw)
                res = _run(bars, signal_name, base_cfg, params, sig_keys, cfg,
                           warmup, start_ts, end_ts)
                attribution[name] = res
                print(f"    {name:12s} n={res['n_trades']:6d} PF={_fmt(res['profit_factor'])} "
                      f"net={_money(res['total_net_pnl']):>14}", flush=True)

            net_zero = attribution["zero"]["total_net_pnl"]
            for name, res in attribution.items():
                res["slippage"] = net_zero - res["total_net_pnl"] - res["commission"]
            spread = attribution["spread_only"]["slippage"]
            impact = attribution["impact_only"]["slippage"]
            total = spread + impact
            row["attribution"] = attribution
            row["split"] = {
                "spread": spread,
                "impact": impact,
                "spread_share": (spread / total) if total else None,
                "impact_share": (impact / total) if total else None,
                "baseline_slippage": attribution["baseline"]["slippage"],
                "additivity_gap": attribution["baseline"]["slippage"] - total,
            }
            impact_share = row["split"]["impact_share"]
            row["lever_verdict"] = _lever_verdict(impact_share)
            print(f"    -> 價差 {_money(spread)} ({_fmt(row['split']['spread_share'], '.1%')}) | "
                  f"衝擊 {_money(impact)} ({_fmt(impact_share, '.1%')})", flush=True)

            sweep: list[dict] = []
            for factor in SIZE_FACTORS:
                normal = (
                    attribution["baseline"] if factor == 1.0
                    else _run(bars, signal_name, base_cfg, params, sig_keys,
                              make_cfg(factor, **COST_VARIANTS["baseline"]),
                              warmup, start_ts, end_ts)
                )
                # The 1.0x baseline reuses the attribution "zero" run, which
                # still pays commission — every other factor gets a
                # commission-free twin. Adding the commission back makes them
                # comparable, and is exact rather than approximate: commission
                # never enters _slippage_price, so removing it cannot change
                # the trade set or any fill price, only the net.
                zero = (
                    {**attribution["zero"],
                     "total_net_pnl": attribution["zero"]["total_net_pnl"]
                     + attribution["zero"]["commission"],
                     "commission": 0.0}
                    if factor == 1.0
                    else _run(bars, signal_name, base_cfg, params, sig_keys,
                              make_cfg(factor, **COST_VARIANTS["zero"], **ZERO_COMMISSION),
                              warmup, start_ts, end_ts)
                )
                n_zero = zero["n_trades"]
                edge = (zero["total_net_pnl"] / n_zero) if n_zero else None
                cost_total = zero["total_net_pnl"] - normal["total_net_pnl"]
                cost = (cost_total / n_zero) if n_zero else None
                sweep.append({
                    "factor": factor,
                    "n_trades": normal["n_trades"],
                    "profit_factor": normal["profit_factor"],
                    "total_net_pnl": normal["total_net_pnl"],
                    "avg_notional": normal.get("avg_notional"),
                    "edge_per_trade": edge,
                    "cost_per_trade": cost,
                    "edge_cost_ratio": (edge / cost) if (edge is not None and cost) else None,
                })
                print(f"    size {factor:.2f}x n={normal['n_trades']:6d} "
                      f"PF={_fmt(normal['profit_factor'])} "
                      f"net={_money(normal['total_net_pnl']):>14} "
                      f"倍數={_fmt(sweep[-1]['edge_cost_ratio'], '.2f')}", flush=True)
            row["size_sweep"] = sweep
            row["limit"] = _ceiling(row)
            row["ceiling_verdict"] = _ceiling_verdict(row["limit"])
            row["size_verdict"] = _size_verdict(sweep)
            if row["limit"] and row["limit"].get("ceiling") is not None:
                print(f"    -> 上限 {row['limit']['ceiling']:.2f}"
                      f"（邊緣 {row['limit']['edge_bps']:.2f} bps / "
                      f"價差 {row['limit']['spread_round_trip_bps']:.2f} bps）", flush=True)
            print(f"    => {row['size_verdict']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    !! {row['error']}", flush=True)

        payload["results"].append(row)
        _persist(payload)

    _persist(payload)
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
