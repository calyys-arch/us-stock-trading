"""Decide which cells deserve a walk-forward, using three replays instead of hours.

The problem this solves. A full WFO for one cell costs 1-4.5 hours, and the
15x7 matrix still has nine cells without one. Spending that compute uniformly
is wasteful now that we know why every finished cell died: per-trade edge is
thinner than per-trade execution cost. python.analytics.execution_ceiling turns
that into a number computable from three single-path replays —

    ceiling = edge_bps / spread_round_trip_bps

— and a cell whose ceiling is below 1.0 cannot clear PF 1.0 at ANY position
size, however good the execution. Those never need a WFO at all.

Two things are screened, cheapest first.

1. FEASIBILITY, pure arithmetic, no replay. Signals only ever see bars from
   the current session (`intraday_engine.run_symbol_day` slices `day_bars` per
   session date), so a 390-minute session holding fewer decision bars than the
   signal's minimum makes a cell structurally incapable of firing. That is how
   `vsa_no_demand_60m` was ruled out — 6 bars available against 8 required,
   confirmed by an 8-minute replay that produced exactly zero trades before
   the check existed to predict it.

2. THE CEILING as a REJECTION test on a GIVEN parameter set — not as a
   predictor of what tuning could reach. That distinction was learned the
   expensive way; see "What was tried and rejected" below. Applied to
   WFO-tuned params it works and has already retired cells honestly
   (vp_breakout_1m at 0.21, fvg_retest_1m at 0.40 — no position size saves
   either). Applied to untuned params it measures the yaml, not the signal.

WHAT WAS TRIED AND REJECTED, so nobody pays for it twice.

Attempt A — screen at `configs/strategy.yaml` defaults, assuming tuning only
raises per-trade edge so defaults would be a lower bound. Calibration against
the published tuned ceilings killed it: 1.95x for auction_reclaim_5m (8.03 vs
4.11), 0.11x for vsa_no_demand_5m (0.29 vs 2.61), 0.52x for vp_breakout_1m.
Ordering survived, scale did not, across an 18x range.

Attempt B — sweep each signal's entry filter toward TIGHTER values, expecting
selectivity to concentrate a conserved pool of edge. auction_reclaim's
`min_rel_volume` sweep refuted it outright: trade count 56 -> 21 -> 8 -> 2 and
total edge 1363 -> 867 -> -4 -> 92 bps-trades. Tightening past the default
destroys edge rather than redistributing it, and leaves no testable sample.

Attempt C — sweep `stop_atr_mult`, on the hypothesis that holding time was the
axis, since the one conserving pair observed (auction_reclaim default vs tuned,
1363 vs 1332 bps-trades) differed partly in the stop. Also refuted, on both
calibration cells: trade count is INVARIANT to the stop (auction 56 across
0.15-0.90, i.e. 0% drift; vsa_no_demand 1533 -> 1503, 2% drift) while the
envelope reached 0.22 against a tuned 2.61 for vsa_no_demand. The stop only
redistributes the P&L of a fixed trade set; it does not change how many
opportunities exist.

Why no single-knob sweep can work. Trade count is driven by the ENTRY filter,
and the productive direction along it is signal-specific: the WFO went LOOSER
for auction_reclaim (`min_rel_volume` 1.2 -> 1.0, 56 -> 111 trades, edge 24.34
-> 12.00 bps) and TIGHTER for vsa_no_demand (`spread_atr_max` 0.55 -> 0.40,
1533 -> 248 trades, edge 0.61 -> 7.96 bps). Finding that region IS the search
a walk-forward performs, so there is no shortcut to it — only the free
feasibility prune above, and the honest rejection test afterwards.

The sweep code below is retained because it answers a narrower question
cleanly (how does edge respond to one parameter, at fixed everything else),
but it is exploration, not a screen. Do not read its envelope as a forecast.

HISTORICAL — the envelope reasoning that attempts A-C were built on. The first version of this screen ran
   each cell at `configs/strategy.yaml` defaults, on the assumption that
   tuning generally raises per-trade edge so a default-param ceiling would be
   a lower bound. Calibration killed that assumption: against their WFO-tuned
   references, default params moved the ceiling by 1.95x (auction_reclaim_5m,
   8.03 vs 4.11), 0.11x (vsa_no_demand_5m, 0.29 vs 2.61) and 0.52x
   (vp_breakout_1m). Ordering survived; scale did not, across an 18x range.

   The mechanism is visible in the trade counts. Per-trade edge is almost
   entirely a function of how selective the entry is, and total gross edge
   (n_trades x edge_bps) is roughly CONSERVED across settings within a signal
   — 1363 vs 1332 bps-trades for auction_reclaim's two settings, 1521 vs 1975
   for vsa_no_demand, 1793 vs 1444 for vp_breakout. Selectivity redistributes
   a fixed pool of edge into fewer, fatter trades; it does not create edge.
   A single fixed setting therefore measures whatever selectivity the yaml
   happens to encode, not the signal's potential.

   So the screen sweeps the one parameter per signal that trades frequency for
   conviction and reports the best ceiling that still leaves enough trades to
   be statistically testable. Spread cost is measured once per cell rather than
   per point: it is a property of the cost model and of a notional pinned to
   `max_notional_pct`, and it moved only 4-12% across the calibration pairs.
   That makes each additional selectivity point one replay instead of three.

Conservation also gives a free prediction worth checking against the sweep:
break-even needs edge_bps above the ~3.2 bps round-trip spread, so a signal
can only be rescued by selectivity if its total edge pool divided by 3.2
leaves an acceptable number of trades.

What this screen deliberately cannot tell you. The ceiling is silent on
out-of-sample consistency, which is where every cell that got as far as a WFO
actually died (wfo_go, Monte Carlo p5). A promising ceiling is permission to
spend compute, never evidence of an edge. It is a necessary condition only.
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

from python.analytics.execution_ceiling import (  # noqa: E402
    edge_bps_from_zero,
    spread_bps_from_runs,
    tier,
    verdict,
)
from python.analytics.volume_route_policy import time_stop_for  # noqa: E402
from python.backtest.intraday_engine import (  # noqa: E402
    IntradayBacktestConfig,
    IntradayBacktestReport,
    metrics_from_report,
    run_intraday_backtest,
)

GATE_REPORT = Path("backtests/reports/entry_hypothesis_gate_report.json")
STRATEGY_PATH = Path("configs/strategy.yaml")
OUT_JSON = Path("backtests/reports/ceiling_prescreen.json")
OUT_MD = Path("backtests/reports/ceiling_prescreen.md")

# Only the two cost settings the ceiling needs. The full study also runs a
# jointly-costed baseline, an impact_only leg and a size sweep; none of them
# enters this limit.
COST_VARIANTS = {
    "spread_only": {"half_spread_bps": 2.0, "impact_bps_per_participation": 0.0},
    "zero": {"half_spread_bps": 0.0, "impact_bps_per_participation": 0.0},
}

# The axis to sweep is the STOP DISTANCE, not an entry filter.
#
# The first attempt swept each signal's entry-selectivity filter
# (`min_rel_volume` for auction_reclaim, and so on) toward tighter values. That
# was wrong in both direction and choice, and the sweep said so: trade count
# collapsed 56 -> 21 -> 8 -> 2 while total edge fell 1363 -> 867 -> -4 -> 92
# bps-trades, i.e. tightening the entry filter DESTROYS edge instead of
# redistributing it, and nothing above the loosest setting had a testable
# sample.
#
# Checking what the WFO actually chose for auction_reclaim_5m points somewhere
# else: `min_rel_volume` 1.2 -> 1.0 and `min_wick_frac` 0.45 -> 0.40, i.e.
# LOOSER entry, together with `stop_atr_mult` 0.15 -> 0.25, i.e. a WIDER STOP.
# It moved away from selectivity, not toward it. That pair (56 trades at 24.34
# bps vs 111 at 12.00 bps) is also the one place total edge conserved tightly,
# to within 2%.
#
# Those two settings differ in three parameters at once, so attributing the
# conservation to the stop is a HYPOTHESIS, not a measurement — which is
# precisely what sweeping the stop alone now tests. The reason to bet on it:
# stop distance governs how long a trade lives and therefore how much of a
# move it can capture, which is the same quantity the timeframe ladder is
# reaching for, only more directly. It is also the one free parameter shared
# by all five signals, so the sweeps stay comparable across cells.
SELECTIVITY = {
    sig: ("stop_atr_mult", [0.15, 0.25, 0.40, 0.60, 0.90], True)
    for sig in ("auction_reclaim", "vsa_no_demand", "vsa_effort",
                "obv_divergence", "absorption_breakout")
}

# Trades per year below which clearing the ceiling stops meaning anything.
# The Monte Carlo p5 gate needs an annualized Sharpe near 3.0 over this
# 239-day window, and annualized Sharpe scales with the square root of the
# fraction of days traded, so an edge that only appears at 40 trades a year is
# clearing the cost bar into a statistical dead end.
MIN_USEFUL_TRADES = 100

# Signals that take chart_minutes and resample internally. The 1m-native
# session signals (orb_vwap, fvg_retest, sweep_reclaim, vwap_band_fade,
# vp_breakout, l2_absorption) have no chart knob, so they are not screenable
# at other timeframes.
RESAMPLEABLE = (
    "absorption_breakout", "auction_reclaim", "vsa_effort",
    "vsa_no_demand", "obv_divergence",
)

# Calibration first: these two have published WFO-tuned ceilings AND a mapped
# selectivity knob, so sweeping them is what makes everything below
# interpretable. vp_breakout_1m and fvg_retest_1m also have published
# ceilings but are 1m-native with no knob in SELECTIVITY, so they cannot
# calibrate this version of the screen.
CALIBRATION_CELLS = ("auction_reclaim_5m", "vsa_no_demand_5m")

# Longest FEASIBLE timeframe first. Per-trade edge grew ~10x going from 1m to
# 5m while cost stayed flat, so the remaining headroom is at longer charts —
# but `_feasibility` caps how far that can go, because a signal needing N
# closed bars cannot fire on a chart where one 390-minute session holds fewer
# than N. 60m is only reachable by auction_reclaim (needs 5 bars, gets 6);
# the 8-bar family tops out near 45m and is only comfortable at 30m or below;
# absorption_breakout needs 22 bars so 15m is already its limit. Infeasible
# combinations are dropped arithmetically rather than replayed.
#
# 30m and 45m are NOT in configs/entry_hypothesis_tests.yaml's charts list.
# They are screened here as exploration; promoting one into the matrix would
# mean adding it to that catalog first.
# Grouped as a complete timeframe LADDER per signal, best-understood signal
# first, so that stopping early still leaves a coherent answer to "does edge
# per trade keep growing with the decision chart?" rather than a scatter of
# unrelated points. `eligible_points` in _feasibility is per symbol per day,
# so even 1 point/day is ~4,800 opportunities across 20 symbols and a year —
# thin charts are not automatically trade-starved.
SCREEN_CELLS = (
    "auction_reclaim_60m", "auction_reclaim_45m", "auction_reclaim_30m",
    "auction_reclaim_15m",
    "vsa_no_demand_45m", "vsa_no_demand_30m", "vsa_no_demand_15m",
    "obv_divergence_45m", "obv_divergence_30m", "obv_divergence_15m",
    "vsa_effort_45m", "vsa_effort_30m", "vsa_effort_15m", "vsa_effort_5m",
    "absorption_breakout_15m", "absorption_breakout_5m",
)

_MAX_WARMUP_DAYS = 40

# Regular trading hours, 09:30-16:00. Signals are evaluated on
# `bars_so_far = bars_today.iloc[:i+1]` (intraday_engine.run_symbol_day), and
# `day_bars` is sliced per session date, so the bar history a signal can see
# NEVER crosses a session boundary. That makes the longest usable decision
# chart a hard function of how many bars the signal demands.
RTH_MINUTES = 390


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _min_decision_bars(signal_name: str, base_cfg: dict) -> int | None:
    """How many closed decision bars the signal needs before it can fire.

    Imported from the signal modules rather than copied, so a change there
    cannot silently invalidate this screen. Returns None when the requirement
    is not expressible as a constant.
    """
    if signal_name == "vsa_no_demand":
        from python.microstructure.signals.vsa_no_demand import _MIN_TRADE_BARS
        # max(_MIN_TRADE_BARS, vol_lookback + 3) when require_confirm (default).
        lookback = int(base_cfg.get("vol_lookback", 2) or 2)
        return max(int(_MIN_TRADE_BARS), lookback + 3)
    if signal_name == "vsa_effort":
        from python.microstructure.signals.vsa_effort import _MIN_TRADE_BARS
        return int(_MIN_TRADE_BARS)
    if signal_name == "obv_divergence":
        from python.microstructure.signals.obv_divergence import _MIN_TRADE_BARS
        lookback = int(base_cfg.get("lookback_bars", 8) or 8)
        return max(int(_MIN_TRADE_BARS), lookback)
    if signal_name == "auction_reclaim":
        from python.microstructure.signals.auction_reclaim import _MIN_TRADE_BARS
        return int(_MIN_TRADE_BARS)
    if signal_name == "absorption_breakout":
        # min_bars = max(level_lookback, volume_lookback, atr_period) + 2,
        # read off the evaluator's own defaults since none of the three is a
        # WFO free parameter.
        import inspect

        from python.microstructure.signals.absorption_breakout import (
            evaluate_absorption_breakout,
        )
        defaults = {
            name: param.default
            for name, param in inspect.signature(evaluate_absorption_breakout)
            .parameters.items()
        }
        vals = [
            int(base_cfg.get(k, defaults.get(k)) or defaults.get(k) or 0)
            for k in ("level_lookback", "volume_lookback", "atr_period")
        ]
        return max(vals) + 2
    return None


def _feasibility(signal_name: str, minutes: int, base_cfg: dict) -> dict:
    """Can this (signal, chart) combination fire at all, and how often?

    Pure arithmetic, no replay. A cell with zero eligible evaluation points is
    structurally dead — the screen would burn ~9 minutes to print zero trades,
    which is exactly the waste this whole exercise is meant to avoid.
    """
    min_bars = _min_decision_bars(signal_name, base_cfg)
    bars_per_session = RTH_MINUTES // max(int(minutes), 1)
    if min_bars is None:
        return {"bars_per_session": bars_per_session, "min_bars": None,
                "eligible_points": None, "feasible": True}
    eligible = bars_per_session - min_bars + 1
    return {
        "bars_per_session": bars_per_session,
        "min_bars": min_bars,
        "eligible_points": eligible,
        "feasible": eligible >= 1,
    }


def _parse_cell(key: str) -> tuple[str, int] | None:
    """'vsa_no_demand_15m' -> ('vsa_no_demand', 15)."""
    if not key.endswith("m") or "_" not in key:
        return None
    head, _, tail = key.rpartition("_")
    try:
        minutes = int(tail[:-1])
    except ValueError:
        return None
    return head, minutes


def _run(bars, signal_name, base_cfg, params, sig_keys, cfg,
         warmup_days, start_ts, end_ts) -> dict:
    """One single-path replay, mirroring optimize.build_intraday_backtest_fn's
    warmup slice and exit-time filter so figures stay comparable to WFO
    full-window numbers, but keeping trade objects so notional is available."""
    merged = {**base_cfg, **params}
    sig_params = {k: merged[k] for k in sig_keys if k in merged}
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


def _calibration_note(rows: list[dict]) -> str:
    """Does the selectivity envelope reach the WFO-tuned ceiling?

    This is the screen validating itself. The envelope sweeps ONE parameter
    per signal; a WFO searches several at once. If the envelope falls well
    short of the tuned reference, the knob is not the axis the WFO actually
    used, and every pending-cell number below understates its cell.
    """
    pairs = [
        (r["cell"], r["published_tuned_ceiling"],
         (r.get("limit") or {}).get("ceiling"))
        for r in rows
        if r.get("published_tuned_ceiling") is not None
        and (r.get("limit") or {}).get("ceiling") is not None
    ]
    if not pairs:
        return "沒有可比的校準點，下面的數字無法判斷可信度。"

    ratios = [got / tuned for _, tuned, got in pairs]
    lo, hi = min(ratios), max(ratios)
    lines = [
        f"- 包絡／調參比值介於 {lo:.2f} 至 {hi:.2f}（{len(pairs)} 個校準點）。",
    ]
    if lo >= 0.85:
        lines.append(
            "- 單一旋鈕的掃描就追回了 WFO 調參的天花板，"
            "所以下面的上包絡可以當成該格的合理上界來分配算力。"
        )
    elif lo >= 0.5:
        lines.append(
            f"- 包絡最多只追到調參值的 {lo:.2f} 倍，代表 WFO 還用了這個旋鈕以外的"
            "選擇性來源（例如停損距離同時改變了持有時間）。"
            "下面的數字要當成**低估**，只用 1.0 這條解析邊界否決。"
        )
    else:
        lines.append(
            f"- **包絡只追到調參值的 {lo:.2f} 倍，這個旋鈕沒有涵蓋 WFO 實際在用的軸線。**"
            "下面的排序仍有參考價值，但不能用來否決任何一格。"
        )
    return "\n".join(lines)


def _render_md(payload: dict) -> str:
    L: list[str] = [
        "# 天花板預篩：哪些格子值得花 WFO",
        "",
        f"- 產生時間：{payload['generated_at']}",
        f"- 視窗：{payload['window']}",
        "- 方法：每格掃描一個選擇性參數，每點一次零成本重播；價差每格量一次。無 WFO。",
        "- 基準參數：`configs/strategy.yaml`，掃描時只改該訊號的選擇性旋鈕",
        "",
        "一次完整 WFO 要 1 至 4.5 小時，一格的上包絡只要 5 次重播。",
        "`ceiling = edge_bps / spread_round_trip_bps`，低於 1.0 的格子",
        "**在任何部位大小下都過不了 PF 1.0**，因為價差那一塊無法用縮小部位規避。",
        "",
        "為什麼要掃描而不是取單一設定：每筆邊緣幾乎完全由進場的選擇性決定，",
        "而總邊緣（筆數 × 每筆邊緣 bps）在同一訊號內大致守恆。",
        "固定在 yaml 預設值上量到的，是那份 yaml 恰好編碼的選擇性，不是訊號的潛力。",
        "",
        "這道預篩只是必要條件。它完全沒有檢驗樣本外一致性，",
        "而那正是所有跑到 WFO 的格子真正陣亡的地方（`wfo_go`、MC p5）。",
        "上包絡高只代表值得花算力，不代表有邊緣。",
        "",
    ]

    calib = [r for r in payload.get("results", []) if r.get("is_calibration")]
    if calib:
        L += ["## 校準：上包絡能不能追回已調參的天花板", "",
              "這兩格有 WFO 調參後的已發布天花板可比。如果掃描的上包絡追不上它，",
              "就表示這個旋鈕沒有涵蓋 WFO 實際在用的那條選擇性軸線。", "",
              "| 格子 | 已發布（調參） | 掃描上包絡 | 包絡／調參 | 包絡筆數 |",
              "|---|---:|---:|---:|---:|"]
        for r in calib:
            lim = r.get("limit") or {}
            pub = r.get("published_tuned_ceiling")
            got = lim.get("ceiling")
            ratio = (got / pub) if (pub and got) else None
            L.append(
                f"| `{r['cell']}` | {_fmt(pub, '.2f')} | {_fmt(got, '.2f')} | "
                f"{_fmt(ratio, '.2f')} | {lim.get('n_trades_zero', '—')} |"
            )
        L += ["", _calibration_note(calib), ""]

    screened = [r for r in payload.get("results", []) if not r.get("is_calibration")]
    infeasible = [r for r in screened if r.get("skipped")]
    if infeasible:
        L += [
            "## 結構上不可能的組合（解析排除，未重播）", "",
            f"訊號只看得到當日盤中的棒（`intraday_engine.run_symbol_day` 以 "
            f"session date 切片），所以一個 {RTH_MINUTES} 分鐘的盤中能提供的決策棒數"
            "就是週期的硬上限。棒數不足的組合連一次都不可能觸發，直接算掉。", "",
            "| 格子 | 單日決策棒 | 訊號最低需求 | 可評估點 |",
            "|---|---:|---:|---:|",
        ]
        for r in infeasible:
            f = r.get("feasibility") or {}
            L.append(
                f"| `{r['cell']}` | {f.get('bars_per_session')} | "
                f"{f.get('min_bars')} | {f.get('eligible_points')} |"
            )
        L.append("")

    live = [r for r in screened if not r.get("skipped")]
    if live:
        ranked = sorted(
            live, key=lambda r: -((r.get("limit") or {}).get("ceiling") or -99),
        )
        L += [
            "## 上包絡結果（依可用天花板排序）", "",
            f"「可用」= 至少 {MIN_USEFUL_TRADES} 筆。更高的天花板如果只在幾十筆時出現，"
            "那不是候選，只是同一份不確定性的更小樣本。", "",
            "| 格子 | 圖 | 旋鈕 | 最佳設定 | 筆數 | 邊緣 bps | 價差 bps | 上包絡 | 分級 |",
            "|---|---:|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in ranked:
            if r.get("error"):
                L.append(f"| `{r['cell']}` | — | — | — | — | — | — | — | 失敗 |")
                continue
            lim = r.get("limit") or {}
            best = r.get("best_usable") or r.get("best_any") or {}
            knob = r.get("selectivity_knob") or ""
            L.append(
                f"| `{r['cell']}` | {r['chart_minutes']}m | `{knob}` | "
                f"{best.get(knob, '—')} | {best.get('n_trades', '—')} | "
                f"{_fmt(best.get('edge_bps'), '.2f')} | "
                f"{_fmt(r.get('spread_round_trip_bps'), '.2f')} | "
                f"**{_fmt(lim.get('ceiling') or best.get('ceiling'), '.2f')}** | "
                f"{r.get('tier', '—')} |"
            )
        L.append("")
        for r in ranked:
            if r.get("error"):
                L.append(f"- `{r['cell']}`：執行失敗 `{r['error']}`")
            else:
                L.append(f"- `{r['cell']}`：{r.get('verdict', '')}")
        L.append("")

        L += ["### 逐格選擇性掃描", ""]
        for r in ranked:
            if not r.get("sweep"):
                continue
            knob = r.get("selectivity_knob") or "knob"
            L += [f"`{r['cell']}`（價差 {_fmt(r.get('spread_round_trip_bps'), '.2f')} bps）", "",
                  f"| `{knob}` | 筆數 | 邊緣 bps | 上限 | 總邊緣 bps-trades |",
                  "|---:|---:|---:|---:|---:|"]
            for p in r["sweep"]:
                L.append(
                    f"| {p.get(knob)} | {p['n_trades']} | "
                    f"{_fmt(p.get('edge_bps'), '.2f')} | "
                    f"{_fmt(p.get('ceiling'), '.2f')} | "
                    f"{_fmt(p.get('total_edge_bps_trades'), ',.0f')} |"
                )
            pool = [p["total_edge_bps_trades"] for p in r["sweep"]
                    if p.get("total_edge_bps_trades")]
            if len(pool) >= 2:
                lo, hi = min(pool), max(pool)
                L += ["", f"- 總邊緣池在 {lo:,.0f} 至 {hi:,.0f} bps-trades 之間"
                          f"（離散 {((hi - lo) / hi):.0%}）。"
                          + ("守恆成立，所以選擇性只是把同一份邊緣重新分配。"
                             if (hi - lo) / hi <= 0.35 else
                             "**離散偏大，這格的選擇性旋鈕不只是重新分配邊緣，"
                             "可能同時改變了進場的性質。**")]
            L.append("")

        worth = [r for r in ranked if r.get("tier") in ("promising", "marginal")]
        L += ["## 建議的 WFO 佇列", ""]
        if worth:
            for i, r in enumerate(worth, 1):
                lim = r.get("limit") or {}
                L.append(
                    f"{i}. `{r['cell']}`（天花板 {_fmt(lim.get('ceiling'), '.2f')}、"
                    f"{lim.get('n_trades_zero')} 筆）"
                )
        else:
            L.append("沒有任何一格達到 marginal 以上。**不建議再花 WFO 算力在這批格子上。**")
        L.append("")
    return "\n".join(L) + "\n"


def _persist(payload: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_render_md(payload), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cells", default=",".join(SCREEN_CELLS),
        help="comma-separated <signal>_<minutes>m cells to screen",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="also re-measure the four published cells at DEFAULT params, to "
             "check that screening at defaults preserves the tuned ordering",
    )
    parser.add_argument("--start", default="2025-08-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--rerender", action="store_true")
    args = parser.parse_args()

    if args.rerender:
        payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        for row in payload.get("results", []):
            if row.get("error") or row.get("skipped") or not row.get("sweep"):
                continue
            usable = [p for p in row["sweep"]
                      if p.get("ceiling") is not None
                      and p["n_trades"] >= MIN_USEFUL_TRADES]
            best = max(usable, key=lambda p: p["ceiling"], default=None)
            row["best_usable"] = best
            row["limit"] = (
                {"edge_bps": best["edge_bps"],
                 "spread_round_trip_bps": row.get("spread_round_trip_bps"),
                 "ceiling": best["ceiling"],
                 "n_trades_zero": best["n_trades"]}
                if best else None
            )
            row["tier"] = tier((row["limit"] or {}).get("ceiling"))
            row["verdict"] = verdict(row["limit"])
        _persist(payload)
        print(f"Wrote {OUT_JSON}\nWrote {OUT_MD}")
        return 0

    strategy = _load_yaml(STRATEGY_PATH)
    published = {}
    if GATE_REPORT.exists():
        published = (json.loads(GATE_REPORT.read_text(encoding="utf-8"))
                     .get("cells") or {})
    study_path = Path("backtests/reports/execution_cost_study.json")
    tuned_ceilings: dict[str, float] = {}
    if study_path.exists():
        for row in json.loads(study_path.read_text(encoding="utf-8")).get("results", []):
            if (row.get("limit") or {}).get("ceiling") is not None:
                tuned_ceilings[row["cell"]] = row["limit"]["ceiling"]

    from run_intraday_backtest import SIGNAL_WARMUP_DAYS, _load_real_bars
    from python.backtest.optimize import SIGNAL_PARAM_KEYS
    from python.data.fixed_universe import load_universe_config

    universe = load_universe_config()
    load_start = str((pd.Timestamp(args.start)
                      - pd.Timedelta(days=_MAX_WARMUP_DAYS)).date())
    print(f"loading 1m bars [{load_start}, {args.end}) for "
          f"{len(universe['symbols'])} symbols...", flush=True)
    bars = _load_real_bars(universe["symbols"], load_start, args.end)
    if not bars:
        raise SystemExit("no cached 1m bars")
    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)

    # Calibration first — it decides whether the rest is interpretable.
    queue: list[tuple[str, bool]] = []
    if args.calibrate:
        queue += [(c, True) for c in CALIBRATION_CELLS]
    queue += [(c.strip(), False) for c in args.cells.split(",") if c.strip()]

    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "window": f"{start_ts.date()} .. {end_ts.date()} (end-exclusive)",
        "method": (
            "three single-path replays per cell at strategy.yaml default "
            "params; ceiling = edge_bps / spread_round_trip_bps"
        ),
        "cost_variants": COST_VARIANTS,
        "results": [],
    }

    for key, is_calibration in queue:
        parsed = _parse_cell(key)
        if not parsed:
            print(f"  !! {key}: cannot parse as <signal>_<minutes>m — skipped", flush=True)
            continue
        signal_name, minutes = parsed
        if signal_name not in SIGNAL_PARAM_KEYS:
            print(f"  !! {key}: unknown signal {signal_name!r} — skipped", flush=True)
            continue
        if minutes != 1 and signal_name not in RESAMPLEABLE:
            print(f"  !! {key}: {signal_name} is 1m-native, no chart knob — skipped",
                  flush=True)
            continue
        base_cfg = strategy.get(signal_name)
        if not base_cfg:
            print(f"  !! {key}: no strategy.yaml defaults — skipped", flush=True)
            continue

        warmup = SIGNAL_WARMUP_DAYS.get(signal_name, 1)
        sig_keys = SIGNAL_PARAM_KEYS[signal_name]
        feasible = _feasibility(signal_name, minutes, base_cfg)
        row: dict = {
            "cell": key,
            "signal": signal_name,
            "chart_minutes": minutes,
            "warmup_days": warmup,
            "params": "strategy.yaml defaults",
            "is_calibration": is_calibration,
            "feasibility": feasible,
        }
        if not feasible["feasible"]:
            row["skipped"] = (
                f"單一 {RTH_MINUTES} 分鐘盤中只有 {feasible['bars_per_session']} 根 "
                f"{minutes}m 決策棒，訊號至少要 {feasible['min_bars']} 根才可能觸發，"
                "所以這格結構上不可能有成交（不必重播）。"
            )
            row["tier"] = "infeasible"
            print(f"\n=== {key} ===\n    -- 跳過：{row['skipped']}", flush=True)
            payload["results"].append(row)
            _persist(payload)
            continue
        if is_calibration:
            row["published_tuned_ceiling"] = tuned_ceilings.get(key)
            row["published_tuned_params"] = (published.get(key) or {}).get(
                "candidate_params")
        tag = " [校準]" if is_calibration else ""
        print(f"\n=== {key}{tag} ({signal_name}, {minutes}m, warmup {warmup}d) ===",
              flush=True)

        knob = SELECTIVITY.get(signal_name)
        if knob is None:
            print(f"  !! {key}: no selectivity knob mapped — skipped", flush=True)
            continue
        knob_name, knob_values, tighter_is_larger = knob

        def cfg_for(**cost_kw) -> IntradayBacktestConfig:
            return IntradayBacktestConfig(
                chart_minutes=max(minutes, 1),
                time_stop_minutes=time_stop_for(max(minutes, 1)),
                **cost_kw,
            )

        try:
            # Spread cost is measured once, at the loosest setting. It is a
            # property of the cost model and of a notional pinned to
            # `max_notional_pct`, not of the signal, so it barely moves with
            # selectivity — 4%, 12% and 5% across the three calibration pairs.
            # Re-measuring it per point would triple the run for that.
            loose = {knob_name: knob_values[0]}
            zero_loose = _run(bars, signal_name, base_cfg, loose, sig_keys,
                              cfg_for(**COST_VARIANTS["zero"]),
                              warmup, start_ts, end_ts)
            spread_loose = _run(bars, signal_name, base_cfg, loose, sig_keys,
                                cfg_for(**COST_VARIANTS["spread_only"]),
                                warmup, start_ts, end_ts)
            spread_bps = spread_bps_from_runs(zero_loose, spread_loose)
            row["spread_round_trip_bps"] = spread_bps
            row["selectivity_knob"] = knob_name
            print(f"    價差 {_fmt(spread_bps, '.2f')} bps"
                  f"（在 {knob_name}={knob_values[0]} 量一次）", flush=True)

            sweep: list[dict] = []
            for i, value in enumerate(knob_values):
                zero = zero_loose if i == 0 else _run(
                    bars, signal_name, base_cfg, {knob_name: value}, sig_keys,
                    cfg_for(**COST_VARIANTS["zero"]), warmup, start_ts, end_ts)
                edge_bps = edge_bps_from_zero(zero)
                n = int(zero["n_trades"])
                point = {
                    knob_name: value,
                    "n_trades": n,
                    "edge_bps": edge_bps,
                    "ceiling": (edge_bps / spread_bps)
                    if (edge_bps is not None and spread_bps) else None,
                    "total_edge_bps_trades": (edge_bps * n)
                    if edge_bps is not None else None,
                }
                sweep.append(point)
                print(f"    {knob_name}={value:<6} n={n:6d} "
                      f"邊緣={_fmt(edge_bps, '6.2f')} bps "
                      f"上限={_fmt(point['ceiling'], '5.2f')}", flush=True)
            row["sweep"] = sweep

            # The envelope: best ceiling that still leaves enough trades to be
            # testable. A higher ceiling reached at 30 trades a year is not a
            # candidate, it is a smaller sample of the same uncertainty.
            usable = [p for p in sweep
                      if p["ceiling"] is not None
                      and p["n_trades"] >= MIN_USEFUL_TRADES]
            best_any = max((p for p in sweep if p["ceiling"] is not None),
                           key=lambda p: p["ceiling"], default=None)
            best = max(usable, key=lambda p: p["ceiling"], default=None)
            row["best_usable"] = best
            row["best_any"] = best_any
            row["limit"] = (
                {"edge_bps": best["edge_bps"],
                 "spread_round_trip_bps": spread_bps,
                 "ceiling": best["ceiling"],
                 "n_trades_zero": best["n_trades"]}
                if best else None
            )
            row["tier"] = tier((row["limit"] or {}).get("ceiling"))
            row["verdict"] = verdict(row["limit"])
            if best:
                print(f"    -> 上包絡 {best['ceiling']:.2f}"
                      f"（{knob_name}={best[knob_name]}、{best['n_trades']} 筆）"
                      f" [{row['tier']}]", flush=True)
            elif best_any:
                row["tier"] = "too_few_trades"
                print(f"    -> 只有在 {best_any['n_trades']} 筆（< "
                      f"{MIN_USEFUL_TRADES}）時才到 {best_any['ceiling']:.2f}，"
                      "樣本不足以檢驗", flush=True)
            if is_calibration and row.get("published_tuned_ceiling"):
                print(f"    -> 已發布（調參）"
                      f"{row['published_tuned_ceiling']:.2f}", flush=True)
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
