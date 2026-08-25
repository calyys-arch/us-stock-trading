"""
Does breadth fix the drawdown, or was the drawdown the period?

`daily_baseline_gates.py` found that on a daily horizon the gates pass on
edge, cost and statistics, and fail on ONE thing: a -48% drawdown against a
-25% limit. That universe is 15 names and every one is a semiconductor or
big-tech name, because `configs/universe.yaml` was built as
`manual_pick_technology_sector_top20_distinct_by_dollar_volume`. A -48%
drawdown is the expected consequence of holding one sector through 2022, so
the obvious question is whether spreading across sectors removes it.

The comparison has to hold the WINDOW fixed. Breadth in data/history arrives
in steps — 15 names in 2016, 33 from 2018, then 118 only from 2024 — so a
wider universe measured over its own longer history would be a different
PERIOD as well as a different universe, and a drawdown that shrank would not
say which one did it. Both arms therefore run 2018-06 to 2026-07, where a
genuinely cross-sector set exists: alongside the semis there are healthcare
(ALNY, MDT, MRK), energy (DVN, CEG), consumer (CMG, CVNA, KDP, NCLH, ORLY,
HLT), financials (PGR, SCHW, SPGI) and telecom (VZ).

  semis_15        the 15 names with full 2016 history, all one sector.
  all_available   every name with enough history, entering on its own first
                  date — 33 at the start, 120 by 2024.

The 84 names that begin in 2024 cannot help with the 2022 drawdown, which is
the drawdown that matters here. That is a limit on what this study can show,
not a detail: it means the test is really "do ~18 non-semi names added in
2018 change 2022", and the answer is reported as such.

Return and cost mechanics are imported from run_daily_baseline_gates so both
studies price trades the same way and the numbers stay comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.backtest.monte_carlo import MonteCarloValidator  # noqa: E402
from scripts.run_daily_baseline_gates import (  # noqa: E402
    _max_dd, _profit_factor, _sharpe, load_panel, run_wfo, strategy_returns,
    TRADING_DAYS,
)

OUT_JSON = ROOT / "backtests/reports/diversification_study.json"
OUT_MD = ROOT / "backtests/reports/diversification_study.md"

WINDOW_START = "2018-06-01"
# The drawdown under examination, isolated so it can be read on its own.
STRESS_WINDOW = ("2021-11-01", "2023-01-31")
MAX_DD_LIMIT = -0.25


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, default=2000)
    args = ap.parse_args()

    wide = load_panel(min_rows=2500)          # the 15 full-history names
    broad = load_panel(min_rows=250)          # everything with real history
    start = pd.Timestamp(WINDOW_START)

    arms = {
        "semis_15": wide.loc[wide.index >= start],
        "all_available": broad.loc[broad.index >= start],
    }

    cells = {}
    for name, panel in arms.items():
        panel = panel.dropna(axis=1, how="all")
        net, n_trips = strategy_returns(panel, "buy_hold")
        mc = MonteCarloValidator(n_sims=args.sims, seed=42).run(
            [float(v) for v in net.tolist()])
        wfo = run_wfo(panel, "buy_hold")
        stress = net.loc[(net.index >= STRESS_WINDOW[0])
                         & (net.index <= STRESS_WINDOW[1])]
        dd = _max_dd(net)
        counts = panel.notna().sum(axis=1)
        cells[name] = {
            "n_symbols_total": int(panel.shape[1]),
            "n_symbols_at_start": int(counts.iloc[0]),
            "n_symbols_at_end": int(counts.iloc[-1]),
            "n_days": int(len(net)),
            "sharpe_annualized": _sharpe(net),
            "cagr": float((1 + net).prod() ** (TRADING_DAYS / len(net)) - 1),
            "max_drawdown": dd,
            "stress_max_drawdown": _max_dd(stress) if len(stress) else None,
            "profit_factor": _profit_factor(net),
            "mc_p5_sharpe": float(mc.sharpe.p5),
            "wfo_decision": wfo["decision"],
            "wfo_positive_folds": wfo["positive_folds"],
            "wfo_required_folds": wfo["required_positive_folds"],
            "drawdown_gate_pass": dd >= MAX_DD_LIMIT,
        }
        c = cells[name]
        print(f"\n== {name} ==", flush=True)
        print(f"   {c['n_symbols_at_start']} 檔起步 -> {c['n_symbols_at_end']} 檔"
              f"（共 {c['n_symbols_total']} 檔），{c['n_days']} 個交易日", flush=True)
        print(f"   Sharpe {c['sharpe_annualized']:.2f}  CAGR {c['cagr']:.1%}  "
              f"MaxDD {c['max_drawdown']:.1%}  PF {c['profit_factor']:.2f}",
              flush=True)
        print(f"   2022 壓力窗回撤 {c['stress_max_drawdown']:.1%}", flush=True)
        print(f"   MC p5 {c['mc_p5_sharpe']:+.2f}   WFO {c['wfo_decision']} "
              f"({c['wfo_positive_folds']}/{c['wfo_required_folds']} 需求)",
              flush=True)
        print(f"   回撤閘門 {'PASS' if c['drawdown_gate_pass'] else 'FAIL'} "
              f"(上限 {MAX_DD_LIMIT:.0%})", flush=True)

    payload = {
        "run_at": datetime.utcnow().isoformat(),
        "window_start": WINDOW_START,
        "stress_window": list(STRESS_WINDOW),
        "max_dd_limit": MAX_DD_LIMIT,
        "cells": cells,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_render(payload), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")
    return 0


def _render(p: dict) -> str:
    a, b = p["cells"]["semis_15"], p["cells"]["all_available"]
    L = [
        "# 分散化能不能解決回撤",
        "",
        f"兩組都跑 {p['window_start']} 起、同一個窗，因為 `data/history` 的廣度是"
        "分段出現的（2016 年 15 檔、2018 年 33 檔、2024 年才 118 檔）。"
        "若讓寬宇宙用它自己較長的歷史，改變的就同時是宇宙和期間，"
        "回撤變小也說不出是哪一個造成的。",
        "",
        "| | `semis_15`（單一類股） | `all_available`（跨產業） |",
        "|---|---:|---:|",
        f"| 檔數（起 → 迄） | {a['n_symbols_at_start']} → {a['n_symbols_at_end']} | "
        f"{b['n_symbols_at_start']} → {b['n_symbols_at_end']} |",
        f"| 年化 Sharpe | {a['sharpe_annualized']:.2f} | {b['sharpe_annualized']:.2f} |",
        f"| CAGR | {a['cagr']:.1%} | {b['cagr']:.1%} |",
        f"| 最大回撤 | {a['max_drawdown']:.1%} | {b['max_drawdown']:.1%} |",
        f"| 2022 壓力窗回撤 | {a['stress_max_drawdown']:.1%} | {b['stress_max_drawdown']:.1%} |",
        f"| 獲利因子 | {a['profit_factor']:.2f} | {b['profit_factor']:.2f} |",
        f"| MC p5 Sharpe | {a['mc_p5_sharpe']:+.2f} | {b['mc_p5_sharpe']:+.2f} |",
        f"| WFO | {a['wfo_decision']} | {b['wfo_decision']} |",
        f"| 回撤閘門（{p['max_dd_limit']:.0%}） | "
        f"{'PASS' if a['drawdown_gate_pass'] else 'FAIL'} | "
        f"{'PASS' if b['drawdown_gate_pass'] else 'FAIL'} |",
        "",
        "## 這個檢驗能說與不能說的",
        "",
        "2024 年才開始的那 84 檔對 2022 年的回撤毫無作用，而 2022 正是要解決的那次回撤。"
        "所以這實際上是在問「2018 年加進來的約 18 檔非半導體標的，能不能改變 2022」，"
        "上表也應該這樣讀。宇宙本身仍帶倖存者偏誤（名單取自 2026 年的流動性快照）。",
        "",
    ]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
