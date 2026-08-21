"""1 / 5 / 15 / 60-minute decision-chart A/B for volume-book signals.

Same yaml defaults, same universe, same cost model. `chart_minutes` and
`time_stop_minutes` are engine/structural knobs (IntradayBacktestConfig),
not Chan free parameters and not in SIGNAL_PARAM_KEYS / yaml grids.

Fair time-stop (so 15m/60m are not cut mid-bar by a 10-minute wall clock):
    time_stop_minutes = max(10, 2 * chart_minutes)
    1m → 10, 5m → 10 (same as the old 1m-vs-5m report), 15m → 30, 60m → 120.

Run order is 5 → 15 → 60 → 1 so the new evidence lands first; 1m is last
because it is the memory/CPU hog. Sequential by design.

Usage:
    .venv/bin/python -u scripts/compare_chart_minutes.py
    .venv/bin/python -u scripts/compare_chart_minutes.py --start 2026-01-02 --end 2026-04-01
    .venv/bin/python -u scripts/compare_chart_minutes.py --resume
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from python.backtest.intraday_engine import (
    SIGNAL_PARAM_KEYS,
    IntradayBacktestConfig,
    IntradayBacktestReport,
    metrics_from_report,
    run_intraday_backtest,
)

SIGNALS = ["vsa_no_demand", "obv_divergence"]
# Fast new evidence first; 1m last (fattest).
CHART_MINUTES_RUN = (5, 15, 60, 1)
CHART_MINUTES_TABLE = (1, 5, 15, 60)
STRATEGY_PATH = Path("configs/strategy.yaml")
REPORT_PATH = Path("backtests/reports/chart_minutes_1m_5m_15m_60m.md")
REPORT_JSON_PATH = Path("backtests/reports/chart_minutes_1m_5m_15m_60m.json")


def time_stop_for(chart_minutes: int) -> int:
    return max(10, 2 * int(chart_minutes))


def _result_key(signal_name: str, chart_minutes: int) -> str:
    return f"{signal_name}_{chart_minutes}m"


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_real_bars(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    from python.data.intraday_cache import get_cached_intraday_panel

    panel = get_cached_intraday_panel(symbols, start, end)
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        if sym in panel.index.get_level_values("code"):
            out[sym] = panel.xs(sym, level="code").sort_index()
    return out


def _summarize(metrics: dict, n_wins: int, chart_minutes: int) -> dict:
    n = int(metrics.get("n_trades", 0))
    win_rate = (n_wins / n) if n else 0.0
    daily = metrics.get("daily_returns") or []
    return {
        "n_trades": n,
        "signals_emitted": int(metrics.get("signals_emitted", 0)),
        "signals_filled": int(metrics.get("signals_filled", 0)),
        "total_net_pnl": float(metrics.get("total_net_pnl", 0.0)),
        "gross_pnl": float(metrics.get("gross_pnl", 0.0)),
        "total_costs": float(metrics.get("total_costs", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "profit_factor_gross": float(metrics.get("profit_factor_gross", 0.0)),
        "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "win_rate": win_rate,
        "n_wins": n_wins,
        "n_days": int(metrics.get("n_days", 0) or len(daily)),
        "chart_minutes": int(chart_minutes),
        "time_stop_minutes": time_stop_for(chart_minutes),
    }


def _run_one(
    bars_by_symbol,
    signal_name: str,
    base_cfg: dict,
    chart_minutes: int,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict:
    keys = SIGNAL_PARAM_KEYS[signal_name]
    params = {k: base_cfg[k] for k in keys if k in base_cfg}
    stop = time_stop_for(chart_minutes)
    engine_cfg = IntradayBacktestConfig(
        chart_minutes=chart_minutes,
        time_stop_minutes=stop,
    )
    warmup_start = start_ts - pd.Timedelta(days=1)
    sliced = {}
    for symbol, bars in bars_by_symbol.items():
        window = bars.loc[(bars.index >= warmup_start) & (bars.index < end_ts)]
        if not window.empty:
            sliced[symbol] = window
    print(
        f"    running {signal_name} chart={chart_minutes}m time_stop={stop}m ...",
        flush=True,
    )
    report = run_intraday_backtest(sliced, signal_name, params, engine_cfg)
    in_window = [t for t in report.trades if start_ts <= t.exit_time < end_ts]
    filtered = IntradayBacktestReport(
        trades=in_window,
        signals_emitted=report.signals_emitted,
        signals_filled=report.signals_filled,
    )
    metrics = metrics_from_report(filtered, engine_cfg.capital)
    n_wins = sum(1 for t in in_window if t.net_pnl > 0)
    return _summarize(metrics, n_wins, chart_minutes)


def _ratio(a: float, b: float) -> str:
    if b == 0:
        return "n/a"
    return f"{a / b:.2f}×"


def _pf_cell(r: dict) -> str:
    return f"{r['profit_factor']:.3f}"


def _vs_5m_row(signal: str, minutes: int, results: dict) -> str:
    a = results[_result_key(signal, minutes)]
    b = results[_result_key(signal, 5)]
    if a["n_trades"] == 0 and a["signals_emitted"] == 0:
        return (
            f"| `{signal}` | {minutes}m / 5m | "
            f"0.00× (0 / {b['n_trades']}) | "
            f"n/a（0 筆，非績效） | "
            f"n/a vs {b['profit_factor']:.3f} | "
            f"無單 | 無單（結構） | 無單（結構） |"
        )
    more_trades = "較多" if a["n_trades"] > b["n_trades"] else (
        "相同" if a["n_trades"] == b["n_trades"] else "較少"
    )
    more_pnl = "較好" if a["total_net_pnl"] > b["total_net_pnl"] else (
        "相同" if a["total_net_pnl"] == b["total_net_pnl"] else "較差"
    )
    pf_note = (
        "較好" if a["profit_factor"] > b["profit_factor"] else (
            "相同" if a["profit_factor"] == b["profit_factor"] else "較差"
        )
    )
    return (
        f"| `{signal}` | {minutes}m / 5m | "
        f"{_ratio(a['n_trades'], b['n_trades'])} "
        f"({a['n_trades']} / {b['n_trades']}) | "
        f"{_ratio(a['total_net_pnl'], b['total_net_pnl'])} "
        f"(${a['total_net_pnl']:,.0f} / ${b['total_net_pnl']:,.0f}) | "
        f"{a['profit_factor']:.3f} vs {b['profit_factor']:.3f} | "
        f"{more_trades} | {more_pnl} | {pf_note} |"
    )


def _fifteen_vs_five_sentence(results: dict) -> str:
    bits = []
    for signal in SIGNALS:
        a = results.get(_result_key(signal, 15))
        b = results.get(_result_key(signal, 5))
        if a is None or b is None:
            bits.append(f"`{signal}`：15m 尚未跑完。")
            continue
        if a["n_trades"] > b["n_trades"]:
            trade_word = "變多"
        elif a["n_trades"] < b["n_trades"]:
            trade_word = "變少"
        else:
            trade_word = "持平"
        if a["profit_factor"] > b["profit_factor"]:
            pf_word = "變好"
        elif a["profit_factor"] < b["profit_factor"]:
            pf_word = "變差"
        else:
            pf_word = "持平"
        bits.append(
            f"`{signal}`：15m 交易{trade_word}（{a['n_trades']} vs {b['n_trades']}），"
            f"淨 PF {pf_word}（{a['profit_factor']:.3f} vs {b['profit_factor']:.3f}）。"
        )
    return " ".join(bits)


def _sixty_note(results: dict) -> list[str]:
    lines = []
    zeros = []
    for signal in SIGNALS:
        r = results.get(_result_key(signal, 60))
        if r is None:
            lines.append(f"- `{signal}` 60m：尚未跑完。")
            continue
        n = r["n_trades"]
        emitted = r["signals_emitted"]
        if n == 0 and emitted == 0:
            zeros.append(signal)
            lines.append(
                f"- `{signal}` 60m：0 筆成交、0 筆發出（結構裝不進一天，不是績效輸贏）。"
            )
        else:
            lines.append(
                f"- `{signal}` 60m：成交 {n}、發出 {emitted}、"
                f"淨 PnL ${r['total_net_pnl']:,.0f}、淨 PF {r['profit_factor']:.3f}。"
                f"有單就照實讀數字，不要用「一天裝不滿 8 根」當藉口忽略結果。"
            )
    if zeros:
        lines.append(
            "- 本輪**沒有改訊號檔**去放寬 8 根門檻或改成跨 session 累積；"
            "0 筆是現況 API 的結構限制，不是策略失敗。"
        )
    return lines


def _render(results: dict, data_label: str, start: str, end: str, pending: list[str]) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# 1 / 5 / 15 / 60 分鐘決策圖對照",
        "",
        "## 設定",
        "",
        f"- 視窗：`[{start}, {end})`（腳本預設三個月，與 `scripts/compare_chart_minutes.py` 的 `--start/--end` 預設對齊）。",
        "- 舊報告 `backtests/reports/chart_minutes_1m_vs_5m.md` 寫的是 `[2026-03-02, 2026-04-01)`，且 **時間停損固定 10 牆鐘分鐘**。那份數字保留不覆寫；本表視窗更長、15m/60m 的時間停損依規則加長，**不可直接跟舊表逐格比**。",
        f"- 宇宙：{data_label}",
        "- 參數：`configs/strategy.yaml` 預設（不是 WFO winner）。",
        "- `chart_minutes` 與 `time_stop_minutes` 都是引擎結構旋鈕，不是 Chan free param，沒有寫進 `SIGNAL_PARAM_KEYS` / yaml grid。",
        "- 時間停損規則：`time_stop_minutes = max(10, 2 * chart_minutes)` → 1m=10、5m=10、15m=30、60m=120（各約兩根決策棒；1m/5m 與舊報告同一時間停損）。",
        "- Warmup：視窗前一天；`run_intraday_backtest` 之後只計視窗內出場。",
        "- 這**不是** research GO，**不改** live 決策圖（live 仍 5m），不改 `LIVE_SIGNALS`，不開 `auto_execute`。",
        f"- 產生時間：{generated}",
        "",
        "## 兩訊號 × 四週期",
        "",
        "| Signal | Chart | Time stop | Trades | Emitted | Net PnL | Gross PnL | Costs | PF (net) | PF (gross) | Win rate | Sharpe | Max DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for signal in SIGNALS:
        for minutes in CHART_MINUTES_TABLE:
            key = _result_key(signal, minutes)
            if key not in results:
                lines.append(
                    f"| `{signal}` | {minutes}m | {time_stop_for(minutes)} | — | — | — | — | — | — | — | — | — | — |"
                )
                continue
            r = results[key]
            pf = r["profit_factor"]
            pfg = r["profit_factor_gross"]
            lines.append(
                f"| `{signal}` | {minutes}m | {r['time_stop_minutes']} | {r['n_trades']} | {r['signals_emitted']} | "
                f"${r['total_net_pnl']:,.0f} | ${r['gross_pnl']:,.0f} | ${r['total_costs']:,.0f} | "
                f"{pf:.3f} | {pfg:.3f} | {r['win_rate']:.1%} | {r['sharpe_ratio']:+.3f} | "
                f"{r['max_drawdown']:.1%} |"
            )
    have_5m = all(_result_key(s, 5) in results for s in SIGNALS)
    lines.extend(["", "## 相對 5m 的倍數", ""])
    if not have_5m:
        lines.append("5m 尚未跑完，倍數表暫缺。")
    else:
        lines.append("| Signal | 對照 | 交易數倍數 | 淨 PnL 倍數 | PF net | 交易 | 淨 PnL | PF |")
        lines.append("|---|---|---:|---:|---|---|---|---|")
        for signal in SIGNALS:
            for minutes in (1, 15, 60):
                key = _result_key(signal, minutes)
                if key not in results:
                    lines.append(f"| `{signal}` | {minutes}m / 5m | — | — | — | — | — | — |")
                    continue
                lines.append(_vs_5m_row(signal, minutes, results))
    lines.extend([
        "",
        "## 15m vs 5m（一句話）",
        "",
        _fifteen_vs_five_sentence(results),
        "",
        "## 60m 結構限制",
        "",
        "RTH 約 6.5 小時，60m 一天最多約 6 根收完的棒。訊號 `_MIN_TRADE_BARS = 8`，"
        "OBV `lookback_bars` 預設 8（WFO 還到 10），而且 OBV 只累今日 session。"
        "因此 yaml 預設下 60m 很可能 0 筆——那是「session-only + 8 根門檻裝不進一天」，"
        "不是「1 小時沒邊緣」。15m 一天約 26 根，現有門檻跑得動，才是這輪真正的新證據。",
        "",
    ])
    lines.extend(_sixty_note(results))
    if pending:
        lines.extend([
            "",
            "## 尚未跑完",
            "",
            *[f"- `{k}`" for k in pending],
        ])
    lines.extend([
        "",
        "## 備註",
        "",
        "- 1 分鐘圖用同一套「根數」回看，牆鐘窗口更短（8 根 = 8 分鐘 vs 5m 的 40 分鐘 vs 15m 的 120 分鐘）。",
        "- 同一標的一次一倉：訊號變多不會線性變成成交變多。",
        "- 舊 1m vs 5m 報告（固定 10 分鐘停損、可能不同視窗）仍在 `chart_minutes_1m_vs_5m.md`。",
        "- 這不是 research GO，不改 `auto_execute` / `LIVE_SIGNALS`；live 決策圖仍是 5m。",
        "",
    ])
    return "\n".join(lines)


def _load_resume(path: Path, start: str, end: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    window = payload.get("window") or {}
    if window.get("start") != start or window.get("end") != end:
        raise SystemExit(
            f"--resume json window {window} != [{start}, {end}) — delete "
            f"{path} or pass matching --start/--end"
        )
    return dict(payload.get("results") or {})


def _persist(results: dict, data_label: str, start: str, end: str, pending: list[str]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "window": {"start": start, "end": end},
        "data_label": data_label,
        "time_stop_rule": "max(10, 2 * chart_minutes)",
        "time_stop_minutes": {str(m): time_stop_for(m) for m in CHART_MINUTES_TABLE},
        "params": "configs/strategy.yaml defaults (not WFO winners)",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "pending": pending,
        "results": results,
    }
    REPORT_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(_render(results, data_label, start, end, pending), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare 1m/5m/15m/60m charts for volume-book signals"
    )
    parser.add_argument("--start", default="2026-01-02")
    parser.add_argument("--end", default="2026-04-01")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip (signal, minutes) cells already present in the json report",
    )
    args = parser.parse_args()

    from python.data.fixed_universe import load_universe_config

    universe_cfg = load_universe_config()
    symbols = universe_cfg["symbols"]
    print(
        f"loading 1m cache for {len(symbols)} symbols [{args.start}, {args.end}) ...",
        flush=True,
    )
    bars_by_symbol = _load_real_bars(symbols, args.start, args.end)
    if not bars_by_symbol:
        raise SystemExit(
            f"no cached 1-minute bars for any universe symbol in [{args.start}, {args.end}]"
        )
    data_label = (
        f"fixed top-{universe_cfg['top_n']} universe "
        f"(computed_at={universe_cfg['computed_at']}), 1m bars via data/history_1m/"
    )
    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)
    strategy = _load_yaml(STRATEGY_PATH)

    planned = [_result_key(s, m) for m in CHART_MINUTES_RUN for s in SIGNALS]
    results: dict[str, dict] = _load_resume(REPORT_JSON_PATH, args.start, args.end) if args.resume else {}

    for minutes in CHART_MINUTES_RUN:
        for signal in SIGNALS:
            key = _result_key(signal, minutes)
            if key in results:
                r = results[key]
                print(
                    f"    skip {signal} {minutes}m (resume): trades={r['n_trades']} "
                    f"net=${r['total_net_pnl']:,.0f} pf={r['profit_factor']:.3f} "
                    f"win={r['win_rate']:.1%} time_stop={r.get('time_stop_minutes', time_stop_for(minutes))}",
                    flush=True,
                )
                continue
            base_cfg = strategy[signal]
            results[key] = _run_one(
                bars_by_symbol, signal, base_cfg, minutes, start_ts, end_ts,
            )
            r = results[key]
            print(
                f"    {signal} {minutes}m: trades={r['n_trades']} emitted={r['signals_emitted']} "
                f"net=${r['total_net_pnl']:,.0f} pf={r['profit_factor']:.3f} "
                f"win={r['win_rate']:.1%} time_stop={r['time_stop_minutes']}",
                flush=True,
            )
            pending = [k for k in planned if k not in results]
            _persist(results, data_label, args.start, args.end, pending)

    pending = [k for k in planned if k not in results]
    _persist(results, data_label, args.start, args.end, pending)
    print(REPORT_PATH.read_text(encoding="utf-8"))
    print(f"wrote {REPORT_PATH} and {REPORT_JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
