"""
Forward paper record for Strategy V1. No broker, no orders -- an honest ledger.

    .venv/bin/python scripts/paper_track_v1.py --init --capital 100000
    .venv/bin/python scripts/paper_track_v1.py            # run daily/weekly
    .venv/bin/python scripts/paper_track_v1.py --report

WHY THIS AND NOT IBKR PAPER. V1 holds for weeks and turns over a few dozen
times a year, so realistic fills are worth little here: cost is a rounding
error against the holding period. What is actually unknown is whether the
thing performs forward the way it did in the backtest. That needs a ledger
that starts today and accumulates, which needs no broker at all.

WHAT IT DOES. Replays every trading day since the last entry: marks holdings
at that day's close, recomputes the volatility brake, rebalances when a rule
fires, charges cost on everything it trades, and appends one line per day to
data/paper_v1/journal.jsonl. Running it weekly instead of daily fills the gap
in correctly rather than skipping days, and re-running on the same day is a
no-op.

THE RECORD STARTS EMPTY, ON PURPOSE. It does not replay history to
manufacture a track record -- that would be another backtest wearing a
different hat. Everything in the journal is dated after --init.

ONE DISCREPANCY WITH THE BACKTEST, MEASURED RATHER THAN FEARED. The backtest's
`gross_returns` re-weights constituents to equal-within-bucket EVERY day and
charges nothing for it, billing only changes in total exposure, so the measured
Sharpe of 0.89 assumed free daily rebalancing. Sizing that assumption: pulling
120 names back to equal-within-bucket costs 0.85% of the book in one-way
turnover per day, which at 4bps is 0.09% per YEAR against an 11.5% CAGR --
about 0.01 of Sharpe. Immaterial. This ledger still rebalances constituents
monthly and charges every dollar it trades, but the gap that creates against
the backtest is basis points, not a correction that changes any decision.

WHAT WOULD FALSIFY THE STRATEGY, stated before the data arrives so it cannot
be rationalized later. The walk-forward said Sharpe 0.89, profit factor 1.17,
max drawdown -24.6% over 1,646 out-of-sample days. Over any comparable stretch
here, a drawdown worse than -30%, or a profit factor below 1.0, is the
strategy failing rather than noise. Below a few hundred days no verdict is
available in either direction, and --report refuses to imply one.
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

from python.portfolio.strategy_v1 import (  # noqa: E402
    COLLAPSE, bucket_weighted_gross, load_prices, load_universe,
    target_weights,
)
from scripts.run_daily_baseline_gates import (  # noqa: E402
    TRADING_DAYS, _max_dd, _profit_factor, _sharpe,
)
from scripts.run_vol_target_study import (  # noqa: E402
    MAX_EXPOSURE, ONE_WAY_COST_BPS, REBALANCE_BAND, VOL_LOOKBACK,
)

JOURNAL = ROOT / "data/paper_v1/journal.jsonl"
TARGET_VOL = 0.15

# What the walk-forward measured, for --report to compare against rather than
# leaving the reader to look it up.
BACKTEST = {"sharpe": 0.89, "profit_factor": 1.17, "max_drawdown": -0.246,
            "oos_days": 1646, "source": "backtests/reports/risk_bucket_study.json"}
# Below this many days, dispersion swamps any signal and no verdict is offered.
MIN_DAYS_FOR_VERDICT = 250


def _short(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise. `relative_to` raises
    on anything outside ROOT, which crashed the whole run under a temp dir."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _append(entry: dict) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def _read() -> list[dict]:
    if not JOURNAL.exists():
        return []
    return [json.loads(line) for line in JOURNAL.read_text().splitlines()
            if line.strip()]


def _equity(holdings: dict[str, float], prices: pd.Series, cash: float
            ) -> tuple[float, float]:
    invested = sum(sh * float(prices[s]) for s, sh in holdings.items()
                   if s in prices.index and np.isfinite(prices[s]))
    return cash + invested, invested


def _wanted_exposure(gross: pd.Series, target_vol: float) -> tuple[float, float]:
    window = gross.tail(VOL_LOOKBACK)
    if len(window) < VOL_LOOKBACK:
        return float("nan"), float("nan")
    realized = float(window.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if realized <= 0:
        return realized, 0.0
    return realized, min(MAX_EXPOSURE, target_vol / realized)


def _size_to(holdings: dict[str, float], prices: pd.Series,
             weights: dict[str, float], budget: float
             ) -> tuple[dict[str, float], float, float]:
    """Allocate `budget` dollars across `weights`; return (holdings, invested,
    cost). Cost is charged on the notional actually traded, both sides."""
    traded_notional = 0.0
    new: dict[str, float] = {}
    for symbol, weight in weights.items():
        price = float(prices.get(symbol, float("nan")))
        if not np.isfinite(price) or price <= 0:
            # No fresh price: carry the existing line rather than dumping it at
            # a stale mark.
            if symbol in holdings:
                new[symbol] = holdings[symbol]
            continue
        shares = round(max(budget, 0.0) * weight / price, 6)
        traded_notional += abs(shares - holdings.get(symbol, 0.0)) * price
        new[symbol] = shares
    for symbol, shares in holdings.items():
        if symbol not in new and shares:
            price = float(prices.get(symbol, float("nan")))
            if np.isfinite(price):
                traded_notional += abs(shares) * price
    invested = sum(sh * float(prices[s]) for s, sh in new.items()
                   if s in prices.index and np.isfinite(prices[s]))
    return new, invested, traded_notional * (ONE_WAY_COST_BPS / 10_000.0)


def _trade_to(holdings: dict[str, float], prices: pd.Series,
              weights: dict[str, float], equity: float, exposure: float
              ) -> tuple[dict[str, float], float, float]:
    """Move to target weights at `prices`; return (holdings, cash, cost).

    The cost has to be funded out of the invested amount, not out of cash. At
    exposure 1.00 there is no cash by definition, so charging cost against it
    drove the balance negative -- the ledger was quietly borrowing to pay fees
    for a strategy whose whole premise is no leverage, and reporting exposures
    just over 1.0 as a result.

    Reserving the cost inside the budget is circular because cost depends on
    trade size, so it iterates to a fixed point. Rounding shares to 6 decimals
    leaves a sub-cent residual that can still land the wrong side of zero, so a
    final pass shrinks the budget by any overdraft. "No leverage" is either an
    invariant or it is not, and a loosened tolerance would only hide it.
    """
    target = equity * exposure
    cost = 0.0
    for _ in range(8):
        new, invested, cost_now = _size_to(holdings, prices, weights,
                                           target - cost)
        if abs(cost_now - cost) < 1e-9:
            cost = cost_now
            break
        cost = cost_now
    cash = equity - invested - cost
    if cash < 0.0:
        new, invested, cost = _size_to(holdings, prices, weights,
                                       target - cost + cash)
        cash = equity - invested - cost
    return new, max(cash, 0.0), cost


def _top_up() -> None:
    """Bring the cache to today's close before advancing the ledger."""
    from python.data.price_cache import top_up_cached_panel

    symbols, _ = load_universe()
    print("更新價格 ...", flush=True)
    result = top_up_cached_panel(symbols, pd.Timestamp.today().normalize())
    parts = [f"{len(result['topped_up'])} 檔延伸",
             f"{len(result['already_current'])} 檔已最新"]
    if result["readjusted"]:
        parts.append(f"{len(result['readjusted'])} 檔因除權/分割重抓")
    if result["not_cached"]:
        parts.append(f"{len(result['not_cached'])} 檔無快取")
    if result["failed"]:
        parts.append(f"{len(result['failed'])} 檔失敗")
    print("  " + "，".join(parts))
    if result["readjusted"]:
        print(f"  重抓：{', '.join(result['readjusted'])}")
    if result["failed"]:
        print(f"  !! 失敗：{', '.join(result['failed'])}"
              f" — 這些標的會用舊價格結算")
    print()


def _run(args: argparse.Namespace) -> int:
    symbols, bucket_of = load_universe()
    panel = load_prices(symbols)
    journal = _read()

    if args.init:
        if journal:
            print(f"帳本已存在（{len(journal)} 筆，起於 "
                  f"{journal[0]['date']}）。不覆寫。要重新開始請先手動刪除 "
                  f"{_short(JOURNAL)}。")
            return 1
        start = panel.index[-1]
        _append({"date": str(start.date()), "action": "init",
                 "equity": args.capital, "cash": args.capital,
                 "holdings": {}, "exposure_held": 0.0, "cost": 0.0,
                 "target_vol": args.target_vol,
                 "recorded_at": datetime.now().isoformat(),
                 "note": "紙上帳本建立；下一次執行才會依規則進場"})
        print(f"帳本建立於 {start.date()}，資金 ${args.capital:,.0f}")
        print(f"寫入 {_short(JOURNAL)}")
        print("\n下一步：等下一個交易日的價格進來後再跑一次（不帶 --init）。")
        return 0

    if not journal:
        print("還沒有帳本。先跑 --init --capital <金額>。")
        return 1

    last = journal[-1]
    holdings = {k: float(v) for k, v in (last.get("holdings") or {}).items()}
    cash = float(last["cash"])
    last_date = pd.Timestamp(last["date"])
    target_vol = float(last.get("target_vol") or args.target_vol)

    pending = panel.index[panel.index > last_date]
    if not len(pending):
        print(f"帳本已到 {last_date.date()}，價格也只到 "
              f"{panel.index[-1].date()}，沒有新交易日。")
        if args.no_update:
            print("（本次用 --no-update 跳過了價格更新）")
        else:
            print("下一個美股收盤後再跑就會推進。")
        return 0

    print(f"帳本在 {last_date.date()}，價格到 {panel.index[-1].date()}，"
          f"補 {len(pending)} 個交易日\n")

    for day in pending:
        upto = panel.loc[:day]
        prices = upto.loc[day]
        gross, _ = bucket_weighted_gross(upto, bucket_of, "equal",
                                         VOL_LOOKBACK, REBALANCE_BAND,
                                         ONE_WAY_COST_BPS)
        realized, want = _wanted_exposure(gross, target_vol)
        equity, invested = _equity(holdings, prices, cash)
        held = invested / equity if equity > 0 else 0.0

        # Constituents drift with prices; the backtest silently re-weighted
        # them daily for free, so this pulls them back monthly and pays for it.
        monthly = (day.year, day.month) != (last_date.year, last_date.month)
        band_tripped = (not holdings) or (
            np.isfinite(want)
            and abs(want - held) / max(held, 1e-9) > REBALANCE_BAND)

        action, cost = "hold", 0.0
        if np.isfinite(want) and (band_tripped or monthly):
            # Decide the label from the state BEFORE trading; afterwards
            # holdings is always non-empty and every entry reads "rebalance".
            first_time = not holdings
            weights = target_weights(
                [s for s in symbols if s in panel.columns], bucket_of, "bucket")
            holdings, cash, cost = _trade_to(holdings, prices, weights,
                                             equity, want)
            action = "enter" if first_time else "rebalance"
            equity, invested = _equity(holdings, prices, cash)
            held = invested / equity if equity > 0 else 0.0
        elif not np.isfinite(want):
            action = "warmup"

        entry = {
            "date": str(day.date()), "action": action,
            "equity": round(equity, 2), "cash": round(cash, 2),
            "exposure_held": round(held, 4),
            "exposure_target": None if not np.isfinite(want) else round(want, 4),
            "realized_vol": None if not np.isfinite(realized) else round(realized, 4),
            "cost": round(cost, 2), "target_vol": target_vol,
            "reason": ("首次進場" if action == "enter" else
                       "曝險帶觸發" if action == "rebalance" and band_tripped else
                       "月度成分再平衡" if action == "rebalance" else
                       "波動樣本不足" if action == "warmup" else "無動作"),
            "holdings": {k: round(v, 6) for k, v in holdings.items()},
        }
        _append(entry)
        last_date = day
        print(f"  {day.date()}  {action:<10} 權益 ${equity:>11,.0f}  "
              f"曝險 {held:>5.2f}  成本 ${cost:>8,.2f}  {entry['reason']}")

    print(f"\n寫入 {_short(JOURNAL)}")
    return 0


def _report() -> int:
    journal = _read()
    if len(journal) < 2:
        print("帳本太短，還沒有東西可報。")
        return 0

    marks = [j for j in journal if j["action"] != "init"]
    equity = pd.Series([j["equity"] for j in marks],
                       index=pd.to_datetime([j["date"] for j in marks]))
    rets = equity.pct_change().dropna()
    start_equity = float(journal[0]["equity"])
    total_cost = sum(float(j.get("cost") or 0.0) for j in journal)
    n_rebal = sum(1 for j in journal if j["action"] in ("enter", "rebalance"))

    print(f"策略 V1 紙上紀錄  {journal[0]['date']} -> {journal[-1]['date']}")
    print(f"  交易日數        {len(marks)}")
    print(f"  起始／現在權益   ${start_equity:,.0f} -> ${equity.iloc[-1]:,.0f}"
          f"   （{equity.iloc[-1] / start_equity - 1:+.2%}）")
    if len(rets) > 2:
        print(f"  Sharpe（年化）   {_sharpe(rets):.2f}")
        print(f"  獲利因子         {_profit_factor(rets):.2f}")
        print(f"  最大回撤         {_max_dd(equity.pct_change().dropna()):.1%}")
        print(f"  實現波動         "
              f"{rets.std(ddof=1) * np.sqrt(TRADING_DAYS):.1%} 年化")
    print(f"  再平衡次數       {n_rebal}")
    print(f"  累計成本         ${total_cost:,.2f}"
          f"   （{total_cost / start_equity:.2%} 起始權益）")

    print(f"\n  回測基準（{BACKTEST['oos_days']} 個樣本外日）："
          f"Sharpe {BACKTEST['sharpe']}, 獲利因子 {BACKTEST['profit_factor']}, "
          f"回撤 {BACKTEST['max_drawdown']:.1%}")
    if len(marks) < MIN_DAYS_FOR_VERDICT:
        print(f"  只有 {len(marks)} 個交易日，不足 {MIN_DAYS_FOR_VERDICT} 日。"
              f"這個長度下離散度遠大於訊號，")
        print(f"  兩個方向都不給結論——好看不算驗證，難看也不算否證。")
    else:
        dd = _max_dd(equity.pct_change().dropna())
        pf = _profit_factor(rets)
        if dd < -0.30 or pf < 1.0:
            print(f"  觸及事先寫定的否證條件（回撤劣於 -30% 或獲利因子 < 1.0）。")
        else:
            print(f"  仍在事先寫定的否證條件之內。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", action="store_true", help="create the journal")
    ap.add_argument("--capital", type=float, default=100_000.0,
                    help="starting equity for --init")
    ap.add_argument("--target-vol", type=float, default=TARGET_VOL,
                    help="volatility target recorded at --init")
    ap.add_argument("--report", action="store_true",
                    help="summarize the record so far")
    ap.add_argument("--no-update", action="store_true",
                    help="skip the price top-up and use the cache as-is")
    args = ap.parse_args()
    if args.report:
        return _report()
    if not args.no_update:
        _top_up()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
