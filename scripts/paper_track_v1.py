"""
Forward paper record for Strategy V1. No broker, no orders -- an honest ledger.

    .venv/bin/python scripts/paper_track_v1.py --init --capital 100000
    .venv/bin/python scripts/paper_track_v1.py            # run daily/weekly
    .venv/bin/python scripts/paper_track_v1.py --report

    # Parallel VIX-overlay ledger (separate 250-day clock, starts empty):
    .venv/bin/python scripts/paper_track_v1.py --init --capital 100000 \
        --journal-path data/paper_v1_vix/journal.jsonl --vix-overlay
    .venv/bin/python scripts/paper_track_v1.py \
        --journal-path data/paper_v1_vix/journal.jsonl --vix-overlay

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

WHAT IT COSTS TO TRADE, PRICED THE WAY A BROKER PRICES IT. This ledger used to
charge a flat 4bps of notional. That is a fair model of spread and says nothing
about commission, which is billed per ORDER: IBKR Pro Fixed charges $0.005 a
share, minimum $1.00 an order, capped at 1% of trade value. At retail size the
minimum is most of the bill -- entering 120 positions of $817 costs $120 in
minimums against $39 of spread -- and the omission was worth about 0.7% a year
at $100k. Both are now charged, per order, by
python/portfolio/broker_costs.py.

WHICH LIMB BINDS, BECAUSE IT IS NOT THE OBVIOUS ONE. Entry is governed by the
$1.00 floor. Small rebalances are governed by the 1% CAP, so a 10-cent
adjustment costs a tenth of a cent, not a dollar. The money goes on the middle:
an order over $100 but under 200 shares pays the full $1.00, which on a $200
drift correction is 50bps of the value moved. An earlier note in this repo
applied the floor to everything and claimed a 1.44%-a-year drag; that was wrong
by roughly 4x.

AND THAT CAP IS A PROPERTY OF THE BROKER, NOT OF COMMISSION. Everything above
holds because IBKR lets its 1% cap override the per-order minimum. Futu, priced
in the same module and reachable by setting `broker: futu` in
configs/broker.yaml, resolves the same conflict the opposite way -- its own
schedule says the minimum wins -- and stacks two minimums ($0.99 commission plus
$1.00 platform fee) for a $1.99 floor no small order can escape. The same $25
correction costs $0.25 here and $1.99 there. Two conclusions on this page invert
under that schedule: fractional shares stop being the largest available
improvement and become a 1.23pp-a-year loss, because fractional targets generate
an order in every name every month and there is no cap to make them cheap, while
integer rounding accidentally suppresses them. Do not carry these numbers to
another broker without re-running scripts/run_retail_cost_study.py --broker.

SO THE MONTHLY CONSTITUENT REBALANCE MOSTLY DOES NOT HAPPEN ANY MORE.
DRIFT_BAND_BPS refuses any drift correction whose commission would exceed 10bps
of the value it moves, which at this account size means most of them. Measured
over 2018-2026 at $100k with fractional shares, that took fees from $6,060 to
$2,473 and monthly orders from 117 to 1. It is worth +0.42pp of CAGR in saved
fees; the run also picked up +0.96pp from letting weights drift, which is one
window's luck and is not part of the case for doing this.

The band NEVER applies to entering or to an exposure change. Those orders are
small by nature -- a tenth off every position -- so a notional floor blocks
precisely the trades that defend against a drawdown. Applying it to everything
cost 2.8pp of CAGR in measurement by silently disabling the volatility brake.
Because exposure changes always rebalance in full, the brake doubles as the
constituent rebalancer, which is why dropping the monthly reset does not let
the book concentrate: the largest single position peaked at 4.4% against a 2.5%
target over eight years.

THE BAND ANCHORS ON THE APPLIED TARGET, which it did not always. This ledger
used to measure the 10% rebalance band against `invested / equity` -- the
exposure the account had drifted to -- while the backtest measured it against
the last applied target. Same name, different rule: over the same panel the two
paths agreed to 0.015 on average but differed by up to 0.294, and because the
drifted anchor depends on every price move since the last trade, two accounts
running this from the same date would separate permanently. Both now call
python/portfolio/strategy_v1.py:band_exposure, and
tests/test_exposure_band_is_one_rule.py replays this ledger against
`exposure_path` day by day to keep them together. Each row still records
`exposure_held` and `exposure_drift` so the wander is visible, but nothing
decides on it.

WHAT TO EXPECT, NOW THAT EXECUTION IS PRICED. On the full 120-name book at
$100k over 2018-2026, share-level with real commission
(backtests/reports/retail_cost_study.json): CAGR 11.24%, Sharpe 0.873 with
fractional shares and the drift band; 9.86%/0.784 with fractional and no band;
9.12%/0.771 with whole shares and no band. Integer-share rounding alone costs
0.75pp a year at this size by stranding 4.8% of the account in cash, which is
more than commission does -- fractional shares are the single largest
improvement available and this ledger assumes them.

WHAT WOULD FALSIFY THE STRATEGY, stated before the data arrives so it cannot
be rationalized later. The walk-forward said Sharpe 0.91, profit factor 1.17,
max drawdown -24.6% over 1,646 out-of-sample days. Over any comparable stretch
here, a drawdown worse than -30%, or a realized Sharpe below zero, is the
strategy failing rather than noise. Profit factor is deliberately NOT one of
these: on a daily return series PF >= 1 is algebraically identical to
sum(returns) >= 0, so it would restate "the account is down" as if it were an
independent test. Below a few hundred days no verdict is available in either
direction, and --report refuses to imply one.

THE VIX OVERLAY IS A SEPARATE LEDGER, NOT A SWITCH ON THIS ONE. `--vix-overlay`
plus `--journal-path` runs the exact same day-by-day replay against a second,
independent journal that starts on its own --init date with its own 250-day
clock -- it does not touch, resume from, or backfill the primary ledger.
Multiplying by python/portfolio/risk_controls.py's VIX-bracket scalar happens
AFTER band_exposure() has already decided the volatility-target exposure for
the day; the VIX scalar is not itself banded, because it can change on any day
VIX crosses a bracket and that change must be traded and costed honestly, not
absorbed by a band designed for a different rule. Neither flag has any effect
unless passed explicitly, and the default journal path and default trading
logic are unchanged when they are not.
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

from python.portfolio.broker_costs import load_schedule  # noqa: E402
from python.portfolio.risk_controls import vix_exposure_scalar  # noqa: E402
from python.portfolio.strategy_v1 import (  # noqa: E402
    COLLAPSE, DRIFT_BAND_BPS, band_exposure, bucket_weighted_gross, load_prices,
    load_universe, target_weights,
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
BACKTEST = {"sharpe": 0.91, "profit_factor": 1.17, "max_drawdown": -0.246,
            "oos_days": 1646, "source": "backtests/reports/risk_bucket_study.json",
            "key": "wfo.bucket_equal"}
# Below this many days, dispersion swamps any signal and no verdict is offered.
MIN_DAYS_FOR_VERDICT = 250
# How long a missing price may be carried forward before the run stops. The
# universe is frozen and liquid, so a gap this long is a data fault, not a
# delisting, and marking off a week-old price is not defensible.
MAX_CARRY_DAYS = 5


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


def _marks(upto: pd.DataFrame, day: pd.Timestamp
           ) -> tuple[pd.Series, dict[str, int]]:
    """Last known close for every symbol, plus how stale each one is.

    `load_prices` does no forward-fill, so a symbol whose vendor fetch lagged by
    a day is NaN on that row while the other 119 are not. Valuing off the raw
    row dropped those names from the sum entirely -- marking them at ZERO, not
    at their last close. One commodity name is 2.5% of this book, so a single
    failed fetch printed a phantom 2.5% loss that reversed the next day; and if
    the rebalance band tripped on that day, the book was resized against the
    understated equity and real shares were sold to match a fake number.
    """
    carried = upto.ffill().loc[day]
    row = upto.loc[day]
    position = int(upto.index.get_indexer([day])[0])
    age: dict[str, int] = {}
    for symbol in upto.columns:
        if pd.notna(row.get(symbol)):
            continue
        last_seen = upto[symbol].last_valid_index()
        age[symbol] = (position + 1 if last_seen is None
                       else position - int(upto.index.get_indexer([last_seen])[0]))
    return carried, age


def _equity(holdings: dict[str, float], prices: pd.Series, cash: float
            ) -> tuple[float, float]:
    """Value the book. Every held name must have a mark; there is no honest way
    to value one at zero, so this refuses rather than inventing a loss."""
    unmarked = [s for s in holdings
                if s not in prices.index or not np.isfinite(prices[s])]
    if unmarked:
        raise ValueError(
            f"沒有可用價格卻持有：{', '.join(sorted(unmarked))}。"
            f"估值為零會憑空造出虧損，所以這裡拒絕結算。")
    invested = sum(sh * float(prices[s]) for s, sh in holdings.items())
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
             weights: dict[str, float], budget: float,
             ceiling: float | None = None
             ) -> tuple[dict[str, float], float, float]:
    """Allocate `budget` dollars across `weights`; return (holdings, invested,
    cost).

    Cost is spread plus commission, and commission is priced PER ORDER because
    that is how a broker charges it. The ledger used to charge a flat 4bps of
    notional, which models spread but is silent on the per-order minimum -- and
    at retail size the minimum is most of the bill.

    WHICH broker comes from configs/broker.yaml, and it is not a detail. The
    per-share rates are nearly identical but the precedence is not: IBKR's 1%
    cap overrides its $1.00 floor, Futu's $1.99 floor overrides its 0.5% cap.
    A $25 drift correction therefore costs $0.25 at one and $1.99 at the other.

    With `ceiling` set, an order whose commission would exceed that share of
    the value it moves is not placed and the existing line is carried instead.
    """
    cost = 0.0
    new: dict[str, float] = {}
    schedule = load_schedule()

    def charge(delta: float, price: float) -> None:
        nonlocal cost
        cost += schedule.charge(delta, price)
        cost += abs(delta) * price * (ONE_WAY_COST_BPS / 10_000.0)

    for symbol, weight in weights.items():
        price = float(prices.get(symbol, float("nan")))
        if not np.isfinite(price) or price <= 0:
            # No fresh price: carry the existing line rather than dumping it at
            # a stale mark.
            if symbol in holdings:
                new[symbol] = holdings[symbol]
            continue
        shares = round(max(budget, 0.0) * weight / price, 6)
        delta = shares - holdings.get(symbol, 0.0)
        notional = abs(delta) * price
        if ceiling is not None and notional > 0 and \
                notional < schedule.min_notional_for_ceiling(price, ceiling):
            new[symbol] = holdings.get(symbol, 0.0)
            continue
        new[symbol] = shares
        charge(delta, price)
    for symbol, shares in holdings.items():
        if symbol not in new and shares:
            price = float(prices.get(symbol, float("nan")))
            if np.isfinite(price):
                charge(shares, price)
    invested = sum(sh * float(prices[s]) for s, sh in new.items()
                   if s in prices.index and np.isfinite(prices[s]))
    return new, invested, cost


def _trade_to(holdings: dict[str, float], prices: pd.Series,
              weights: dict[str, float], equity: float, exposure: float,
              ceiling: float | None = None
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
                                           target - cost, ceiling)
        if abs(cost_now - cost) < 1e-9:
            cost = cost_now
            break
        cost = cost_now
    cash = equity - invested - cost
    if cash < 0.0:
        new, invested, cost = _size_to(holdings, prices, weights,
                                       target - cost + cash, ceiling)
        cash = equity - invested - cost
    return new, max(cash, 0.0), cost


def last_closed_session(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """The most recent date whose US close has definitely printed.

    The ledger must never mark against a partial bar. Requesting today while
    the US market is open returns one, and because a day is only pending once,
    re-running later does not repair it -- the intraday print is frozen into an
    append-only journal as if it were a close.

    The previous rule was "ask for yesterday", which was safe but cost a day
    more than intended: this machine's local date is a timezone ahead of New
    York, so "yesterday" was often the session before last. Deciding in US
    Eastern removes that, and the ledger stops running two sessions behind.

    Weekends walk back to Friday. Holidays are not modelled and do not need to
    be -- asking for a date with no session simply returns no bar.
    """
    now = (now or pd.Timestamp.now(tz="America/New_York"))
    if now.tzinfo is None:
        now = now.tz_localize("America/New_York")
    day = now.normalize()
    # 16:00 is the close; allow it to settle before trusting the print.
    if now.hour < 17:
        day -= pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day.tz_localize(None)


def _top_up() -> None:
    """Bring the cache to the last closed US session before advancing."""
    from python.data.price_cache import top_up_cached_panel

    symbols, _ = load_universe()
    print("更新價格 ...", flush=True)
    # yfinance treats `end` as EXCLUSIVE, so asking for the session itself
    # returns everything up to the day before it. The +1 is what makes the
    # request mean what it says.
    want = last_closed_session()
    result = top_up_cached_panel(symbols, want + pd.Timedelta(days=1))
    print(f"  目標交易日 {want.date()}")
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
              f" — 持有中的會延用最後已知收盤價，超過 "
              f"{MAX_CARRY_DAYS} 個交易日就停止推進")
    print()


def _fetch_vix(panel_start: pd.Timestamp, panel_end: pd.Timestamp) -> pd.Series:
    """One-shot ^VIX close series covering the whole panel span, for
    --vix-overlay.

    VIX is a single symbol, not the 120-name universe, so it does not need
    _top_up()'s cache/retry machinery: one fetch_daily_bars call covers
    whatever this run needs to replay, and that call costs one HTTP round
    trip regardless of how wide the window is -- there is no saving in
    narrowing it to just the pending days, and starting near the panel's own
    first date (like scripts/run_v1_vix_study.py does) gives the per-day
    lookup something to fall back on if a session or two is missing.

    Raises whatever fetch_daily_bars raises (network failure, no data, etc).
    The caller decides what "can't get VIX" means for the ledger; this
    function does not swallow the failure or invent a fallback value.
    """
    from python.simulation.hist_data_us import fetch_daily_bars

    start = (panel_start - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    # yfinance treats `end` as EXCLUSIVE; +1 day so the last session is
    # included, same convention as _top_up()'s `want + pd.Timedelta(days=1)`.
    end = (panel_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"更新 VIX（{start} 到 {end}）...", flush=True)
    df = fetch_daily_bars("^VIX", start, end)
    vix = df["close"].rename("VIX")
    print(f"  ^VIX：{len(vix)} 日，{vix.index[0].date()} 到 {vix.index[-1].date()}\n")
    return vix


def _vix_on_or_before(vix: pd.Series, day: pd.Timestamp) -> float:
    """Last known ^VIX close on or before `day`; NaN if none exists.

    This IS the ffill this ledger is willing to do: carry the last real
    print forward. It is not willing to carry forward across a gap that
    starts before any real print exists -- that returns NaN, and the caller
    stops the run rather than assuming the market was calm (scalar 1.0),
    which is exactly the assumption this whole mechanism exists to avoid
    making blindly.
    """
    eligible = vix.loc[:day]
    return float(eligible.iloc[-1]) if len(eligible) else float("nan")


def _run(args: argparse.Namespace) -> int:
    symbols, bucket_of = load_universe()
    panel = load_prices(symbols)
    journal = _read()
    # getattr, not args.vix_overlay: tests/test_exposure_band_is_one_rule.py
    # (and possibly other callers) build a bare `Args` stand-in with only the
    # attributes THEY need, predating this flag. Defaulting to False here
    # keeps every such caller's behavior exactly what it was before this flag
    # existed, instead of an AttributeError.
    vix_overlay = getattr(args, "vix_overlay", False)

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
    # The band's anchor. Older journal rows predate this field, so fall back to
    # the target that was in force -- not to `exposure_held`, which is the
    # drifted actual and is exactly the anchor this replaced.
    applied = last.get("exposure_applied")
    if applied is None:
        applied = last.get("exposure_target")
    applied = None if applied is None else float(applied)
    # The VIX overlay's OWN anchor: the last exposure this run actually traded
    # to, AFTER the VIX-bracket scalar. Deliberately separate from `applied`
    # above -- that one is band_exposure()'s anchor and must stay pre-VIX, or
    # the vol-target band (designed for drifting weights) would silently
    # absorb VIX regime changes it was never asked to gate. Only meaningful
    # under --vix-overlay; None/absent otherwise.
    applied_final = last.get("exposure_applied_final")
    applied_final = None if applied_final is None else float(applied_final)

    pending = panel.index[panel.index > last_date]
    if not len(pending):
        print(f"帳本已到 {last_date.date()}，價格也只到 "
              f"{panel.index[-1].date()}，沒有新交易日。")
        if args.no_update:
            print("（本次用 --no-update 跳過了價格更新）")
        else:
            print("下一個美股收盤後再跑就會推進。")
        return 0

    vix_series = None
    if vix_overlay:
        try:
            vix_series = _fetch_vix(panel.index[0], panel.index[-1])
        except Exception as exc:
            print(f"  VIX 抓取失敗，停在這裡：{exc}")
            print("  --vix-overlay 需要每個交易日的 ^VIX 值才能算曝險縮放，"
                  "寧可停在這裡也不要假設 scalar=1.0（那等於假設市場平靜，"
                  "恰好是這個機制最不該在不確定時做的假設）。")
            return 1

    print(f"帳本在 {last_date.date()}，價格到 {panel.index[-1].date()}，"
          f"補 {len(pending)} 個交易日\n")

    for day in pending:
        upto = panel.loc[:day]
        prices, stale = _marks(upto, day)
        too_stale = {s: n for s, n in stale.items()
                     if n > MAX_CARRY_DAYS and (s in holdings or n <= len(upto))}
        held_too_stale = {s: n for s, n in too_stale.items() if s in holdings}
        if held_too_stale:
            # Stop rather than write a mark nobody can defend. The journal is
            # append-only, so a bad row is permanent.
            print(f"  {day.date()}  停在這裡：{', '.join(sorted(held_too_stale))} "
                  f"已超過 {MAX_CARRY_DAYS} 個交易日沒有新價格。")
            print(f"  先修好價格資料再跑，不要讓一筆無法辯護的紀錄永久寫進帳本。")
            break
        if stale:
            carried = {s: n for s, n in stale.items() if s in holdings}
            if carried:
                print(f"  {day.date()}  延用舊價：" + "，".join(
                    f"{s}（{n} 日前）" for s, n in sorted(carried.items())))

        gross, _ = bucket_weighted_gross(upto, bucket_of, "equal",
                                         VOL_LOOKBACK, REBALANCE_BAND,
                                         ONE_WAY_COST_BPS)
        realized, want = _wanted_exposure(gross, target_vol)
        equity, invested = _equity(holdings, prices, cash)
        held = invested / equity if equity > 0 else 0.0

        # Constituents drift with prices; the backtest silently re-weighted
        # them daily for free, so this pulls them back monthly and pays for it.
        monthly = (day.year, day.month) != (last_date.year, last_date.month)
        # Band against the last APPLIED target, the same anchor the backtest and
        # the order sheet use. This used to compare against `held` --
        # invested/equity, the drifted actual -- which made the ledger's
        # exposure path diverge from the measured one by up to 0.294 and made
        # the divergence depend on price history since the last trade.
        target, band_tripped = band_exposure(want, applied, REBALANCE_BAND)
        if not holdings:
            target, band_tripped = want, True

        # VIX overlay: multiply the ALREADY-BANDED target by that day's
        # VIX-bracket scalar. The scalar itself is never banded -- it can
        # change any day VIX crosses a bracket, and that change must be
        # traded and costed for real, not absorbed by the 10% vol-target
        # band, which is a different rule for a different signal.
        vix_today = vix_scalar = target_final = float("nan")
        if vix_overlay:
            vix_today = _vix_on_or_before(vix_series, day)
            if not np.isfinite(vix_today):
                print(f"  {day.date()}  停在這裡：抓不到 {day.date()} 或更早的 "
                      f"^VIX 值（ffill 也補不到）。VIX 疊加需要每天的 VIX 值才能"
                      f"算縮放，寧可停在這裡也不要假設 scalar=1.0（那等於假設"
                      f"市場平靜，恰好是這個機制最不該在不確定時做的假設）。")
                break
            vix_scalar = vix_exposure_scalar(vix_today)
            target_final = target * vix_scalar

        decision_target = target_final if vix_overlay else target

        action, cost = "hold", 0.0
        trade_now = False
        if vix_overlay:
            # `band_tripped` / `monthly` decide the vol-target layer; VIX gets
            # its own trigger because it can move on a day neither of those
            # fires, and that move must not be suppressed by a band built for
            # a different signal. Epsilon guards against re-trading on
            # floating-point noise when the scalar is unchanged.
            stale_applied = (applied_final is None or
                             not np.isfinite(applied_final))
            vix_drifted = stale_applied or (
                abs(decision_target - applied_final) >
                1e-6 * max(abs(applied_final), 1e-9))
            trade_now = np.isfinite(decision_target) and (
                band_tripped or monthly or vix_drifted)
        else:
            trade_now = np.isfinite(decision_target) and (band_tripped or monthly)

        if trade_now:
            # Decide the label from the state BEFORE trading; afterwards
            # holdings is always non-empty and every entry reads "rebalance".
            first_time = not holdings
            weights = target_weights(
                [s for s in symbols if s in panel.columns], bucket_of, "bucket")
            # Entering and moving exposure go through unbanded; a month that
            # only drifted pays the band and mostly does nothing.
            ceiling = None if (band_tripped or first_time) else DRIFT_BAND_BPS
            holdings, cash, cost = _trade_to(holdings, prices, weights,
                                             equity, decision_target, ceiling)
            action = "enter" if first_time else "rebalance"
            equity, invested = _equity(holdings, prices, cash)
            held = invested / equity if equity > 0 else 0.0
            applied = target
            if vix_overlay:
                applied_final = target_final
        elif not np.isfinite(target):
            action = "warmup"

        if action == "enter":
            reason = "首次進場"
        elif action == "rebalance":
            if band_tripped:
                reason = "曝險帶觸發"
            elif monthly:
                reason = "月度成分再平衡"
            elif vix_overlay:
                reason = "VIX 分級變動觸發"
            else:
                reason = "月度成分再平衡"
        elif action == "warmup":
            reason = "波動樣本不足"
        else:
            reason = "無動作"

        entry = {
            "date": str(day.date()), "action": action,
            "equity": round(equity, 2), "cash": round(cash, 2),
            "exposure_held": round(held, 4),
            "exposure_target": None if not np.isfinite(want) else round(want, 4),
            # The band anchor, carried forward so the next run reproduces the
            # backtest's path. `exposure_held` drifts with prices and is kept
            # only as a diagnostic of how far the book has wandered from it.
            "exposure_applied": None if applied is None or not np.isfinite(applied)
                                else round(applied, 4),
            "exposure_drift": None if applied is None or not np.isfinite(applied)
                              else round(held - applied, 4),
            "realized_vol": None if not np.isfinite(realized) else round(realized, 4),
            "cost": round(cost, 2), "target_vol": target_vol,
            "reason": reason,
            "holdings": {k: round(v, 6) for k, v in holdings.items()},
        }
        if vix_overlay:
            entry["vix_level"] = (None if not np.isfinite(vix_today)
                                  else round(float(vix_today), 4))
            entry["vix_scalar"] = (None if not np.isfinite(vix_scalar)
                                   else round(float(vix_scalar), 4))
            entry["exposure_target_pre_vix"] = (None if not np.isfinite(target)
                                                else round(target, 4))
            entry["exposure_applied_final"] = (
                None if applied_final is None or not np.isfinite(applied_final)
                else round(applied_final, 4))
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

    # Start the equity path AT the init row, not after it. Excluding it dropped
    # the init-to-first-mark move -- the one that contains the entry cost -- from
    # Sharpe, profit factor and drawdown, while the headline total return still
    # counted it. Two numbers, two different starting points.
    equity = pd.Series([j["equity"] for j in journal],
                       index=pd.to_datetime([j["date"] for j in journal]))
    equity = equity[~equity.index.duplicated(keep="last")]
    marks = [j for j in journal if j["action"] != "init"]
    rets = equity.pct_change().dropna()
    start_equity = float(journal[0]["equity"])
    total_cost = sum(float(j.get("cost") or 0.0) for j in journal)
    n_rebal = sum(1 for j in journal if j["action"] in ("enter", "rebalance"))

    print(f"策略 V1 紙上紀錄  {journal[0]['date']} -> {journal[-1]['date']}")
    print(f"  交易日數        {len(marks)}")
    print(f"  起始／現在權益   ${start_equity:,.0f} -> ${equity.iloc[-1]:,.0f}"
          f"   （{equity.iloc[-1] / start_equity - 1:+.2%}）")
    drawdown, profit_factor = _max_dd(rets), _profit_factor(rets)
    if len(rets) > 2:
        print(f"  Sharpe（年化）   {_sharpe(rets):.2f}")
        print(f"  獲利因子         {profit_factor:.2f}")
        print(f"  最大回撤         {drawdown:.1%}")
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
        # "Profit factor below 1.0" used to sit here as if it were a second,
        # independent condition. On a daily return series it is not: PF >= 1 is
        # algebraically identical to sum(returns) >= 0, so it only restated
        # "the account is down". Replaced with a test that has its own content
        # -- realized Sharpe far enough below the walk-forward's 0.91 that the
        # gap is unlikely to be dispersion at this sample length.
        dd = drawdown
        realized_sharpe = _sharpe(rets)
        breached = []
        if dd < -0.30:
            breached.append(f"回撤 {dd:.1%} 劣於 -30%")
        if realized_sharpe < 0.0:
            breached.append(f"實現 Sharpe {realized_sharpe:.2f} 為負，"
                            f"回測為 {BACKTEST['sharpe']}")
        if breached:
            print(f"  觸及事先寫定的否證條件：{'；'.join(breached)}。")
        else:
            print(f"  仍在事先寫定的否證條件之內"
                  f"（實現 Sharpe {realized_sharpe:.2f}，回撤 {dd:.1%}）。")
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
    ap.add_argument("--journal-path", type=str, default=None,
                    help="override the journal file (default: "
                         "data/paper_v1/journal.jsonl). Not passing this "
                         "leaves the primary ledger's path, and every run "
                         "against it, byte-for-byte unchanged.")
    ap.add_argument("--vix-overlay", action="store_true",
                    help="scale the banded exposure target by "
                         "python/portfolio/risk_controls.py's VIX-bracket "
                         "scalar before trading. Default off; off means the "
                         "exact pre-existing logic, unchanged.")
    args = ap.parse_args()
    if args.journal_path:
        # Module-level global by design: _append/_read/_run/--init all read
        # and write through it, and this process runs exactly one command
        # (init, advance, or report) before exiting, so mutating it once here
        # -- before any of those run -- is safe and keeps every call site
        # below untouched. Not passing --journal-path never reaches this
        # branch, so the default path is never touched.
        global JOURNAL
        path = Path(args.journal_path)
        JOURNAL = path if path.is_absolute() else ROOT / path
    if args.report:
        return _report()
    if not args.no_update:
        _top_up()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
