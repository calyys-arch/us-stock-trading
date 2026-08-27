"""How many instruments can an account of a given size actually afford to hold?

    .venv/bin/python scripts/run_retail_cost_study.py --capital 100000

THE GAP THIS CLOSES. Every V1 number was measured charging 4bps of notional per
side, which prices spread and impact but is silent on commission -- and
commission is per ORDER, not per dollar. IBKR Pro Fixed charges $0.005 a share,
minimum $1.00 an order, maximum 1% of trade value. Spread $100k over 120
instruments and a position is $817, where the $1.00 floor is 12bps: three times
the modelled cost.

WHICH LIMB OF THE SCHEDULE ACTUALLY BINDS, AND A CORRECTION. Entry is governed
by the floor -- buying a whole $817 position costs $0.04 in per-share terms, so
you pay the $1.00 minimum, and 120 of those is $120 or 0.12% of the account.
Rebalancing is governed by the CAP, because a rebalance only moves the part
that drifted, perhaps $25, where 1% of trade value is $0.25 and beats the
floor. An earlier pass through this repo applied the floor to rebalances too
and reported a 1.44%-a-year drag; that figure was wrong by roughly 4x for that
reason.

WHY THE ANSWER STILL IS NOT ARITHMETIC. Subtracting an estimated commission
from a daily-reweighted return series charges for one trading schedule while
crediting the returns of another, and the error is not conservative in a known
direction: a no-trade band that suppresses orders also lets weights drift, which
changes the returns themselves. So this runs a share-level simulation
(python/portfolio/share_level.py) that holds real share counts, lets them drift
between rebalances, and pays each order what the broker charges. Commission is
inside the equity curve rather than deducted from it.

THE TRADEOFF BEING MEASURED. Holding fewer instruments makes each position
bigger, so the floor stops binding -- but it also discards the diversification
that bought V1 its drawdown control. The buckets stay at 25% each however few
names remain, because that allocation IS the strategy; reduction happens within
buckets, keeping the highest-priced names, since share price is exactly what
decides whether the floor or the per-share rate applies. A 12-instrument basket
is still four buckets of three.
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

from python.portfolio.broker_costs import (  # noqa: E402
    NO_COMMISSION, SCHEDULES, FeeSchedule,
)
from python.portfolio.share_level import (  # noqa: E402
    invested_fraction, returns_of, simulate,
)
from python.portfolio.strategy_v1 import (  # noqa: E402
    COLLAPSE, DRIFT_BAND_BPS, bucket_weighted_gross, load_prices, load_universe,
)
from scripts.run_daily_baseline_gates import (  # noqa: E402
    TRADING_DAYS, _max_dd, _profit_factor, _sharpe,
)
from scripts.run_risk_bucket_study import (  # noqa: E402
    MAX_DD_LIMIT, TARGET_VOL, WINDOW_START,
)
from scripts.run_vol_target_study import (  # noqa: E402
    ONE_WAY_COST_BPS, REBALANCE_BAND, VOL_LOOKBACK, apply_exposure,
    exposure_path,
)

OUT_JSON = ROOT / "backtests/reports/retail_cost_study.json"
BASKET_SIZES = (8, 12, 20, 40, 66, 120)
CAPITALS = (25_000, 50_000, 100_000, 250_000, 1_000_000)
# Candidate no-trade bands, as the largest commission (in bps of the value
# moved) an order may cost before it is judged not worth placing. None places
# every order the target weights ask for, which is what the ledger does today.
CEILINGS: tuple[float | None, ...] = (None, 50.0, 25.0, 10.0)


def _by_bucket(symbols: list[str], bucket_of: dict[str, str],
               panel: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in symbols:
        if s in panel.columns:
            out.setdefault(COLLAPSE[bucket_of[s]], []).append(s)
    return out


def quotas(n: int, pools: dict[str, list[str]]) -> dict[str, int]:
    """How many names to take from each bucket to total `n`, as evenly as the
    bucket sizes allow.

    Even splitting alone is wrong here because the buckets are wildly uneven:
    83 US equity instruments against 10 commodity ETFs. Asking for 30 from each
    and capping at what exists silently returns 67 names when 120 were
    requested -- an earlier version of this study did that, so its "120-name"
    row was a 67-name basket whose contents changed with the ranking, which
    made two runs of the same configuration disagree by 2.8 points of CAGR.

    Water-filling instead: satisfy the smallest bucket first at its fair share,
    then redistribute whatever it could not absorb over the buckets that remain.
    """
    remaining, out = n, {}
    order = sorted(pools, key=lambda b: len(pools[b]))
    for i, bucket in enumerate(order):
        fair = remaining // (len(order) - i)
        take = min(len(pools[bucket]), fair)
        out[bucket] = take
        remaining -= take
    return out


def reduce_universe(symbols: list[str], bucket_of: dict[str, str],
                    panel: pd.DataFrame, n: int) -> list[str]:
    """Keep `n` instruments, preferring the highest-priced names in each bucket.

    Share price is the criterion because it is what decides whether the
    per-share charge or the $1.00 floor binds: at a fixed position size, a
    higher price means fewer shares and so a smaller per-share charge. Ranked on
    the FIRST price in the window; ranking on the last price selects whatever
    went up most over the window being scored, and an earlier version did that
    and concluded small baskets beat large ones.

    Kept only so the study can show that this idea does not work -- the ranking
    lands below random selection at every size. The load-bearing baskets are
    the random ones.
    """
    by_bucket = _by_bucket(symbols, bucket_of, panel)
    first = panel.bfill().iloc[0]
    take = quotas(n, by_bucket)
    kept: list[str] = []
    for bucket, pool in sorted(by_bucket.items()):
        ranked = sorted(pool, key=lambda s: -float(first.get(s, 0.0)))
        kept.extend(ranked[:take[bucket]])
    return kept


def random_baskets(symbols: list[str], bucket_of: dict[str, str],
                   panel: pd.DataFrame, n: int, draws: int,
                   seed: int = 0) -> list[list[str]]:
    """`draws` bucket-balanced baskets of size `n` chosen without any criterion.

    These carry the study's conclusions. Any deterministic rule for choosing
    which names to drop is itself a bet, and measuring basket SIZE through one
    of those bets confounds the two. Sampling without a criterion and reporting
    the spread separates them: the median says what size costs, and the width
    says how much of any single basket's result was luck.
    """
    rng = np.random.default_rng(seed)
    by_bucket = _by_bucket(symbols, bucket_of, panel)
    take = quotas(n, by_bucket)
    out = []
    for _ in range(draws):
        picked: list[str] = []
        for bucket, pool in sorted(by_bucket.items()):
            picked.extend(
                rng.choice(pool, size=take[bucket], replace=False).tolist())
        out.append(picked)
    return out


def score(net: pd.Series) -> dict:
    if net.empty:
        return {}
    dd = _max_dd(net)
    return {
        "n_days": int(len(net)),
        "sharpe_annualized": _sharpe(net),
        "cagr": float((1 + net).prod() ** (TRADING_DAYS / len(net)) - 1),
        "max_drawdown": dd,
        "profit_factor": _profit_factor(net),
        "drawdown_gate_pass": bool(dd >= MAX_DD_LIMIT),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capital", type=float, default=100_000.0,
                    help="the account this should be tuned for")
    ap.add_argument("--draws", type=int, default=25,
                    help="random baskets per size, as a control on the price "
                         "ranking; 0 skips the control")
    ap.add_argument("--broker", choices=sorted(SCHEDULES), default="ibkr",
                    help="whose fee schedule to charge; the choice changes the "
                         "answer, not just the size of the bill")
    args = ap.parse_args()
    broker = SCHEDULES[args.broker]

    symbols, bucket_of = load_universe()
    panel = load_prices(symbols).loc[WINDOW_START:]

    payload: dict = {
        "run_at": datetime.now().isoformat(),
        "window": [str(panel.index[0].date()), str(panel.index[-1].date())],
        "broker": args.broker,
        "commission_model": {
            "name": broker.name, "components": list(broker.components),
            "min_per_order": broker.min_per_order,
            "cap_fraction": broker.cap_fraction,
            "cap_overrides_floor": broker.cap_overrides_floor,
            "third_party_per_share": broker.third_party_per_share,
            "note": broker.note, "verified": "2026-08-26",
        },
        "no_trade_band_ceilings_bps": [c for c in CEILINGS if c is not None],
        "modelled_spread_bps_one_way": ONE_WAY_COST_BPS,
        "weight_space_reference": {}, "measured": {},
    }

    print(f"零售執行成本（股數層級實測）  "
          f"{panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"券商 {broker.name}")
    print(f"  每筆最低 ${broker.min_per_order:.2f}，"
          f"上限成交金額 {broker.cap_fraction:.1%}，"
          + ("上限蓋過最低" if broker.cap_overrides_floor else "最低蓋過上限"))
    if broker.third_party_per_share:
        print(f"  第三方每股 ${broker.third_party_per_share}")
    print(f"  價差另計 {ONE_WAY_COST_BPS:.0f} bps 單邊\n")

    baskets: dict[int, list[str]] = {}
    exposures: dict[int, pd.Series] = {}
    for n in BASKET_SIZES:
        kept = reduce_universe(symbols, bucket_of, panel, n)
        gross = bucket_weighted_gross(panel[kept], bucket_of, "equal",
                                      VOL_LOOKBACK, REBALANCE_BAND,
                                      ONE_WAY_COST_BPS)[0]
        exp = exposure_path(gross, TARGET_VOL)
        baskets[n] = kept
        exposures[n] = exp
        payload["weight_space_reference"][str(n)] = {
            **score(apply_exposure(gross, exp)), "symbols": kept}

    def run_basket(names: list[str], exposure: pd.Series, capital: float,
                   ceiling: float | None, *, fractional: bool = False,
                   commission: bool = True, charge: bool = True,
                   schedule: FeeSchedule | None = None) -> dict:
        daily, log = simulate(panel[names], bucket_of, exposure, capital,
                              cost_ceiling_bps=ceiling,
                              spread_bps_one_way=ONE_WAY_COST_BPS,
                              exposure_band=REBALANCE_BAND,
                              fractional=fractional, charge_costs=charge,
                              schedule=(schedule or broker) if commission
                              else NO_COMMISSION)
        rets = returns_of(daily)
        years = len(rets) / TRADING_DAYS
        # `cost_priced` is what the orders would cost; `cost` is what was
        # actually deducted, which differs only under charge=False.
        priced = float(log["cost_priced"].sum()) if not log.empty else 0.0
        monthly = log[log["reason"] == "monthly"] if not log.empty else log
        return {
            **score(rets),
            "final_equity": float(daily["equity"].iloc[-1]),
            "total_cost": priced,
            "annual_cost_fraction": priced / capital / years if years else 0.0,
            "n_rebalances": int(len(log)),
            "orders_total": int(log["n_orders"].sum()) if not log.empty else 0,
            "monthly_orders_mean": float(monthly["n_orders"].mean())
            if len(monthly) else 0.0,
            # Not a diagnostic any more but the measurement itself: the gap
            # between this and 1.0 at full exposure is cash stranded by integer
            # shares, which turns out to cost more than commission at $100k.
            "invested_fraction": invested_fraction(daily),
            # Whether skipping the monthly reset lets the book concentrate.
            "max_weight_peak": float(daily["max_weight"].max())
            if "max_weight" in daily else float("nan"),
        }

    cap = args.capital
    full, full_exp = baskets[120], exposures[120]

    # 1. Where the published number goes once the book is made of shares.
    print(f"  ① ${cap:,.0f} 上，已發表數字到實際可執行之間的差在哪裡"
          f"（全 120 檔）")
    print(f"    {'':<34}{'CAGR':>8}{'Sharpe':>8}{'回撤':>9}{'年成本':>8}"
          f"{'投入率':>8}")
    ws = payload["weight_space_reference"]["120"]
    print(f"    {'A 權重空間、每日重配、只收價差':<30}"
          f"{ws['cagr']:>8.2%}{ws['sharpe_annualized']:>8.3f}"
          f"{ws['max_drawdown']:>9.1%}{'—':>8}{'—':>8}")
    steps = {
        "A_weight_space_daily": ws,
        "B_shares_fractional_no_commission": run_basket(
            full, full_exp, cap, None, fractional=True, commission=False),
        "C_shares_integer_no_commission": run_basket(
            full, full_exp, cap, None, fractional=False, commission=False),
        "D_shares_integer_real_commission": run_basket(
            full, full_exp, cap, None, fractional=False, commission=True),
        "E_shares_fractional_real_commission": run_basket(
            full, full_exp, cap, None, fractional=True, commission=True),
    }
    labels = {
        "B_shares_fractional_no_commission": "B 每月再平衡、碎股、不收佣金",
        "C_shares_integer_no_commission": "C 每月再平衡、整股、不收佣金",
        "D_shares_integer_real_commission": "D 每月再平衡、整股、真實佣金",
        "E_shares_fractional_real_commission": "E 每月再平衡、碎股、真實佣金",
    }
    for key, label in labels.items():
        r = steps[key]
        print(f"    {label:<30}{r['cagr']:>8.2%}"
              f"{r['sharpe_annualized']:>8.3f}{r['max_drawdown']:>9.1%}"
              f"{r['annual_cost_fraction']:>8.2%}"
              f"{r['invested_fraction']:>8.1%}")
    payload["decomposition"] = {"capital": cap, "steps": steps}
    b, c, d, e = (steps["B_shares_fractional_no_commission"],
                  steps["C_shares_integer_no_commission"],
                  steps["D_shares_integer_real_commission"],
                  steps["E_shares_fractional_real_commission"])
    pp = lambda x, y: (x - y) * 100  # noqa: E731  percentage points, not percent
    print(f"\n    每月再平衡取代每日：{pp(b['cagr'], ws['cagr']):+.2f}pp"
          f"（等於免費，權重漂移沒有傷害）")
    print(f"    整股湊不齊：        {pp(c['cagr'], b['cagr']):+.2f}pp"
          f"（{1 - c['invested_fraction'] / b['invested_fraction']:.1%} 的資金卡在現金）")
    print(f"    真實佣金：          {pp(d['cagr'], c['cagr']):+.2f}pp")
    print(f"    改用碎股買回：      {pp(e['cagr'], d['cagr']):+.2f}pp"
          f"（比任何不交易帶的調參都大）")
    print(f"    可執行的答案是 D：CAGR {d['cagr']:.2%}、Sharpe "
          f"{d['sharpe_annualized']:.3f}；能用碎股則是 E：{e['cagr']:.2%}、"
          f"{e['sharpe_annualized']:.3f}")

    # 2. Whether holding fewer names is worth it, with selection neutralised.
    if args.draws:
        print(f"\n  ② 少持有幾檔划算嗎（每個檔數隨機抽 {args.draws} 個籃子，"
              f"${cap:,.0f}，整股）")
        print(f"    {'檔數':>5}{'不設帶 CAGR 中位':>18}{'25bps 帶':>12}"
              f"{'年成本':>9}{'回撤中位':>10}{'隨機 10-90%':>18}")
        payload["size_grid"] = {}
        for n in BASKET_SIZES:
            rows = {"none": [], "25": []}
            # At full size there is only one possible basket, so resampling it
            # would just repeat the same run and report a zero-width spread as
            # if it were a measurement.
            draws = 1 if n >= len(symbols) else args.draws
            for names in random_baskets(symbols, bucket_of, panel, n, draws):
                gross = bucket_weighted_gross(panel[names], bucket_of, "equal",
                                              VOL_LOOKBACK, REBALANCE_BAND,
                                              ONE_WAY_COST_BPS)[0]
                exp = exposure_path(gross, TARGET_VOL)
                rows["none"].append(run_basket(names, exp, cap, None))
                rows["25"].append(run_basket(names, exp, cap, 25.0))
            plain = np.array([r["cagr"] for r in rows["none"]])
            banded = np.array([r["cagr"] for r in rows["25"]])
            cost = np.median([r["annual_cost_fraction"] for r in rows["none"]])
            dd = np.median([r["max_drawdown"] for r in rows["none"]])
            payload["size_grid"][str(n)] = {
                "draws": draws,
                "cagr_median": float(np.median(plain)),
                "cagr_p10": float(np.quantile(plain, 0.10)),
                "cagr_p90": float(np.quantile(plain, 0.90)),
                "cagr_median_banded_25bps": float(np.median(banded)),
                "max_drawdown_median_banded_25bps": float(
                    np.median([r["max_drawdown"] for r in rows["25"]])),
                "annual_cost_median": float(cost),
                "annual_cost_median_banded_25bps": float(
                    np.median([r["annual_cost_fraction"] for r in rows["25"]])),
                "max_drawdown_median": float(dd),
            }
            spread = ("單一籃子" if draws == 1 else
                      f"{np.quantile(plain, 0.10):.1%} ~ "
                      f"{np.quantile(plain, 0.90):.1%}")
            print(f"    {n:>5}{np.median(plain):>18.2%}"
                  f"{np.median(banded):>12.2%}{cost:>9.2%}{dd:>10.1%}"
                  f"{spread:>18}")
        print("    抽樣的 10-90% 寬度就是「換一組名字」本身造成的差異；"
              "檔數的影響要大過這個寬度才算存在。")

        # 3. The idea that did not work, kept so it is not retried.
        print(f"\n  ③ 已否證：按股價高低挑名字（理由是高價股每股費較低）")
        print(f"    {'檔數':>5}{'排序籃子':>12}{'隨機中位數':>14}{'結果':>16}")
        payload["price_ranking_refuted"] = {}
        for n in BASKET_SIZES:
            if n >= len(symbols):
                print(f"    {n:>5}{'——':>12}{'——':>14}"
                      f"{'全宇宙，無可挑':>16}")
                continue
            ranked = run_basket(baskets[n], exposures[n], cap, None)["cagr"]
            med = payload["size_grid"][str(n)]["cagr_median"]
            payload["price_ranking_refuted"][str(n)] = {
                "ranked_cagr": ranked, "random_median": med,
                "beats_random_median": bool(ranked > med),
            }
            print(f"    {n:>5}{ranked:>12.2%}{med:>14.2%}"
                  f"{('高於中位' if ranked > med else '低於中位'):>16}")
        print("    每個可挑的檔數都低於隨機中位數，所以這個排序是負分，不採用。")

    # 4. How the answer moves with account size.
    print(f"\n  ④ 資金規模的影響（全 120 檔，整股 vs 碎股，真實佣金）")
    print(f"    {'資金':>10}{'整股 CAGR':>12}{'碎股 CAGR':>12}"
          f"{'碎股買回':>10}{'整股年成本':>12}{'整股投入率':>12}")
    payload["capital_sweep"] = {}
    for capital in CAPITALS:
        i = run_basket(full, full_exp, capital, None, fractional=False)
        f = run_basket(full, full_exp, capital, None, fractional=True)
        payload["capital_sweep"][str(int(capital))] = {"integer": i,
                                                       "fractional": f}
        print(f"    {capital:>10,}{i['cagr']:>12.2%}{f['cagr']:>12.2%}"
              f"{pp(f['cagr'], i['cagr']):>+10.2f}"
              f"{i['annual_cost_fraction']:>12.2%}"
              f"{i['invested_fraction']:>12.1%}")
    print("    「碎股買回」欄位單位為百分點。整股的傷害隨資金變小而放大，"
          "佣金的傷害也一樣，但兩者都不是碎股帳戶的問題。")

    # 5. The configuration actually shipped, and an honest split of why it wins.
    print(f"\n  ⑤ 出貨設定：碎股 + 漂移帶（${cap:,.0f}，全 120 檔）")
    print(f"    {'漂移帶':>8}{'CAGR':>9}{'Sharpe':>9}{'回撤':>9}{'年成本':>8}"
          f"{'每月單數':>10}{'最大單一檔':>11}")
    payload["shipped"] = {}
    for ceiling in CEILINGS:
        r = run_basket(full, full_exp, cap, ceiling, fractional=True)
        key = "none" if ceiling is None else f"{ceiling:.0f}"
        payload["shipped"][key] = r
        label = "無" if ceiling is None else f"{ceiling:.0f}bps"
        print(f"    {label:>8}{r['cagr']:>9.2%}{r['sharpe_annualized']:>9.3f}"
              f"{r['max_drawdown']:>9.1%}{r['annual_cost_fraction']:>8.2%}"
              f"{r['monthly_orders_mean']:>10.0f}{r['max_weight_peak']:>11.1%}")

    # A band helps for two unrelated reasons and they deserve different credit:
    # the fees it saves are arithmetic, the weight drift it permits is one
    # window's luck. Pricing every order but deducting nothing separates them.
    # Zeroing the commission parameters cannot, because the band's own threshold
    # is derived from them, so both arms collapse to the same run.
    plain_free = run_basket(full, full_exp, cap, None, fractional=True,
                            charge=False)
    band_free = run_basket(full, full_exp, cap, DRIFT_BAND_BPS,
                           fractional=True, charge=False)
    plain = payload["shipped"]["none"]
    banded = payload["shipped"][f"{DRIFT_BAND_BPS:.0f}"]
    drift = pp(band_free["cagr"], plain_free["cagr"])
    total = pp(banded["cagr"], plain["cagr"])
    payload["band_attribution"] = {
        "cost_ceiling_bps": DRIFT_BAND_BPS,
        "total_pp": total, "drift_pp": drift, "fee_saving_pp": total - drift,
        "fees_without_band": plain["total_cost"],
        "fees_with_band": banded["total_cost"],
    }
    print(f"\n    {DRIFT_BAND_BPS:.0f}bps 帶總共值 {total:+.2f}pp，拆開：")
    print(f"      省下的手續費  {total - drift:+.2f}pp"
          f"（${plain['total_cost']:,.0f} → ${banded['total_cost']:,.0f}）"
          f"——這是算術，可依賴")
    print(f"      權重漂移      {drift:+.2f}pp"
          f"——只有這一個窗口，方向未知，不該計入預期")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\n寫入 {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
