"""
Signal-trap heuristics: score whether a day's price action around one of
OUR historical signals resembles known institutional manipulation /
liquidity-trap footprints. REPORT-ONLY (user decision, 2026-07-28): scores
are written to backtests/reports/signal_trap_report.md for human review and
never gate or filter a trade.

Honesty contract — what these scores ARE and ARE NOT:
  - They are PATTERN-CONSISTENCY proxies computed from observable data
    (daily OHLCV; captured tick/depth archives when they exist for the
    date). A high score means "this bar's shape is consistent with a bull
    trap / stop hunt / etc.", NOT "someone manipulated this stock" —
    intent is not observable from market data, and real spoofing cases are
    proven with order-audit trails we will never have.
  - Every sub-score returns None (UNKNOWN) when its required data is
    missing (no high/low columns, no tick archive for that date, no news/
    filing cache for that symbol). None is surfaced as "evidence
    unavailable" — it is never treated as 0 ("nothing suspicious").

Sub-scores (all in [0, 1] when computable):
  false_breakout_score   bull/bear trap: intraday break of the N-day range
                         that closes back inside it, worse on high volume.
  stop_hunt_score        long wick sweeping beyond the recent extreme with
                         a recovered close (classic stop-run bar).
  marking_the_close_score volume/price-drift concentration into the final
                         minutes (needs captured tick trades for the date).
  order_book_churn_score cancel-heavy, off-touch book activity vs actual
                         trades (spoofing/layering PROXY; needs captured
                         L2 depth for the date).
  short_distort_score    sharp high-volume drop accompanied by news flow
                         but NO 8-K filing (unverifiable-catalyst pattern
                         behind short & distort campaigns). Three-valued
                         evidence from finnhub_client / edgar_client.
  pinging_score          burst of odd-lot-or-smaller prints clustered in a
                         short window with little price movement (hidden/
                         iceberg-liquidity-probing PROXY; needs captured
                         tick trades for the date).
  dark_pool_internalization_score  day's off-exchange (TRF/ADF-reported)
                         print share above a rough market-wide baseline
                         (needs captured tick trades WITH the `exchange`
                         column for the date; see that function's docstring
                         — the off-exchange code set is unverified against a
                         real capture sample as of 2026-07-29).
  print_lag_score        day's share of prints carrying a CTA/UTP late/out-
                         of-sequence condition code (L/Z/U/T — "this print's
                         timestamp isn't when the trade actually happened"),
                         above a rough baseline (needs captured tick trades
                         WITH the `special_conditions` column; condition-code
                         set is unverified against a real capture sample as
                         of 2026-07-29, same tier as dark_pool_internalization_score).
  order_flow_imbalance_score  magnitude of the day's buy-vs-sell AGGRESSOR
                         volume imbalance, aggressor side classified with the
                         tick rule (no true side flag exists on captured
                         trades — see that function's docstring for why the
                         tick rule was chosen over an L2-depth-imbalance
                         proxy). Unlike every other sub-score above, this is
                         NOT a manipulation-pattern-vs-baseline proxy — it is
                         a directly bounded [0,1] flow-imbalance MAGNITUDE
                         (0 = balanced aggressor volume, 1 = one-sided), so a
                         high score on its own is not "suspicious", just
                         directionally lopsided flow; needs captured tick
                         trades (same 'time'/'price'/'size' columns as
                         pinging_score/marking_the_close_score).

combine_trap_score() aggregates the AVAILABLE sub-scores (simple mean) and
reports which were unavailable, so a report row always shows both the
number and its evidence basis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_RANGE_LOOKBACK_DAYS = 20
_VOLUME_LOOKBACK_DAYS = 20


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _day_row(ohlcv: pd.DataFrame, day: pd.Timestamp):
    """(prior_window, day_row) or (None, None) when the panel can't support
    the calculation. `ohlcv`: single-symbol frame indexed by date with
    open/high/low/close/volume."""
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(ohlcv.columns):
        return None, None
    if day not in ohlcv.index:
        return None, None
    pos = ohlcv.index.get_loc(day)
    if isinstance(pos, slice) or pos < _RANGE_LOOKBACK_DAYS:
        return None, None
    prior = ohlcv.iloc[pos - _RANGE_LOOKBACK_DAYS : pos]
    return prior, ohlcv.iloc[pos]


def _volume_ratio(prior: pd.DataFrame, row) -> float:
    mean_vol = float(prior["volume"].mean())
    if mean_vol <= 0:
        return 1.0
    return float(row["volume"]) / mean_vol


# ── daily-bar sub-scores ─────────────────────────────────────────────────────

def false_breakout_score(ohlcv: pd.DataFrame, day: pd.Timestamp) -> float | None:
    """Bull trap: high pierces the prior 20-day high but the close falls
    back below it (mirror for bear traps at the low). Score grows with the
    fraction of the intraday penetration given back by the close and with
    the volume spike (traps need participation to be worth setting)."""
    prior, row = _day_row(ohlcv, pd.Timestamp(day))
    if prior is None:
        return None
    prior_high = float(prior["high"].max())
    prior_low = float(prior["low"].min())
    day_range = max(float(row["high"]) - float(row["low"]), 1e-9)

    score = 0.0
    if float(row["high"]) > prior_high and float(row["close"]) < prior_high:
        penetration = (float(row["high"]) - prior_high) / day_range
        giveback = (float(row["high"]) - float(row["close"])) / day_range
        score = max(score, _clip01(0.5 * min(penetration * 4, 1.0) + 0.5 * giveback))
    if float(row["low"]) < prior_low and float(row["close"]) > prior_low:
        penetration = (prior_low - float(row["low"])) / day_range
        giveback = (float(row["close"]) - float(row["low"])) / day_range
        score = max(score, _clip01(0.5 * min(penetration * 4, 1.0) + 0.5 * giveback))
    if score == 0.0:
        return 0.0

    vol_boost = _clip01((_volume_ratio(prior, row) - 1.0) / 2.0)  # 3x volume -> +1
    return _clip01(0.7 * score + 0.3 * vol_boost)


def stop_hunt_score(ohlcv: pd.DataFrame, day: pd.Timestamp) -> float | None:
    """Stop-run bar: a long wick sweeps beyond the prior 20-day extreme
    (where resting stops cluster) and the close recovers back inside. The
    wick share of the bar's range is the core signal; the sweep must
    actually exceed the prior extreme to count."""
    prior, row = _day_row(ohlcv, pd.Timestamp(day))
    if prior is None:
        return None
    prior_low = float(prior["low"].min())
    prior_high = float(prior["high"].max())
    o, h, l, c = (float(row[k]) for k in ("open", "high", "low", "close"))
    day_range = max(h - l, 1e-9)

    score = 0.0
    if l < prior_low and c > prior_low:
        lower_wick = min(o, c) - l
        score = max(score, _clip01(lower_wick / day_range * 1.5))
    if h > prior_high and c < prior_high:
        upper_wick = h - max(o, c)
        score = max(score, _clip01(upper_wick / day_range * 1.5))
    if score == 0.0:
        return 0.0

    vol_boost = _clip01((_volume_ratio(prior, row) - 1.0) / 2.0)
    return _clip01(0.75 * score + 0.25 * vol_boost)


# ── microstructure sub-scores (need captured archives) ──────────────────────

def marking_the_close_score(
    trades: pd.DataFrame | None,
    session_close: pd.Timestamp,
    window_minutes: int = 10,
) -> float | None:
    """Volume + signed price drift concentrated into the last minutes of
    the session (the classic 'marking the close' footprint). `trades`: the
    day's captured tick prints (data/depth/<SYM>_trades_*.jsonl loaded into
    a frame with tz-aware 'time', 'price', 'size'). None when no archive
    exists for the date."""
    if trades is None or len(trades) < 50 or not {"time", "price", "size"}.issubset(trades.columns):
        return None
    trades = trades.sort_values("time")
    cutoff = pd.Timestamp(session_close) - pd.Timedelta(minutes=window_minutes)
    late = trades[trades["time"] >= cutoff]
    if late.empty:
        return 0.0

    volume_share = float(late["size"].sum()) / max(float(trades["size"].sum()), 1e-9)
    # Expected share if volume were uniform across a 390-minute session:
    expected_share = window_minutes / 390.0
    concentration = _clip01((volume_share - expected_share) / (5 * expected_share))

    day_vwap = float((trades["price"] * trades["size"]).sum() / max(trades["size"].sum(), 1e-9))
    ret_std = float(trades["price"].pct_change().std() or 0.0)
    if ret_std <= 0:
        drift = 0.0
    else:
        late_move = abs(float(late["price"].iloc[-1]) - day_vwap) / day_vwap
        drift = _clip01(late_move / (ret_std * np.sqrt(len(late)) * 3 + 1e-9))
    return _clip01(0.6 * concentration + 0.4 * drift)


def order_book_churn_score(depth_events: pd.DataFrame | None, n_trades: int) -> float | None:
    """Spoofing/layering PROXY from captured L2 events: books dominated by
    inserts/deletes AWAY from the touch, relative to how much actually
    trades. High churn + off-touch concentration + one-sidedness is the
    observable residue layering leaves; it is NOT proof (market makers
    legitimately re-quote constantly, hence 'proxy' everywhere).
    `depth_events` columns: operation (0=insert 1=update 2=delete), side,
    position, size (python/interfaces/ibkr_tick_capture.py's schema)."""
    if depth_events is None or len(depth_events) < 100 or \
            not {"operation", "side", "position", "size"}.issubset(depth_events.columns):
        return None

    inserts = depth_events[depth_events["operation"] == 0]
    deletes = depth_events[depth_events["operation"] == 2]
    churn_ratio = (len(inserts) + len(deletes)) / max(n_trades, 1)
    churn = _clip01((churn_ratio - 10.0) / 40.0)   # >=50 book changes per trade -> saturates

    off_touch = depth_events[depth_events["position"] >= 3]
    off_touch_share = len(off_touch) / len(depth_events)
    off_touch_component = _clip01((off_touch_share - 0.3) / 0.5)

    bid_size = float(depth_events.loc[depth_events["side"] == 1, "size"].sum())
    ask_size = float(depth_events.loc[depth_events["side"] == 0, "size"].sum())
    total = bid_size + ask_size
    imbalance = abs(bid_size - ask_size) / total if total > 0 else 0.0

    return _clip01(0.4 * churn + 0.3 * off_touch_component + 0.3 * imbalance)


def pinging_score(
    trades: pd.DataFrame | None,
    small_size_threshold: int = 100,
    burst_window_seconds: int = 60,
    burst_min_count: int = 8,
) -> float | None:
    """Pinging / hidden-liquidity-probing PROXY: a burst of odd-lot-or-smaller
    prints (<= small_size_threshold shares — the standard round-lot boundary)
    clustered within a short trailing window, with little price movement
    during that burst. This is the observable residue an algo leaves when it
    sends tiny orders to test for resting hidden/iceberg liquidity without
    intent to trade in size — genuine retail flow doesn't cluster this
    tightly with near-zero price drift. `trades`: same schema as
    marking_the_close_score (captured tick prints with 'time'/'price'/'size'
    columns, python/interfaces/ibkr_tick_capture.py's AllLast archive). None
    when no archive exists for the date — same forward-only constraint as
    every tick-based score in this module (data/ticks/ only has coverage from
    whenever the capture script started running; it is UNKNOWN for any
    earlier date, never a 0)."""
    if trades is None or len(trades) < 50 or not {"time", "price", "size"}.issubset(trades.columns):
        return None
    trades = trades.sort_values("time").reset_index(drop=True)
    small = trades[trades["size"] <= small_size_threshold].reset_index(drop=True)
    if len(small) < burst_min_count:
        return 0.0

    # Vectorized trailing-window burst count: for each small print at time
    # t_i, how many OTHER small prints (including itself) fall in
    # [t_i - window, t_i]. searchsorted on the already-sorted time index
    # avoids an O(n^2) scan.
    times = small["time"]
    window = pd.Timedelta(seconds=burst_window_seconds)
    start_idx = times.searchsorted(times - window, side="left")
    counts = np.arange(len(times)) - start_idx + 1
    max_burst = int(counts.max())
    if max_burst < burst_min_count:
        return 0.0

    burst_end = int(np.argmax(counts))
    burst_start = int(start_idx[burst_end])
    burst_trades = small.iloc[burst_start : burst_end + 1]

    day_range = max(float(trades["price"].max()) - float(trades["price"].min()), 1e-9)
    burst_range = float(burst_trades["price"].max()) - float(burst_trades["price"].min())
    # Burst range < 10% of the day's range -> price barely moved during the
    # probe -> stillness ~1.0; a burst that coincides with a real directional
    # move is more likely genuine flow than a probe.
    stillness = _clip01(1.0 - burst_range / (0.1 * day_range + 1e-9))

    intensity = _clip01((max_burst - burst_min_count) / burst_min_count)  # 2x threshold saturates
    return _clip01(0.6 * intensity + 0.4 * stillness)


# Exchange/venue codes IB's tick-by-tick AllLast `exchange` field uses for
# prints reported through FINRA's Trade Reporting Facility / ADF rather than
# a lit exchange — i.e. dark pool / ATS / wholesaler-internalization prints
# (the SIP tape can't tell you WHICH dark pool, only that it wasn't lit).
#
# ⚠️ UNVERIFIED (2026-07-29): data/ticks/ has no captured files yet (the
# capture script hasn't been run for long enough to produce a sample), so
# this set is a best-effort guess from FINRA/SIP tape conventions, NOT
# confirmed against a real ib_async TickByTickAllLast.exchange value. Once
# scripts/capture_market_microstructure.py has run for a few sessions,
# inspect data/ticks/<SYM>/<date>.jsonl's `exchange` column directly and
# correct this set / the DARK_POOL_BASELINE_SHARE constant below before
# trusting this score's absolute magnitude — treat it as experimental
# relative to the other (validated-format) tick-based scores in this module
# until then.
DARK_POOL_EXCHANGE_CODES = frozenset({"TRF", "ADF", "D", "FINRA"})
# Rough market-wide off-exchange (dark pool + ATS + wholesaler
# internalization) share of consolidated US equity volume in recent years
# (~40-45% per FINRA/Rule 605 commentary) — a single global constant, not
# calibrated per symbol/session. The score only rises once a day's share
# exceeds this "normal" baseline.
DARK_POOL_BASELINE_SHARE = 0.4


def dark_pool_internalization_score(
    trades: pd.DataFrame | None,
    off_exchange_codes: frozenset[str] = DARK_POOL_EXCHANGE_CODES,
    baseline_share: float = DARK_POOL_BASELINE_SHARE,
) -> float | None:
    """Dark pool / internalization PROXY: the day's share of captured prints
    reported via an off-exchange venue code, above the rough market-wide
    baseline. `trades`: same schema as marking_the_close_score, but requires
    an `exchange` column (python/interfaces/ibkr_tick_capture.py's AllLast
    archive already records this per print). None when no archive exists for
    the date, or when the archive predates the `exchange` field being
    recorded. See DARK_POOL_EXCHANGE_CODES's docstring: the code set here is
    UNVERIFIED against a real capture sample as of 2026-07-29 — calibrate it
    once one exists."""
    if trades is None or len(trades) < 50 or "exchange" not in trades.columns:
        return None
    total = len(trades)
    off_exchange = trades["exchange"].astype(str).str.upper().isin(
        {code.upper() for code in off_exchange_codes}
    )
    off_share = float(off_exchange.sum()) / total
    return _clip01((off_share - baseline_share) / (1.0 - baseline_share))


# CTA/UTP consolidated-tape trade condition codes that mean "this print's
# TIMESTAMP does not reflect when the trade actually happened" — i.e. a
# delayed/backfilled report, not a fresh one (sources: Nasdaq Trader Equity
# Trade Journal field spec, dxFeed TimeAndSale sale-condition reference,
# 2026-07-29):
#   L = Sold Last            trade prints in sequence but reported LATE
#   Z = Sold (Out of Sequence)   reported out of sequence, different time than the actual trade
#   U = Extended Hours Sold (Out of Sequence)   >90s late extended-hours report
#   T = Extended Hours Trade / Late Trade Before-After Hours
#
# ⚠️ UNVERIFIED (2026-07-29) against a real ib_async TickByTickAllLast.
# specialConditions VALUE — same caveat as DARK_POOL_EXCHANGE_CODES above:
# no data/ticks/ capture sample exists yet to confirm IB actually surfaces
# these letters (vs. numeric codes, vs. not passing them through at all).
# Calibrate this set once a real capture exists; until then this score's
# absolute magnitude is experimental, same tier as dark_pool_internalization_score.
LATE_PRINT_CONDITION_CODES = frozenset({"L", "Z", "U", "T"})
# A handful of late/out-of-sequence prints is routine tape noise on any
# session — the score only rises once a day's share of them exceeds this.
# Unlike DARK_POOL_BASELINE_SHARE (a published market-wide statistic), this
# is a bare guess with no cited source — treat it as a placeholder to
# recalibrate once real data/ticks/ sessions exist to compute an actual
# per-symbol baseline from.
LATE_PRINT_BASELINE_SHARE = 0.05


def print_lag_score(
    trades: pd.DataFrame | None,
    late_condition_codes: frozenset[str] = LATE_PRINT_CONDITION_CODES,
    baseline_share: float = LATE_PRINT_BASELINE_SHARE,
) -> float | None:
    """Delayed/backfilled-print PROXY: the day's share of captured trades
    carrying a late/out-of-sequence CTA/UTP condition code, above a rough
    baseline. A high share means "an unusual number of this day's prints
    did NOT arrive when they claim the trade happened" — consistent with
    either heavy off-tape reporting catching up in bursts, or (less
    innocently) a participant timing when a trade gets reported to obscure
    the sequence of events around a move. `trades`: same schema as
    marking_the_close_score, but requires a `special_conditions` column
    (python/interfaces/ibkr_tick_capture.py's AllLast archive already
    records this per print, unparsed). Handles both a plain single-code
    string and a whitespace-separated multi-code string per print. None
    when no archive exists for the date, or the archive predates the
    `special_conditions` field being recorded. See
    LATE_PRINT_CONDITION_CODES's docstring: UNVERIFIED against a real
    capture sample as of 2026-07-29 — calibrate once one exists."""
    if trades is None or len(trades) < 50 or "special_conditions" not in trades.columns:
        return None
    total = len(trades)
    codes_upper = {c.upper() for c in late_condition_codes}

    def _is_late(raw: object) -> bool:
        tokens = str(raw).upper().replace(",", " ").split()
        return any(tok in codes_upper for tok in tokens)

    late_share = float(trades["special_conditions"].apply(_is_late).sum()) / total
    return _clip01((late_share - baseline_share) / (1.0 - baseline_share))


def order_flow_imbalance_score(
    trades: pd.DataFrame | None,
    min_trades: int = 50,
) -> float | None:
    """Buy-vs-sell AGGRESSOR volume imbalance for the session, |0,1|-bounded
    by construction (it is a share of total classified volume, not a
    baseline-relative proxy like most other scores in this module).

    Method — the TICK RULE (Lee, 1991), NOT Lee-Ready:
    `trades` (python/interfaces/ibkr_tick_capture.py's AllLast archive) has
    no explicit buy/sell-initiator flag — IB's TickByTickAllLast only carries
    price/size/exchange/specialConditions, never a side. The textbook
    Lee-Ready algorithm (Lee & Ready, 1991) classifies each trade against the
    PREVAILING quote midpoint (trade above mid -> buy-initiated, below mid ->
    sell-initiated, at mid -> fall back to the tick rule) — but that needs a
    trade-by-trade quote snapshot time-aligned to each print, which we do not
    have (BidAsk tick-by-tick is captured to a SEPARATE, unsynchronized
    stream/file; joining them by nearest-timestamp would silently invent a
    matching precision claim we cannot back up). Rather than fabricate that
    join, this function falls back to the plain TICK RULE alone: a trade is
    buy-initiated if it prints ABOVE the immediately preceding trade's price,
    sell-initiated if BELOW, and inherits the last classified direction if
    price is UNCHANGED (the standard "zero-tick" extension); the very first
    trade in the sample has no preceding price and is left unclassified. This
    is the same simplification `order_book_churn_score` avoids needing (it
    reads depth events directly) and that a genuine L2-depth bid/ask-size-at-
    the-touch imbalance proxy would also avoid — that alternative was
    considered and REJECTED for now because reconstructing "size resting at
    the touch at time T" from raw insert/update/delete depth events
    (operation/side/position/size) requires an actual order-book replay
    engine (docs/microstructure_pivot_plan.md §4b's `depth_replay.py`, Phase
    3, not yet built); duplicating a mini book-reconstruction here ad hoc,
    untested against `depth_replay.py`'s eventual real one, would be a worse
    honesty trade-off than the tick rule's well-documented (if imperfect)
    literature track record.

    What this score IS NOT proof of: academic comparisons against exchange-
    supplied trade-side flags put the plain tick rule's classification
    accuracy around 85-90% on NYSE-era data (Lee & Radhakrishna, 2000) — it
    is a noisy proxy for the true aggressor, not the true aggressor. It is
    ALSO ⚠️ UNVERIFIED (2026-07-29) for IB's specific tick-by-tick feed: no
    data/ticks/ capture sample exists yet to confirm IB's AllLast print
    sequencing/timestamps are fine-grained and gap-free enough for the tick
    rule to behave as the literature assumes (a feed with coalesced or
    reordered prints would bias the zero-tick carry-forward into longer runs
    than really happened) — treat the absolute magnitude as experimental,
    same tier as dark_pool_internalization_score/print_lag_score, until a
    real sample can be spot-checked.

    None when `trades` is missing/too short/lacks the required columns
    (same 'time'/'price'/'size' contract as marking_the_close_score/
    pinging_score). Returns 0.0 (not None) when trades exist but no trade
    could be classified at all (e.g. every print at an identical price with
    no seed direction) — that is a real, if degenerate, "no aggressor
    imbalance observed" outcome, not missing evidence."""
    if trades is None or len(trades) < min_trades or not {"time", "price", "size"}.issubset(trades.columns):
        return None
    trades = trades.sort_values("time").reset_index(drop=True)
    prices = trades["price"].to_numpy(dtype=float)
    sizes = trades["size"].to_numpy(dtype=float)

    buy_volume = 0.0
    sell_volume = 0.0
    last_direction = 0  # +1 buy, -1 sell, 0 = not yet established (zero-tick with no seed)
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            last_direction = 1
        elif diff < 0:
            last_direction = -1
        # diff == 0 -> inherit last_direction unchanged (zero-tick rule)
        if last_direction == 1:
            buy_volume += sizes[i]
        elif last_direction == -1:
            sell_volume += sizes[i]
        # last_direction == 0 (no direction ever established yet): unclassified, excluded

    total_classified = buy_volume + sell_volume
    if total_classified <= 0:
        return 0.0
    return _clip01(abs(buy_volume - sell_volume) / total_classified)


# ── event-evidence sub-score ─────────────────────────────────────────────────

def short_distort_score(
    ohlcv: pd.DataFrame,
    day: pd.Timestamp,
    has_news: bool | None,
    has_8k: bool | None,
) -> float | None:
    """'Short & distort' footprint: an outsized high-volume down move that
    arrives WITH news flow but WITHOUT any 8-K filing — i.e. a scary story
    no one was legally required to disclose. Three-valued evidence:
      has_news / has_8k None -> we lack coverage -> return None (UNKNOWN),
      because 'no filing found' is only meaningful when we actually looked.
    An 8-K near the move (has_8k True) means a REAL disclosed catalyst —
    score collapses toward 0."""
    prior, row = _day_row(ohlcv, pd.Timestamp(day))
    if prior is None:
        return None
    prev_close = float(prior["close"].iloc[-1])
    if prev_close <= 0:
        return None
    day_return = float(row["close"]) / prev_close - 1.0
    ret_std = float(prior["close"].pct_change().dropna().std() or 0.0)
    if ret_std <= 0:
        return None

    move_severity = _clip01((-day_return - 2 * ret_std) / (4 * ret_std))
    if move_severity == 0.0:
        return 0.0
    vol_boost = _clip01((_volume_ratio(prior, row) - 1.0) / 2.0)
    base = _clip01(0.6 * move_severity + 0.4 * vol_boost)

    if has_8k is None or has_news is None:
        return None
    if has_8k:
        return _clip01(base * 0.1)   # disclosed catalyst — not the pattern
    if has_news:
        return base                  # news-driven crash with no filing
    return _clip01(base * 0.4)       # crash with neither news nor filing coverage


# ── aggregation ──────────────────────────────────────────────────────────────

@dataclass
class TrapAssessment:
    symbol: str
    day: str
    components: dict                       # {name: float | None}
    trap_score: float | None               # mean of available components
    unavailable: list[str] = field(default_factory=list)
    event_flags: dict = field(default_factory=dict)  # earnings/8-K/econ markers

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "day": self.day,
            "trap_score": self.trap_score,
            "components": self.components,
            "unavailable": self.unavailable,
            "event_flags": self.event_flags,
        }


def combine_trap_score(components: dict[str, float | None]) -> tuple[float | None, list[str]]:
    """(mean of available sub-scores, [names of unavailable ones]). All-None
    -> (None, all names): no evidence is NOT a zero score."""
    available = {k: v for k, v in components.items() if v is not None}
    unavailable = sorted(k for k, v in components.items() if v is None)
    if not available:
        return None, unavailable
    return float(np.mean(list(available.values()))), unavailable


def assess_signal_day(
    symbol: str,
    day: pd.Timestamp,
    ohlcv: pd.DataFrame,
    has_news: bool | None = None,
    has_8k: bool | None = None,
    trades: pd.DataFrame | None = None,
    depth_events: pd.DataFrame | None = None,
    session_close: pd.Timestamp | None = None,
    event_flags: dict | None = None,
) -> TrapAssessment:
    """One report row: every sub-score for (symbol, day) from whatever
    evidence the caller could supply (None inputs simply mark those
    components unavailable)."""
    day = pd.Timestamp(day)
    n_trades = len(trades) if trades is not None else 0
    components = {
        "false_breakout": false_breakout_score(ohlcv, day),
        "stop_hunt": stop_hunt_score(ohlcv, day),
        "marking_the_close": (
            marking_the_close_score(trades, session_close) if session_close is not None else None
        ),
        "order_book_churn": order_book_churn_score(depth_events, n_trades),
        "short_distort": short_distort_score(ohlcv, day, has_news, has_8k),
        "pinging": pinging_score(trades),
        "dark_pool_internalization": dark_pool_internalization_score(trades),
        "print_lag": print_lag_score(trades),
        "order_flow_imbalance": order_flow_imbalance_score(trades),
    }
    score, unavailable = combine_trap_score(components)
    return TrapAssessment(
        symbol=symbol,
        day=str(day.date()),
        components=components,
        trap_score=score,
        unavailable=unavailable,
        event_flags=event_flags or {},
    )
