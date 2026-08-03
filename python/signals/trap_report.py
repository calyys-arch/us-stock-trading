"""
Builds backtests/reports/signal_trap_report.md — the report-only diagnostic
layer's output. Attaches trap_detector sub-scores and event markers
(earnings / 8-K / economic-calendar proximity) to the backtest's historical
signals so a human can review which of OUR signals fired into suspicious
price action.

STRICTLY OFFLINE at report time: every evidence source here reads LOCAL
CACHES only (data/news/, data/filings/, data/calendar/, data/ticks/,
data/depth/, data/finra_ats/) — populated beforehand by
scripts/refresh_event_data.py, scripts/capture_market_microstructure.py, and
scripts/backfill_finra_ats.py. A backtest run must be reproducible and
network-free once its data is cached, so missing caches show up as
"evidence unavailable" in the report, never as a fetch.

Nothing in this module feeds back into strategies, gates, or execution.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..core.calendar import session_open_close
from ..data.finra_ats import elevated_vs_baseline as _finra_elevated_vs_baseline
from .trap_detector import TrapAssessment, assess_signal_day

log = logging.getLogger(__name__)

REPORT_PATH = Path("backtests/reports/signal_trap_report.md")
TICKS_DIR = Path("data/ticks")     # trades + bidask (python/interfaces/ibkr_tick_capture.py)
DEPTH_DIR = Path("data/depth")     # L2 depth snapshots
NEWS_DIR = Path("data/news")
FILINGS_DIR = Path("data/filings")
CALENDAR_DIR = Path("data/calendar")

# Which root each event `kind` lives under — mirrors
# ibkr_tick_capture.TickCaptureWriter's rotation rule.
_KIND_TO_ROOT = {"trades": TICKS_DIR, "bidask": TICKS_DIR, "depth": DEPTH_DIR}

# Only surface rows worth a human's attention; everything else is counted
# in the summary but not listed.
MIN_REPORTED_SCORE = 0.3
MAX_REPORT_ROWS = 200
XSECTION_TOP_WEIGHTS_PER_DAY = 3


# ── cache-only evidence loaders ──────────────────────────────────────────────

def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def has_news_cached(symbol: str, day: pd.Timestamp) -> bool | None:
    """True/False from the Finnhub news month-cache; None when that month
    was never cached (free tier only covers ~12 months back)."""
    cached = _load_json(NEWS_DIR / symbol.upper() / f"{day:%Y-%m}.json")
    if cached is None:
        return None
    day_start = pd.Timestamp(day.date()).timestamp()
    return any(day_start <= float(row.get("datetime", 0)) < day_start + 86400 for row in cached)


def eight_k_dates_cached(symbol: str) -> set[str] | None:
    cached = _load_json(FILINGS_DIR / "8k" / f"{symbol.upper()}.json")
    if cached is None:
        return None
    return {f["filing_date"] for f in cached.get("filings", []) if f.get("filing_date")}


def earnings_dates_cached(years: range) -> dict[str, set[str]]:
    """{symbol: {iso_date}} from the year-level earnings-calendar caches
    that exist locally (missing years just contribute nothing)."""
    out: dict[str, set[str]] = {}
    for year in years:
        rows = _load_json(CALENDAR_DIR / f"earnings_{year}.json") or []
        for row in rows:
            symbol = str(row.get("symbol", "")).upper().strip()
            if symbol and row.get("date"):
                out.setdefault(symbol, set()).add(row["date"])
    return out


def econ_dates_cached() -> set[str]:
    dates: set[str] = set()
    if CALENDAR_DIR.exists():
        for path in CALENDAR_DIR.glob("economic_*.json"):
            for row in _load_json(path) or []:
                if row.get("time"):
                    dates.add(str(row["time"])[:10])
    return dates


def load_day_ticks(symbol: str, day: pd.Timestamp, kind: str) -> pd.DataFrame | None:
    """Captured microstructure archive for (symbol, day), or None when the
    capture script wasn't running that day. `kind` in {"trades", "bidask",
    "depth"} — trades/bidask live under data/ticks/, depth under
    data/depth/ (python/interfaces/ibkr_tick_capture.py's file layout)."""
    root = _KIND_TO_ROOT[kind]
    path = root / symbol.upper() / f"{day:%Y%m%d}.jsonl"
    if not path.exists():
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    return df


# ── assessment assembly ──────────────────────────────────────────────────────

def _near(dates: set[str] | None, day: pd.Timestamp, window_days: int = 1) -> bool | None:
    if dates is None:
        return None
    for offset in range(-window_days, window_days + 1):
        if str((day + pd.Timedelta(days=offset)).date()) in dates:
            return True
    return False


def assess_one(
    symbol: str,
    day: pd.Timestamp,
    ohlcv: pd.DataFrame,
    earnings_by_symbol: dict[str, set[str]],
    econ_dates: set[str],
    signal_context: str,
) -> TrapAssessment:
    day = pd.Timestamp(day)
    eight_k = eight_k_dates_cached(symbol)
    has_8k = _near(eight_k, day, window_days=1)
    has_news = has_news_cached(symbol, day)

    session = session_open_close(day.tz_localize("America/New_York") if day.tzinfo is None else day)
    trades = load_day_ticks(symbol, day, "trades")
    depth = load_day_ticks(symbol, day, "depth")

    earnings_dates = earnings_by_symbol.get(symbol.upper())
    week_start = (day - pd.Timedelta(days=day.weekday())).strftime("%Y-%m-%d")
    event_flags = {
        "signal": signal_context,
        "earnings_within_1d": _near(earnings_dates, day, 1) if earnings_dates is not None else None,
        "eight_k_within_1d": has_8k,
        "econ_event_same_day": str(day.date()) in econ_dates if econ_dates else None,
        # Tier 2 dark-pool proxy (python/data/finra_ats.py): coarse,
        # week-level, symbol-relative — None when that calendar week isn't
        # in data/finra_ats/<SYM>.jsonl yet (scripts/backfill_finra_ats.py).
        "dark_pool_participation_elevated": _finra_elevated_vs_baseline(symbol, week_start),
    }
    return assess_signal_day(
        symbol=symbol, day=day, ohlcv=ohlcv,
        has_news=has_news, has_8k=has_8k,
        trades=trades, depth_events=depth,
        session_close=session[1] if session else None,
        event_flags=event_flags,
    )


def collect_pairs_assessments(trades: list, panel_by_symbol: dict[str, pd.DataFrame]) -> list[TrapAssessment]:
    """One assessment per (leg symbol, entry date) of every pairs trade —
    the entry is the moment a trap would have caught us."""
    out = []
    for trade in trades:
        for symbol in (trade.code_a, trade.code_b):
            ohlcv = panel_by_symbol.get(symbol)
            if ohlcv is None:
                continue
            out.append((symbol, pd.Timestamp(trade.entry_date),
                        f"pairs entry ({trade.side}, exit {pd.Timestamp(trade.exit_date).date()})"))
    return out


def collect_xsection_assessments(targets_by_day: dict) -> list[tuple[str, pd.Timestamp, str]]:
    """Top-|weight| names per day — the positions most exposed to a trap."""
    out = []
    for day, target in targets_by_day.items():
        weights = getattr(target, "weights", None) or {}
        top = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)
        for code, weight in top[:XSECTION_TOP_WEIGHTS_PER_DAY]:
            out.append((code, pd.Timestamp(day), f"xsection weight {weight:+.3f}"))
    return out


def build_trap_report(
    signal_specs: list[tuple[str, pd.Timestamp, str]],
    panel_by_symbol: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_label: str,
    out_path: str | Path = REPORT_PATH,
) -> Path:
    """Assess every (symbol, day, context) spec and write the markdown
    report. Rows below MIN_REPORTED_SCORE with no event flags are only
    counted, not listed."""
    earnings = earnings_dates_cached(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))
    econ = econ_dates_cached()

    assessments: list[TrapAssessment] = []
    for symbol, day, context in signal_specs:
        ohlcv = panel_by_symbol.get(symbol)
        if ohlcv is None:
            continue
        try:
            assessments.append(assess_one(symbol, day, ohlcv, earnings, econ, context))
        except Exception:
            log.exception("trap_report: assessment failed for %s %s", symbol, day)

    def _flagged(a: TrapAssessment) -> bool:
        if a.trap_score is not None and a.trap_score >= MIN_REPORTED_SCORE:
            return True
        return any(v is True for k, v in a.event_flags.items() if k != "signal")

    flagged = [a for a in assessments if _flagged(a)]
    flagged.sort(key=lambda a: (a.trap_score is not None, a.trap_score or 0.0), reverse=True)
    truncated = len(flagged) > MAX_REPORT_ROWS
    shown = flagged[:MAX_REPORT_ROWS]

    n_with_score = sum(1 for a in assessments if a.trap_score is not None)
    unavailable_counts: dict[str, int] = {}
    for a in assessments:
        for name in a.unavailable:
            unavailable_counts[name] = unavailable_counts.get(name, 0) + 1

    lines = [
        "# Signal Trap Report (report-only diagnostics)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Data: {data_label}",
        f"Window: {pd.Timestamp(start).date()} .. {pd.Timestamp(end).date()}",
        "",
        "> **What this is**: heuristic PATTERN-CONSISTENCY scores (0-1) attached to this",
        "> backtest's historical signals — bars shaped like bull/bear traps, stop runs,",
        "> close-marking, cancel-heavy books, or news-without-filing crashes. A high score",
        "> means 'consistent with the footprint', NOT proof of manipulation. Nothing here",
        "> gates or filters any trade; this file exists for human review only.",
        "> 'evidence unavailable' means the required cache (news/filings/tick archive) does",
        "> not cover that symbol/date — it is never counted as 'nothing found'.",
        "",
        "## Summary",
        "",
        f"- Signals assessed: {len(assessments)} ({n_with_score} with at least one computable sub-score)",
        f"- Flagged for review (score >= {MIN_REPORTED_SCORE} or event within window): {len(flagged)}"
        + (f" — showing top {MAX_REPORT_ROWS}" if truncated else ""),
        "- Evidence unavailable counts: "
        + (", ".join(f"{k}={v}" for k, v in sorted(unavailable_counts.items())) or "none"),
        "",
        "## Flagged signals",
        "",
    ]
    if not shown:
        lines.append("(none)")
    for a in shown:
        score_str = f"{a.trap_score:.2f}" if a.trap_score is not None else "n/a"
        lines.append(f"### {a.symbol} {a.day} — trap score {score_str}")
        lines.append("")
        lines.append(f"- Signal: {a.event_flags.get('signal', '')}")
        comp_strs = []
        for name, value in a.components.items():
            comp_strs.append(f"{name}={value:.2f}" if value is not None else f"{name}=unavailable")
        lines.append(f"- Components: {', '.join(comp_strs)}")
        flags = {k: v for k, v in a.event_flags.items() if k != "signal"}
        flag_strs = [f"{k}={'yes' if v else ('no' if v is False else 'unknown')}" for k, v in flags.items()]
        lines.append(f"- Events: {', '.join(flag_strs)}")
        lines.append("")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("trap_report: %d/%d signals flagged -> %s", len(flagged), len(assessments), out_path)
    return out_path
