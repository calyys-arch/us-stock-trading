"""
Load captured trade prints from data/ticks/<SYMBOL>/<YYYYMMDD>.jsonl.

Same archive as python/signals/trap_report.py. This module keeps TRADE
rows only (price + size). IBKR writes trades and BidAsk into the same
file; BidAsk rows (bid_price/ask_price, no last price) are dropped.
Futu rows already are trades and carry ticker_direction.

Times are converted to tz-naive US/Eastern to match the 1-minute bar
index convention in python/microstructure/context.py. Naive ISO stamps
from Futu (session prints at 09:30) are treated as already-ET.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger(__name__)

TICKS_DIR = Path("data/ticks")
_ET = ZoneInfo("America/New_York")
_TRADE_COLUMNS = ["time", "price", "size", "ticker_direction", "source"]


def tick_path(symbol: str, day, ticks_dir: Path = TICKS_DIR) -> Path:
    key = _day_key(day)
    return Path(ticks_dir) / symbol.upper() / f"{key}.jsonl"


def load_trade_ticks(symbol: str, day, ticks_dir: Path = TICKS_DIR) -> pd.DataFrame | None:
    """Trade prints for (symbol, session date), or None when the capture
    script was not running that day. Never invents a side or a print."""
    path = tick_path(symbol, day, ticks_dir=ticks_dir)
    if not path.exists():
        return None
    rows: list[dict] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not _is_trade_row(rec):
                    continue
                rows.append({
                    "time": rec.get("time"),
                    "price": float(rec["price"]),
                    "size": float(rec["size"]),
                    "ticker_direction": str(rec.get("ticker_direction") or ""),
                    "source": str(rec.get("source") or ""),
                })
    except OSError:
        log.warning("tick_cache: cannot read %s", path)
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["time"] = df["time"].map(_to_et_naive)
    df = df.dropna(subset=["time", "price", "size"])
    if df.empty:
        return None
    return df.sort_values("time").reset_index(drop=True)


def _is_trade_row(rec: dict) -> bool:
    if rec.get("bid_price") is not None or rec.get("ask_price") is not None:
        return False
    try:
        price = float(rec.get("price"))
        size = float(rec.get("size"))
    except (TypeError, ValueError):
        return False
    return price > 0 and size > 0


def _day_key(day) -> str:
    if isinstance(day, str):
        return day.replace("-", "")[:8]
    ts = pd.Timestamp(day)
    return f"{ts.year:04d}{ts.month:02d}{ts.day:02d}"


def ticks_up_to(ticks: pd.DataFrame | None, now) -> pd.DataFrame | None:
    """Causal prefix: prints with time <= now. None if empty / missing.

    Tick analogue of `bars.iloc[:i+1]`. Prefer this over importing the
    backtest helper from a signal module."""
    if ticks is None or ticks.empty:
        return None
    now_ts = pd.Timestamp(now)
    times = ticks["time"] if "time" in ticks.columns else ticks.index
    tmax = pd.Timestamp(times.max())
    tmin = pd.Timestamp(times.min())
    if tmin <= now_ts and tmax <= now_ts:
        return ticks
    visible = ticks.loc[pd.DatetimeIndex(times) <= now_ts]
    if visible.empty:
        return None
    return visible


def _to_et_naive(value) -> pd.Timestamp | None:
    if value is None or value == "":
        return pd.NaT
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return pd.NaT
    if ts.tzinfo is not None:
        return ts.tz_convert(_ET).tz_localize(None)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts
