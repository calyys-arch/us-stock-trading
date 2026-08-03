"""
Distill this repo's microstructure diagnostic artifacts into semantic
(subject, predicate, object) events and push them into a UHAI GreyCat
project's `microstructure` pack (uhai/microstructure/*.gcl — copy that
folder into <uhai-project>/src/microstructure/ per uhai/README.md).

"Distill first, then graph" — this script NEVER pushes raw 1-minute bars,
ticks, or per-bar signal detections (those stay in data/history_1m/*.parquet,
data/ticks/, data/depth/, which are already fast and cheap to query
locally). It only pushes the SUMMARIZED findings that are actually worth a
relationship-aware store: WFO/gate verdicts, tested parameter sets,
promotion decisions, and universe snapshots. Parquet stays the source of
truth for volume; GreyCat's triple store is for meaning.

Sources (all optional — a fresh checkout with none of these yet is a
silent no-op, not an error):
  - backtests/reports/intraday_backtest_report.json (scripts/run_intraday_backtest.py)
      -> subject "signal:<name>", predicates "hasBacktestResult" (decision,
         window, wfo/MC/stress summary, gates) and "hasParameterSet"
         (candidate_params actually tested).
  - backtests/logs/promotion_history.jsonl (python/backtest/promotion.py)
      -> subject "strategy:<name>", predicate "hadPromotionDecision"
         (PROMOTED/REJECTED + gates + before/after Sharpe).
  - configs/universe.yaml               (scripts/refresh_universe.py /
      manual picks)
      -> subject "universe:fixed_universe", predicate "asOf" (symbol list +
         ranking metric + computed_at).

Incremental: a small watermark file (data/uhai_sync/_state.json) tracks how
far each source has already been synced, so re-running this script after a
fresh backtest/promotion only pushes what's new — not the whole history
every time.

Delivery: if UHAI_GREYCAT_URL is set (process env or a local .env, same
convention as python/data/finnhub_client.py), events are POSTed to that
GreyCat server's exposed `microstructure::ingestMicroEvents` endpoint
(GreyCat's `@expose` HTTP convention: POST /<module>::<fn>, JSON body
keyed by parameter name). If it is NOT set (default — nothing to
configure for a plain local run), the script runs in OFFLINE/DRY-RUN mode:
it still computes the exact same event batch and writes it to
data/uhai_sync/<run_ts>.jsonl for manual import later, and exits 0. This
mirrors this repo's "always leave a paper trail, network optional"
convention (see python/data/price_cache.py, intraday_cache.py).

Usage:
    python scripts/sync_uhai.py             # sync new events since last run
    python scripts/sync_uhai.py --dry-run    # force offline mode even if UHAI_GREYCAT_URL is set
    python scripts/sync_uhai.py --reset      # ignore watermark, resync everything from scratch
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

REPORT_JSON_PATH = Path("backtests/reports/intraday_backtest_report.json")
PROMOTION_HISTORY_PATH = Path("backtests/logs/promotion_history.jsonl")
UNIVERSE_CONFIG_PATH = Path("configs/universe.yaml")
SYNC_DIR = Path("data/uhai_sync")
STATE_PATH = SYNC_DIR / "_state.json"

_INGEST_ENDPOINT = "microstructure::ingestMicroEvents"


def _load_env() -> None:
    """Same convention as finnhub_client.py / edgar_client.py: pick up a
    local .env next to the repo root without requiring it in the real
    process environment."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _iso_to_epoch_us(iso_str: str) -> int:
    import pandas as pd

    ts = pd.Timestamp(iso_str)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp() * 1_000_000)


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        log.warning("sync_uhai: corrupt state file %s, starting fresh", STATE_PATH)
        return {}


def _save_state(state: dict) -> None:
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _events_from_backtest_report(state: dict) -> list[dict]:
    if not REPORT_JSON_PATH.exists():
        return []
    payload = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))
    generated_at = payload.get("generated_at")
    if generated_at is not None and generated_at == state.get("last_report_generated_at"):
        return []  # this exact run was already synced

    events = []
    now_iso = generated_at or datetime.now(timezone.utc).isoformat()
    for result in payload.get("results", []):
        signal = result.get("signal", "unknown")
        subject = f"signal:{signal}"
        events.append({
            "subject": subject,
            "predicate": "hasBacktestResult",
            "object_json": json.dumps({
                "decision": result.get("decision"),
                "window": result.get("window"),
                "data_label": result.get("data_label"),
                "wfo_pass_ratio": result.get("wfo_pass_ratio"),
                "oos_sharpe_mean": result.get("oos_sharpe_mean"),
                "gates": result.get("gates"),
            }, default=str),
            "source": "run_intraday_backtest.py",
            "ts": now_iso,
        })
        if result.get("candidate_params"):
            events.append({
                "subject": subject,
                "predicate": "hasParameterSet",
                "object_json": json.dumps(result["candidate_params"], default=str),
                "source": "run_intraday_backtest.py",
                "ts": now_iso,
            })

    state["last_report_generated_at"] = generated_at
    return events


def _events_from_promotion_history(state: dict) -> list[dict]:
    if not PROMOTION_HISTORY_PATH.exists():
        return []
    lines = PROMOTION_HISTORY_PATH.read_text(encoding="utf-8").splitlines()
    already_synced = state.get("promotion_history_lines_synced", 0)
    new_lines = lines[already_synced:]

    events = []
    for line in new_lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            log.warning("sync_uhai: skipping malformed promotion_history.jsonl line")
            continue
        events.append({
            "subject": f"strategy:{record.get('strategy', 'unknown')}",
            "predicate": "hadPromotionDecision",
            "object_json": json.dumps({
                "decision": record.get("decision"),
                "reason": record.get("reason"),
                "candidate_oos_sharpe": record.get("candidate_oos_sharpe"),
                "baseline_oos_sharpe": record.get("baseline_oos_sharpe"),
                "gates": record.get("gates"),
                "config_written": record.get("config_written"),
            }, default=str),
            "source": "promotion.py",
            "ts": record.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        })

    state["promotion_history_lines_synced"] = len(lines)
    return events


def _events_from_universe_config(state: dict) -> list[dict]:
    if not UNIVERSE_CONFIG_PATH.exists():
        return []
    import yaml

    doc = yaml.safe_load(UNIVERSE_CONFIG_PATH.read_text(encoding="utf-8"))
    fixed = doc.get("fixed_universe", {})
    computed_at = fixed.get("computed_at")
    if computed_at is not None and computed_at == state.get("last_universe_computed_at"):
        return []

    state["last_universe_computed_at"] = computed_at
    ts = f"{computed_at}T00:00:00Z" if computed_at else datetime.now(timezone.utc).isoformat()
    return [{
        "subject": "universe:fixed_universe",
        "predicate": "asOf",
        "object_json": json.dumps({
            "symbols": fixed.get("symbols"),
            "top_n": fixed.get("top_n"),
            "ranking_metric": fixed.get("ranking_metric"),
            "source_pool": fixed.get("source_pool"),
        }, default=str),
        "source": "configs/universe.yaml",
        "ts": ts,
    }]


def collect_events(reset: bool) -> tuple[list[dict], dict]:
    state = {} if reset else _load_state()
    events = []
    events.extend(_events_from_backtest_report(state))
    events.extend(_events_from_promotion_history(state))
    events.extend(_events_from_universe_config(state))
    return events, state


def _write_offline_export(events: list[dict]) -> Path:
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = SYNC_DIR / f"{run_ts}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return out_path


def _push_to_greycat(events: list[dict], base_url: str, token: str | None) -> bool:
    """POST to <base_url>/microstructure::ingestMicroEvents. Returns True on
    success, False on ANY failure (network, auth, schema mismatch) — this
    function is fail-safe by design, matching the rest of this repo's
    external-integration modules (finnhub_client.py, edgar_client.py):
    a broken UHAI connection must never break a local backtest/sync run."""
    import httpx

    payload = {
        "events": [
            {**ev, "ts": _iso_to_epoch_us(ev["ts"])}
            for ev in events
        ],
    }
    headers = {"Authorization": token} if token else {}
    url = base_url.rstrip("/") + "/" + _INGEST_ENDPOINT
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        resp.raise_for_status()
        log.info("sync_uhai: pushed %d event(s) to %s -> %s", len(events), url, resp.text[:200])
        return True
    except httpx.HTTPError as exc:
        log.warning("sync_uhai: push to %s failed (%s) — falling back to offline export", url, exc)
        return False


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Force offline export even if UHAI_GREYCAT_URL is set")
    parser.add_argument("--reset", action="store_true", help="Ignore the watermark and resync everything from scratch")
    args = parser.parse_args(argv)

    _load_env()

    events, state = collect_events(reset=args.reset)
    if not events:
        print("sync_uhai: nothing new to sync (all sources unchanged since last run)")
        return

    base_url = os.environ.get("UHAI_GREYCAT_URL")
    token = os.environ.get("UHAI_GREYCAT_TOKEN")

    pushed = False
    if base_url and not args.dry_run:
        pushed = _push_to_greycat(events, base_url, token)

    if pushed:
        print(f"sync_uhai: pushed {len(events)} event(s) to {base_url}")
    else:
        out_path = _write_offline_export(events)
        mode = "dry-run" if args.dry_run else "offline (UHAI_GREYCAT_URL not set, or push failed)"
        print(f"sync_uhai: {mode} — wrote {len(events)} event(s) to {out_path}")

    _save_state(state)


if __name__ == "__main__":
    main()
