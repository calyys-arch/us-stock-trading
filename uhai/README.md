# UHAI microstructure pack

This folder is **not part of this repo's Python/TypeScript app** — it is a
[GreyCat](https://greycat.io) pack meant to be copied into a separate
Universal Hybrid AI (UHAI) GreyCat project, where it sits alongside an
existing `src/trading_signals/` pack. It gives the trading system's
report-only microstructure diagnostic layer (see
[`../docs/microstructure_pivot_plan.md`](../docs/microstructure_pivot_plan.md))
a queryable, relationship-aware home, separate from this repo's own
Parquet/JSONL caches which stay the source of truth for raw bars/ticks.

## Why a separate store at all

This repo already has fast, cheap local storage (`data/history_1m/*.parquet`,
`data/ticks/`, `data/depth/`, `docs/*.md` reports) for the actual trading
data. None of that needs to move. What GreyCat's triple store adds is a
place to record **relationships between distilled findings** — "this
parameter set passed WFO for this signal in this window", "this universe
snapshot was live when that promotion happened" — the kind of cross-cutting
question a flat file layout answers poorly. Concretely: *volume* (bars,
ticks) stays in Parquet; *meaning* (decisions, verdicts, snapshots) goes
here. See the "distill first, then graph" rule in
`scripts/sync_uhai.py`'s module docstring.

## Deployment

1. Copy this folder's contents into your UHAI project:
   ```sh
   cp -R microstructure/*.gcl <uhai-project>/src/microstructure/
   ```
2. Confirm `<uhai-project>/project.gcl` already has `@include("src");` (it
   should, if `src/trading_signals/` is already included the same way).
3. From the UHAI project root:
   ```sh
   greycat-lang lint project.gcl   # 0 errors expected
   greycat build
   greycat serve                   # or `greycat run` for one-shot scripts
   ```

The pack was lint/build/test-verified against `greycat 8.0.555-dev` /
`std 8.0.635-stable` in an isolated scratch project before being committed
here — see `model.gcl` and `api.gcl` for the schema and exposed functions.

## Files

- `microstructure/model.gcl` — the `MicroEvent` node type (a
  subject/predicate/object triple with a `source` and `ts`) plus its two
  indices: `events_by_time` (range queries) and `events_by_subject`
  ("everything about `signal:sweep_reclaim`").
- `microstructure/api.gcl` — `@expose @tag("mcp")` functions:
  `ingestMicroEvents`, `queryMicroEvents`, `countMicroEventsInRange`. These
  are the three calls `../scripts/sync_uhai.py` makes over HTTP.

## What gets synced (and what doesn't)

`../scripts/sync_uhai.py` only ever pushes **already-distilled** JSON
records — WFO/gate verdicts, parameter sets that were tested, universe
snapshots, promotion history. It never pushes raw 1-minute bars, ticks, or
per-bar signal detections; those stay local. See that script's docstring
for the exact sources and the `subject`/`predicate` vocabulary used.
