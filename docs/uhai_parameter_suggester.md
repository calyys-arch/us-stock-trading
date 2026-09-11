# UHAI parameter suggester

What UHAI adds to `scripts/self_improve_loop.py`, what it deliberately does
not touch, and what was tried and abandoned.

This supersedes `uhai_smart_search_architecture.md`,
`uhai_smart_search_integration_report.md`, `uhai_quickstart.md`,
`uhai_integration_issues_review.md`, `uhai_integration_fixes_applied.md` and
`uhai_integration_final_review.md`, all of which described a GreyCat/GCL
design that does not compile and could not work as specified (see
"Abandoned: the GCL memory layer" below).

## The gap it fills

`self_improve_loop.py` searched a fixed coarse grid from
`configs/param_grids.yaml` — 27 candidates for `pairs_trading`, 6 for
`xsection_mean_reversion` — and started from scratch every run. Two things
were missing:

1. **Resolution.** The grid samples `entry_z` at 1.5 / 2.0 / 2.5. If the real
   optimum sits at 2.28, no run can ever find it.
2. **Memory.** Every past run's results were already being written to
   `backtests/logs/promotion_history.jsonl` and then never read again.

## What it does

`python/uhai/history.py` reads that history back. `python/uhai/client.py`
sends it to UHAI's Python sidecar (`POST /optimize/suggest-params`), which
fits a Gaussian Process over (parameters -> OOS Sharpe) and returns points
worth testing next: its predicted optimum first, then Expected-Improvement
points around it.

Those points are **appended to** the fixed grid, never substituted for it:

```
=== iter 0 | pairs_trading | window [2018-01-02, 2021-11-01] | 32 candidates ===
    candidates: 27 fixed grid + 5 UHAI suggestion(s)
```

Every suggestion then goes through exactly the same pipeline as a grid
candidate — per-fold in-sample re-optimization, out-of-sample validation, and
all five gates (`wfo_go`, `oos_drawdown_within_limit`, `has_oos_trades`,
`monte_carlo_p5_sharpe`, `reality_check_pass`). Nothing here shortcuts a
verdict or writes a config.

## Constraints it is held to

- **Suggestions cannot escape the grid's range.** The search box is the
  min/max of each axis in `configs/param_grids.yaml`, so a suggestion can only
  ever refine within a range a human already sanctioned.
- **Suggestions cannot introduce a parameter.** A suggestion with a key absent
  from `configs/strategy.yaml` is dropped.
- **Chan's ceiling is re-checked per suggestion.** `preflight_check` runs
  against the fixed grid before suggestions arrive, so each one is separately
  put through `check_max_parameters` — otherwise it would be the one candidate
  in the run that never met the 5-parameter limit.
- **Integer axes stay integers.** `xsection_mean_reversion`'s `lookback_days`
  is a count of days; a GP proposing 2.7 is rounded to 3. Continuous axes are
  rounded to 3 decimals, which is all the resolution a coarse grid over a few
  years of bars supports.
- **Boolean axes are excluded.** A Real-valued GP over a boolean would emit
  0.37. Those axes stay grid-only.
- **Synthetic history is excluded from real runs.** A `--demo` run records
  `data_source: synthetic`; fitting the suggester that proposes real-money
  parameters on synthetic Sharpes would be worse than having no memory.
- **`enabled` and `auto_execute` are untouched.** They remain in
  `promotion.py`'s `_FORBIDDEN_WRITE_KEYS`. Going live stays a human decision.

## Failure is the normal case

The sidecar being down is expected, not exceptional. Every failure path
returns no suggestions and the run proceeds on the fixed grid:

```
UHAI: no sidecar at http://127.0.0.1:8092 — fixed grid only
=== iter 0 | pairs_trading | ... | 27 candidates ===
```

`--disable-uhai` does the same thing deliberately.

The client also **probes `/health` and refuses to POST to a service it cannot
identify as the sidecar**. This matters concretely: UHAI's own default sidecar
port is 8082, which in this repo is the dashboard
(`scripts/start_dashboard.py`, `README.md`, `watch_paper_session.py`'s
hard-coded `DASH_URL`) — an API surface that includes `/api/engine/start` and
`/api/positions/flatten_all`. Hence the distinct default port (8092), the
distinct variable name (`UHAI_PYTHON_SVC_URL`, not the `PYTHON_SVC_URL` UHAI's
own GCL reads), and the identity probe.

## When it uses a GP, and when it does not

A Gaussian Process needs roughly 4 observations per dimension before it
localises anything. Below that the reply is a Latin Hypercube sample and is
labelled as such:

```
UHAI: 5 suggestion(s) via latin_hypercube from 9 history point(s)
```

`pairs_trading` has 3 continuous axes, so it needs 12 real-data runs before
the GP path engages. Until then the space-filling design is the honest answer.

This threshold is load-bearing rather than cosmetic. skopt's `Optimizer`
defaults to `n_initial_points=10` and counts it down as `tell()` is called, so
supplying history alone leaves it in random initialisation — `ask()` returns
uniform random points while the reply still claims to be a GP result. The
sidecar sets `n_initial_points=1` and falls back to LHS if no surrogate was
fitted, and `test_bayesian_optimizer.py` guards both.

## Staleness reporting

`scripts/check_uhai_recommendations.py` reports, per strategy, how long ago
its parameters were last promoted and whether the LIVE parameters' OOS
Sharpe — re-measured on each run's window via `baseline_oos_sharpe`, not the
best candidate that run's search happened to try — is trending down. It
recommends only — it runs no backtest and writes no config. Exit code 2
means something looks stale, so cron can branch on it.

This distinction is load-bearing, not cosmetic: `promotion.py` records both
numbers on every run regardless of decision. `candidate_oos_sharpe` is
whatever the grid/UHAI search tried that run — on a REJECTED run (most runs)
that candidate was thrown away, so its trend says nothing about whether the
strategy actually deployed still works; a search that keeps finding
better-looking-but-rejected candidates can rise even while the live strategy
decays. `baseline_oos_sharpe` is `configs/strategy.yaml`'s CURRENT values,
scored fresh on that run's window every time — its trend is the one honest
answer to "is what I actually have live still working on recent data".
`assess_degradation()` reads the latter; a malformed record missing it is
excluded from the trend rather than treated as a zero.

Each promotion record now also carries `extra.market_regime` — which regime
(`Bull` / `Bear` / `Sideways`) the tested window mostly sat in, from
`python/analytics/regime.py`'s existing label. It is descriptive: no gate
reads it, so it cannot change a verdict. It is recorded so a later run can ask
what a parameter set scored the last time the market looked like today.

## Is fitting on past runs look-ahead bias?

No. The GP chooses *which parameters to test*; it never sees the out-of-sample
data those parameters are then judged on. Each suggested candidate still goes
through the same walk-forward IS/OOS separation as a grid candidate.

The real statistical cost is multiple comparisons: 32 candidates instead of 27
means slightly more opportunity for a spurious winner. That is what the
Reality Check gate is for — and it now accounts for the actual search width.
`RealityCheck.run(..., n_candidates_searched=len(param_grid))` applies a
Bonferroni correction (`p_adjusted = min(1, p_raw * N)`) at zero extra cost, so
a run with UHAI suggestions added (a wider search) is held to a
correspondingly higher bar than the fixed grid alone. An exact, less
conservative alternative (`RealityCheck.run_multi`, resampling the best-of-N
statistic directly) exists for a slower, periodic validation pass — see
`python/backtest/reality_check.py`'s module docstring for the measured cost
tradeoff (~1 min vs ~30 min per call on this repo's pairs backtest) and why it
is not the per-iteration default.

## Running it

```bash
# fixed grid only — the default when no sidecar is configured
python scripts/self_improve_loop.py --strategy pairs_trading --no-write

# explicitly without suggestions
python scripts/self_improve_loop.py --disable-uhai --no-write

# with the suggester
cd <universal-hybrid-ai>/python-svc && PYTHON_SVC_PORT=8092 python main.py &
UHAI_PYTHON_SVC_URL=http://127.0.0.1:8092 \
  python scripts/self_improve_loop.py --strategy pairs_trading --no-write

# staleness report
python scripts/check_uhai_recommendations.py
```

The sidecar needs `scikit-optimize>=0.10.2` (added to its
`requirements.txt`). Drop `--no-write` only after reviewing what a run
proposes.

## Tests

- `tests/test_uhai_suggester.py` (29) — synthetic-history exclusion, parameter
  shape changes, integer-axis coercion, bounds, degradation, regime tagging,
  and the refusal to POST to an unidentified local service.
- `<universal-hybrid-ai>/python-svc/test_bayesian_optimizer.py` (12) — that a
  `gaussian_process` label means a surrogate was actually fitted, and that
  suggestions move toward the best observed region rather than away from it
  (a sign error there is the one failure mode capable of doing real damage
  over repeated runs).

## Abandoned: the GCL memory layer

An earlier attempt put the memory in GreyCat as
`universal-hybrid-ai/src/trading/param_optimization.gcl` (~450 lines), querying
history with SPARQL and calling the sidecar over `http::post`. It was removed
rather than repaired, for three independent reasons:

1. **It did not compile.** `greycat build` fails at its first statement
   (`use core;` — GCL has no such form; no real file in the project uses one).
   Because `project.gcl` already has `@include("src")`, the file was breaking
   the entire UHAI project build, not just its own feature.
2. **Its APIs do not exist.** No SPARQL in this codebase; the microstructure
   pack exposes `queryMicroEvents`, not `microstructure::query()`; HTTP goes
   through `io::Http` / `SidecarClient::postJson`, not `http::post`.
3. **Its premise was unimplementable.** The pack stores payloads as an
   `object_json` string and GCL has no JSON parser, so parameters written by
   `scripts/sync_uhai.py` cannot be read back as structured values there. On
   top of that, `src/microstructure/` had never been deployed into UHAI.

Reading the history locally in Python is simpler, has no deployment
dependency, costs no network hop, and is testable. GreyCat keeps the role it
is actually good at: the relationship-aware store of distilled findings that
`scripts/sync_uhai.py` already writes to.
