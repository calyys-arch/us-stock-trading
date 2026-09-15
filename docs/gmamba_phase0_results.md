# G-Mamba-lite Phase 0 kill-test — results

**Date**: 2026-09-15
**Trigger**: user pointed at their other repo, `calyys-arch/Universal-Hybrid-AI`
(private; cloned locally to inspect, since GitHub returned 404 until the
account was connected), and asked how its recent upgrades could apply
here. The most relevant new capability since the last integration review
(`docs/uhai_integration_review.md`, 2026-09-10) is a graph-conditioned
selective-state-space forecasting model ("G-Mamba") just added to that
repo's `trading-signals` pack (commit `9ae2f54`). This doc is the Phase 0
kill-test of whether that architecture finds signal on **this** repo's own
data — same falsification-first discipline as `docs/v2_research_plan.md`'s
momentum and 8-K drift tests.

## 0. Why not just reuse the Universal-Hybrid-AI pack directly

Checked and ruled out before writing any code — see the conversation this
doc was written in for the full audit:

1. **Different upstream data.** `industry-packs/trading-signals/connectors/signal_outcome_jsonl.gcl`'s
   own header says it is fed by a *different* product ("stock_analysis"
   owns the source of truth), not this repo. This repo's own
   `scripts/sync_uhai.py` pushes to a different pack (`microstructure`)
   entirely disconnected from `trading-signals`.
2. **Untrained.** `python-svc/main.py` in that repo states `/forecast/signal`
   "Serves an UNTRAINED model until GMAMBA_CHECKPOINT_PATH points" to a
   real checkpoint — there is no measured IC or backtest result for it
   anywhere in that repo.
3. **Capacity clash with this repo's Chan discipline.** `python/backtest/param_guard.py`
   caps every live/paper strategy at `MAX_FREE_PARAMETERS=5`. Even a
   heavily scaled-down G-Mamba has thousands of trainable weights (10,066
   in this test — see §2). That is not a difference of degree; it needs
   its own validation paradigm, not a squeeze into the existing WFO/Monte
   Carlo/Reality Check gate stack.

So: vendor the architecture (`python/research/gmamba_lite.py`, ~250 lines,
trimmed of the three optional structural-regularization losses this test
doesn't use), point it at this repo's own universe, and hold it to the
**same** bar as every other Phase 0 kill-test, not a looser one.

## 1. Setup

| | |
|---|---|
| Universe | Same 127-name mid+large-cap combined universe as `docs/v2_research_plan.md`'s momentum Phase 0 (`scripts/run_v2_momentum_phase0.py --universe combined`) |
| Features (7, OHLCV-derived only) | `ret_1d`, `ret_5d`, `ret_21d`, `vol_20d`, `dollar_vol_z`, `dist_from_high_20d`, `dist_from_low_20d` — cross-sectionally rank-normalized to [0,1] per day |
| Why not the pack's actual Wyckoff/VSA flags | Those are computed and stored historically for the *other* product's universe, not this one. Substituting technical features is a disclosed simplification, not a hidden one. |
| Static graph | All-zero placeholder (V×V) — no verified sector/co-holding map exists for this ad-hoc 127-name universe; fabricating one would be building a result, not testing one. The model's own learned `DynamicAdjacency` is still live. |
| Sequence length (L) | 20 trading days, matching the original pack's `lookbackDays()` |
| Target | Forward 21-trading-day return, matching momentum/8-K Phase 0's horizon |
| Sampling | One cross-section every 21 trading days (~monthly) |
| Split | Chronological 70/30 with a 21-trading-day embargo gap (purge/embargo) between train and val |
| Model | `GMambaConfig(d_model=16, n_layers=1, d_state=8, expand=2, d_attn=8)` — heavily scaled down from the original's defaults (`d_model=64, n_layers=2, d_state=16`) |
| Training | 60 epochs, full-batch Adam, fixed epoch count declared **before** running — no best-epoch checkpoint selection on val (that itself would be a form of snooping on the only measurement that decides the verdict) |

**Falsification criteria (declared before running)**: NO-GO if mean per-date
IC \(\le\) 0, or \(|t\text{-stat of per-date IC}| < 2\).

## 2. Result

```
n_params                               10066   (2013x param_guard.py's MAX_FREE_PARAMETERS=5)
n_train_samples                        88      (2016-03-03 .. 2023-06-06)
n_val_samples                          37      (2023-08-07 .. 2026-08-12, after embargo)
mean_daily_ic                          0.058
ic_t_stat                              1.47
tercile_spread_mean_monthly            1.63%
tercile_spread_sharpe_annualized_gross 0.93    (gross, no costs modeled)

VERDICT: NO-GO (Phase 0) — IC t-stat 1.47, |t| < 2
```

Full machine-readable output: `backtests/gmamba_phase0/phase0_seed42_epochs60.json`.

## 3. How this compares to the other two "different mechanism" Phase 0 tests

| Test | Spread/IC t-stat | Verdict | Sample |
|---|---|---|---|
| 12-1 momentum (combined universe) | 0.16 | NO-GO — flat null | 116 months |
| 8-K filing drift (Option D) | 0.84 | NO-GO — directionally right, not significant | 554 events / 66 months |
| **G-Mamba-lite (this test)** | **1.47** | **NO-GO — closest to the bar of the three, still short** | **37 validation months** |

Reading this honestly, not hopefully: the higher t-stat is exactly what a
much higher-capacity, graph-aware model would produce even under the null
— more free parameters fit training noise more aggressively, which can
leak into a still-small (37-point) validation set as an apparently
stronger IC without it being a real, stable effect. A t-stat of 1.47 on 37
non-overlapping monthly cross-sections is **not** evidence of edge; it is
"not yet cleanly rejected," which is a weaker claim. The proper next
check — before spending more effort here — would be a second held-out
period or a different random seed/model-size sweep to see whether 1.47
is stable or noise; that has NOT been done, and this doc explicitly avoids
treating one run's t-stat as a green light to keep iterating on hyperparameters
until it crosses 2.0 (that would be p-hacking with extra steps, the same
failure mode `docs/v2_research_plan.md` already documents guarding against
for the Bayesian-suggester UHAI feature).

## 4. Recommendation

**Do not proceed to a Phase 1** (larger universe / longer history / real
static graph / hyperparameter search) on the strength of this one run.
The result is a NO-GO by the pre-declared bar, and the model's much larger
hypothesis space makes "try a few more configurations" a materially
riskier move here than it was for momentum or 8-K drift — the same
multiple-comparisons concern the user raised earlier this session about
Bayesian parameter search applies with more force to a model this size on
data this small (88 training cross-sections is not enough to responsibly
tune thousands of weights).

If this is ever revisited, it should only be with: (a) a genuinely larger,
point-in-time-correct universe (500+ names, not 127 ad-hoc ones), (b) a
real structural graph (sector/ETF co-holding, not an all-zero
placeholder), and (c) a pre-registered, single hyperparameter
configuration tested exactly once against a fresh, never-before-touched
validation window — the same one-shot discipline `scripts/design_wfo_gate.py`
already established for this repo's existing gates.

## 5. What stays useful regardless of this verdict

- `python/uhai/explain.py` + the `check_uhai_recommendations.py`
  strategy-list fix (see the accompanying commit) — independent of G-Mamba,
  low-risk, already merged.
- `python/research/gmamba_lite.py` is a clean, reusable vendored copy of
  the architecture if a properly-scoped Phase 1 is ever authorized later;
  no need to re-port it from the equations again.
