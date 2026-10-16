> English edition · [Frozen original](../../../../experiments/memory_interchange/report.md) · [English index](../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

# Execution report: BLOCKED

Date: 2026-10-12 (America/Los_Angeles). Protocol v1.0.

Research question: for identical full memory content with different formation histories, can geometric similarity of candidate memory states predict interchangeability when receiver control is retained?

This is an independent experimental implementation. Existing experiments and paper conclusions were not changed; no push or publication was performed in this version.

## Actual completion and gates

| Item | Status |
|---|---|
| Array truth and independent matrix evaluation; n=6/7 orbit verification | PASS |
| Orbit-level T/V/G/B splitting, content matching and timing checks | PASS |
| Control receives neither permutation nor m; content-isolated state caches | PASS |
| Sham, full-state and uninterrupted-continuation implementation controls | PASS |
| float64 CKA dual formulas, scale/joint-row-permutation invariance and known-matrix tests | PASS |
| LOSO/history-label isolation and fold-local missing-value handling | Synthetic implementation tests passed; not research results |
| GPU | NVIDIA GeForce RTX 5070 |

Independent verification: S₇'s 5,040 permutations form 694 orbits; S₆ has 115. Orbit sizes are 2, 4 or 8.

The 65 training words cover 7 net transformations. All 16 all-A/B four-slot words are new sequences, but only ABAB/BABA introduce the 8th net transformation. This is a data-generation property, not evidence of learned group structure.

| Dataset | Orbit count | Permutation count |
|---|---:|---:|
| T | 451 | 3274 |
| V | 104 | 778 |
| G | 69 | 492 |
| B | 70 | 496 |

The formal behavioral subset was fixed in advance at 134 permutations/19 complete orbits; remaining B contents are reserved.

split_seed=20261012; shuffle canonical orbits, round T/V/G down and assign the remainder to B. File SHA-256: `a53ec093b315c53fd7b6066ff5ca5c17508195d3035b85c37cb2ccec7448ef8c`.

Permutation input ends at stimulus offset; queries appear only after the swap time. Networks use final-query cross-entropy only, without intermediate supervision, swap training or geometry losses.

## Pilot: task learning before scientific analysis

| memory_dim | Updates | Best step | Fixed-V accuracy | V CE (nat) | Training seconds | Peak torch allocated memory (MiB) | ≥97% |
|---:|---:|---:|---:|---:|---:|---:|---|
| 128 | 20000 | 10000 | 14.7827% | 1.947035 | 123.054 | 310.56 | FAIL |
| 256 | 20000 | 11000 | 14.8926% | 1.946468 | 171.986 | 233.94 | FAIL |

Pilot seed=0; the same fixed validation set has 8,192 trials. Selection uses validation accuracy, then cross-entropy, without interchange-based capacity or seed selection.

Each capacity completed 20,000 Adam updates. No additional capacity, initialization, curriculum or hyperparameter search was performed.

![Pilot learning curves](../../../../experiments/memory_interchange/results/figures/pilot_training.png)

Neither the default nor sole fallback configuration passed the original-task gate. Formal execution stopped under protocol sections Twelve/Thirteen.

**H1 is inconclusive; H2 and H3 are untested. There are no formal G/B state measurements, six-seed labels, prediction errors, significance claims or scientific counterexamples.**

Formal seeds 101, 102, 103, 104, 105, 106 were not run, all because pilot task learning failed.

## Failure diagnostics: V only

Chance accuracy is 1/7=14.2857%; uniform-output cross-entropy is ln(7)=1.945910 nat. Both configurations validate near that level. Parameters do update and gradients are finite; failure does not mean the optimizer never executed.

| memory_dim | Answer-change rate after stimulus removal | Answer-change rate after stimulus shuffling | Maximum logit difference after stimulus removal | Content-effect L2 at step 8 | Content-effect L2 at step 37 |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.0000% | 0.0000% | 2.98e-08 | 1.43708 | 2.93999e-06 |
| 256 | 0.0000% | 0.0000% | 1.49e-07 | 1.38056 | 9.29054e-06 |

Content effect is the m-state L2 difference between original and zeroed stimuli for the same V trial with identical control input. Step 8 is stimulus offset; step 37 is the final d=0 query. Near-insensitivity of final output to initial content is consistent with decay during retention/updating, but does not alone identify the causal learning-failure mechanism.

![Content influence decays over time](../../../../experiments/memory_interchange/results/figures/pilot_content_sensitivity.png)

The first 52 V orbits serve geometry-throughput/probe diagnostics (392 contents). The first complete orbits from the latter half serve interventions (34 contents/6 orbits). The subsets are orbit-disjoint. Probes use alpha=1 ridge fitted on T only, never V/G/B.

| memory_dim | Maximum sham logp error | Maximum full-state logp error | Worst native V intervention stratum | V probe per-item accuracy | V probe full-permutation accuracy |
|---:|---:|---:|---:|---:|---:|
| 128 | 0 | 2.38e-07 | 20.1681% | 12.4544% | 0.0000% |
| 256 | 0 | 2.38e-07 | 19.3277% | 13.0892% | 0.0000% |

Caches retain naturally formed states, per-slot truth trajectories, donor initial permutations and all-query native/sham/matched/wrong-content/same-context-full-state outputs. Wrong content uses a fixed cross-orbit derangement, with receiver and donor-counterfactual answers saved. Every raw record is retained for implementation diagnosis only.

**Matched and wrong-content replacements behave alike in these diagnostics. Near-zero new errors and tiny probability damage cannot be called a learned-content compatibility ceiling: the native network already fails the task, preventing identification of H1/H2/H3.**

Pilot similarity–damage correlations/predictors were not computed, and high-CKA counterexamples were not selected. Failed task learning also prevented the unseen four-step extension.

## Computation cost and formal-entry decision

- 128 dimensions: approximately 162.5 training updates/second; full V intervention evaluation 3.915 seconds (approximately 223232 primary matched queries/second, with controls and saving included in timing); V geometry/initialization/probe diagnostics 2.543 seconds. Peak torch allocation was 692.91/198.15 MiB for the two evaluations. Linear extrapolation by content count, plus 120 seconds of geometry margin per formal seed, estimates 0.431 GPU hours for six-seed training/main evaluation; this excludes machine variation in CPU bootstrap and file writing.
- 256 dimensions: approximately 116.3 training updates/second; full V intervention evaluation 3.846 seconds (approximately 227262 primary matched queries/second, with controls and saving included in timing); V geometry/initialization/probe diagnostics 4.821 seconds. Peak torch allocation was 489.07/149.84 MiB for the two evaluations. Linear extrapolation by content count, plus 120 seconds of geometry margin per formal seed, estimates 0.512 GPU hours for six-seed training/main evaluation; this excludes machine variation in CPU bootstrap and file writing.

These estimates assess budget feasibility only. Failed original-task gates already prevent formal execution; GPU availability and budget are not this version's blocker.

Cumulative GPU-operation wall-clock accounting: 337.979 seconds (0.0939 hours), capped at 12 hours. Accounting includes overhead around training, validation, caching and GPU diagnostics. Briefly overlapping training/diagnostic intervals are counted separately, making the estimate conservative.

## Reproducible artifacts and limitations

- `config.yaml`, `split_manifest.json` and `results/frozen_plan.json` preserve protocol, orbits, fixed B subset and source hashes.
- `runs/pilot_128/` and `runs/pilot_256/` each contain `initial.pt`, `best.pt`, `latest.pt`, curves, complete optimizer and recoverable Python/NumPy/Torch/CUDA RNG states.
- `analysis/raw_d*_S*.npz` saves outputs/truth by donor, receiver, content, query type and item. `orbit_blocks.npz` saves statistics and query counts for every orbit/direction/query type.
- `directions.csv`, `native_strata.csv`, `labels.csv` retain direction/context/S/d strata. Raw data suffice to recompute metrics; shared-state queries are not independent CKA samples.
- Prediction uses fixed alpha=1 ridge. Outer folds hold out model seeds; fitting uses 91 pairs without AB/BA; 62 unseen pairs include 32 cross-context pairs. The same implementation provides fold-local preprocessing, initialization-CKA checks and hierarchical seed/behavioral-orbit resampling. Synthetic implementation-test values are not reported as scientific results.

The m/c modules are designated architecturally. Unidirectional control does not establish a naturally independent memory brain area. Exact mathematical content labels do not establish learned algebraic structure or justify extrapolation to biological systems.

This version did not update related-work searches or assess originality. Protocol references provide initial context only; verify versions and neighboring work before submission.

![Task and intervention diagram](../../../../experiments/memory_interchange/results/figures/01_task.png)

**Final status: BLOCKED.**
