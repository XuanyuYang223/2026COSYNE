> English edition · [Frozen original](../../../../../experiments/memory_interchange/v1_2/report.md) · [English index](../../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

# v1.2 execution report: BLOCKED

Date: 2026-10-12 (America/Los_Angeles).

Status **BLOCKED**, reason `pilot_stage_budget_exhausted`. The original question, primary intervention and scientific metrics follow v1.0.

## Restoration, preservation and curve review

Hashes remain unchanged for 292 v1.0/v1.1 source, configuration, checkpoint, audit and report files. The original split and 134 formal B contents are unchanged. Candidate B was not continued.

A was preferred for old-V two-step accuracy 84.9195% and worst-operation accuracy 78.0255%, more balanced than B's 83.3821%/56.25% while retaining default initialization. Selection did not use geometry, interchange or formal-seed results. One pilot difference cannot establish generally superior initialization.

Restore source: `/home/yangx/2026COSYNE/experiments/memory_interchange/runs/v1_1/candidate_A_seed0/stage_6/latest.pt`. SHA-256: `733e4cdf820a11554d004400c0f8de76c48747b3bf59d4a00bd7f480c9c885c7`.

Source latest has 5,000 stage updates and 18,500 cumulative updates; every Adam parameter has step=18,500. Weights and Python/NumPy/Generator/Torch/CUDA RNG match exactly. Fixed T/V accuracies and stratum counts agree; CE agrees within preregistered 1e-6 nat tolerance.

Absolute T/V replay errors: {'T': {'accuracy': 0.0, 'cross_entropy': 0.0}, 'V': {'accuracy': 0.0, 'cross_entropy': 0.0}}. Rounding and restoration errors were checked separately; no stage 5 checkpoint was substituted. The old stage 6 report used endpoint/latest; best was also at 5,000.

All ten existing stage 6 evaluations at 500-step intervals are saved, including the last 2,000 updates' T/V, CE, k, actual operations, context, d and query type. Over the last 2,000 updates, total-V accuracy changed by +4.1992 percentage points, two-step accuracy by +11.4202, and CE by −0.101599 nat. Sampled evaluations continued improving. This trend alone does not identify a failure mechanism.

![Prior-curve review](../../../../../experiments/memory_interchange/results/v1_2/figures/previous_curve_review.png)

## Curriculum and checkpoint gates

Fixed V is evaluated every 500 updates; total accuracy ≥97% and every original factor ≥95% must pass twice consecutively. Stage 6's cap includes the existing 5,000 updates, allowing at most 45,000 additional updates. Stage 7 allows 50,000, using the original 65 words and fixed 8,192-trial V. Formal early stages allow 20,000 each; per-model maximum is 200,000. No new initializations, capacity, learning rates, repeated stimuli, auxiliary losses or error-targeted sampling were added.

| Stage | Task | Actual stage updates | Cumulative course updates | V accuracy | V CE nat | Worst true-operation stratum | Two consecutive advancement passes |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | k≤0 / short | 4500 | 4500 | 99.2920% | 0.0440458 | 99.2920% | True |
| 2 | k≤0 / mid | 1000 | 5500 | 99.9756% | 0.00990222 | 99.9756% | True |
| 3 | k≤0 / original | 1000 | 6500 | 100.0000% | 0.00381966 | 100.0000% | True |
| 4 | k≤1 / original | 4000 | 10500 | 99.4385% | 0.032757 | 98.9258% | True |
| 5 | k≤1 / original | 3000 | 13500 | 99.0967% | 0.0370515 | 97.9146% | True |
| 6 | k≤2 / original | 15000 | 28500 | 98.9258% | 0.0338632 | 96.4968% | True |
| 7 | k≤3 / original | 50000 | 78500 | 98.3154% | 0.0497601 | 91.6058% | False |

Total pilot updates: 78500; added this version: 60000. Stages 1–5 were inherited without retraining. Actual steps, hashes and Adam/RNG for latest/best/advance are in checkpoint_inventory.json; no advance checkpoint is invented for a failed stage.

![Continuous learning curves](../../../../../experiments/memory_interchange/results/v1_2/figures/continuous_learning.png)

Stage 6's 33-word result is not the full task. Main H+S can contain three non-N updates, requiring stage 7. Latest, best and advance remain separate; high overall accuracy with failed factor strata does not make best an advancement checkpoint.

### Complete operation strata at stage 7 termination

| True operation | T n | T accuracy | T CE nat | V n | V accuracy | V CE nat |
|---|---:|---:|---:|---:|---:|---:|
| C | 1034 | 99.6132% | 0.0108708 | 982 | 99.0835% | 0.0205363 |
| CC | 525 | 99.8095% | 0.00997053 | 521 | 99.6161% | 0.0150648 |
| CCC | 242 | 96.2810% | 0.0954364 | 274 | 91.6058% | 0.213869 |
| CCI | 270 | 98.5185% | 0.0497546 | 264 | 94.3182% | 0.155947 |
| CI | 519 | 98.2659% | 0.0403657 | 523 | 97.7055% | 0.0523889 |
| CIC | 266 | 91.3534% | 0.232824 | 247 | 92.3077% | 0.246032 |
| CII | 259 | 96.5251% | 0.10244 | 263 | 97.3384% | 0.103279 |
| I | 1014 | 99.9014% | 0.00501856 | 1023 | 99.7067% | 0.00665771 |
| IC | 504 | 98.8095% | 0.0408175 | 509 | 99.4106% | 0.025052 |
| ICC | 253 | 98.0237% | 0.0694801 | 256 | 98.4375% | 0.0832327 |
| ICI | 246 | 90.2439% | 0.248563 | 266 | 93.9850% | 0.185912 |
| II | 500 | 99.6000% | 0.00678919 | 491 | 99.7963% | 0.019684 |
| IIC | 242 | 96.6942% | 0.100158 | 260 | 96.1538% | 0.0984419 |
| III | 270 | 97.7778% | 0.0582304 | 274 | 98.5401% | 0.0432373 |
| KEEP | 2048 | 99.8047% | 0.00488328 | 2039 | 99.5096% | 0.013285 |

| k | T n | T accuracy | V n | V accuracy |
|---:|---:|---:|---:|---:|
| 0 | 2048 | 99.8047% | 2039 | 99.5096% |
| 1 | 2048 | 99.7559% | 2005 | 99.4015% |
| 2 | 2048 | 99.1211% | 2044 | 99.1194% |
| 3 | 2048 | 95.7031% | 2104 | 95.3422% |

Stage 7 best: step 49500, V 98.6084%. This table uses the terminating latest checkpoint; best and latest are not conflated.


### Restore point step 5,000: fixed T/V two-step diagnostics

| Dataset | Two-step count | Single-query accuracy | Correct full forward mapping | Correct full reverse mapping | Both mappings correct | Forward bijective | Reverse bijective | Query mappings are mutual inverses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T | 1365 | 83.7363% | 35.5311% | 40.2930% | 20.0000% | 36.8498% | 41.2454% | 20.0000% |
| V | 1366 | 84.9195% | 31.9912% | 38.2138% | 16.4714% | 33.3089% | 39.7511% | 16.4714% |

Candidate answers are correct-order two-step, first-only, second-only, no update and reversed order. Every original trial retains all candidate answers/mappings and pairwise overlaps. The table uses only queries where candidate and correct answers differ; empty subsets support no judgment.

| Dataset | True order | Candidate | Distinguishable query count | Prediction agreement |
|---|---|---|---:|---:|
| T | II | first_only | 238 | 8.8235% |
| T | II | second_only | 238 | 8.8235% |
| T | II | none | 0 | Indistinguishable |
| T | II | reversed | 0 | Indistinguishable |
| T | CC | first_only | 293 | 10.9215% |
| T | CC | second_only | 293 | 10.9215% |
| T | CC | none | 0 | Indistinguishable |
| T | CC | reversed | 0 | Indistinguishable |
| T | IC | first_only | 287 | 9.7561% |
| T | IC | second_only | 233 | 8.5837% |
| T | IC | none | 293 | 12.9693% |
| T | IC | reversed | 293 | 8.1911% |
| T | CI | first_only | 239 | 15.0628% |
| T | CI | second_only | 291 | 3.7801% |
| T | CI | none | 295 | 7.7966% |
| T | CI | reversed | 289 | 5.8824% |
| V | II | first_only | 253 | 7.1146% |
| V | II | second_only | 253 | 7.1146% |
| V | II | none | 0 | Indistinguishable |
| V | II | reversed | 0 | Indistinguishable |
| V | CC | first_only | 291 | 9.6220% |
| V | CC | second_only | 291 | 9.6220% |
| V | CC | none | 0 | Indistinguishable |
| V | CC | reversed | 0 | Indistinguishable |
| V | IC | first_only | 305 | 6.2295% |
| V | IC | second_only | 270 | 6.2963% |
| V | IC | none | 325 | 8.6154% |
| V | IC | reversed | 314 | 5.7325% |
| V | CI | first_only | 229 | 20.0873% |
| V | CI | second_only | 274 | 4.3796% |
| V | CI | none | 270 | 6.2963% |
| V | CI | reversed | 269 | 7.4349% |

All-query agreement, agreement on subsets uniquely distinguishing each candidate, and full-candidate-mapping agreement are saved separately in diagnostic summary.json and NPZ. Coincident candidates are not evidence of skipped/reversed operations. Reversing II/CC equals correct execution; IC/CI patterns are described only on predefined distinguishable subsets. Complete-query reconstruction uses existing readouts, without writing truth into hidden states or adding training losses.

Both actual operations and visible words are saved: IC is AB for r=0, BA for r=1; CI reverses this correspondence. Stratified CSVs retain counts, accuracy and CE for actual operations, context, visible order, six slot pairs, d, query type and descriptive crossed cells. Such patterns suggest further tests, not a unique internal mechanism.

T true-operation strata below the original 95% gate: CC, CI, IC, II. Save stimulus-offset/swap/final-query sensitivity, actual-graph gradients and gate distributions for each, selecting at most 64 trials in fixed order, without L2/gate thresholds.
V true-operation strata below the original 95% gate: CC, CI, IC, II. Save stimulus-offset/swap/final-query sensitivity, actual-graph gradients and gate distributions for each, selecting at most 64 trials in fixed order, without L2/gate thresholds.

| V true order | r | Visible order | n | Accuracy | CE nat |
|---|---:|---|---:|---:|---:|
| II | 0 | AA | 175 | 94.8571% | 0.202081 |
| II | 1 | BB | 168 | 91.6667% | 0.292572 |
| CC | 0 | BB | 174 | 86.7816% | 0.361061 |
| CC | 1 | AA | 167 | 83.8323% | 0.455777 |
| IC | 0 | AB | 175 | 82.8571% | 0.432541 |
| IC | 1 | BA | 193 | 82.3834% | 0.52152 |
| CI | 0 | BA | 149 | 87.9195% | 0.276132 |
| CI | 1 | AB | 165 | 69.0909% | 0.750604 |

Nonzero content sensitivity establishes stimulus influence, not storage of the correctly updated complete content. Correct behavior could also result from history stored in control and computation at final readout; the original interchange analysis remains necessary.


### Stage 6 advancement/termination: fixed T/V two-step diagnostics

| Dataset | Two-step count | Single-query accuracy | Correct full forward mapping | Correct full reverse mapping | Both mappings correct | Forward bijective | Reverse bijective | Query mappings are mutual inverses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T | 1365 | 97.8755% | 86.6667% | 88.4982% | 77.8755% | 86.7399% | 88.4982% | 77.8755% |
| V | 1366 | 97.8038% | 82.5037% | 87.6281% | 73.0600% | 82.5037% | 87.7745% | 73.0600% |

Candidate answers are correct-order two-step, first-only, second-only, no update and reversed order. Every original trial retains all candidate answers/mappings and pairwise overlaps. The table uses only queries where candidate and correct answers differ; empty subsets support no judgment.

| Dataset | True order | Candidate | Distinguishable query count | Prediction agreement |
|---|---|---|---:|---:|
| T | II | first_only | 238 | 0.8403% |
| T | II | second_only | 238 | 0.8403% |
| T | II | none | 0 | Indistinguishable |
| T | II | reversed | 0 | Indistinguishable |
| T | CC | first_only | 293 | 0.3413% |
| T | CC | second_only | 293 | 0.3413% |
| T | CC | none | 0 | Indistinguishable |
| T | CC | reversed | 0 | Indistinguishable |
| T | IC | first_only | 287 | 1.0453% |
| T | IC | second_only | 233 | 0.0000% |
| T | IC | none | 293 | 1.3652% |
| T | IC | reversed | 293 | 0.6826% |
| T | CI | first_only | 239 | 3.7657% |
| T | CI | second_only | 291 | 0.3436% |
| T | CI | none | 295 | 0.3390% |
| T | CI | reversed | 289 | 0.6920% |
| V | II | first_only | 253 | 1.1858% |
| V | II | second_only | 253 | 1.1858% |
| V | II | none | 0 | Indistinguishable |
| V | II | reversed | 0 | Indistinguishable |
| V | CC | first_only | 291 | 0.3436% |
| V | CC | second_only | 291 | 0.3436% |
| V | CC | none | 0 | Indistinguishable |
| V | CC | reversed | 0 | Indistinguishable |
| V | IC | first_only | 305 | 1.3115% |
| V | IC | second_only | 270 | 1.1111% |
| V | IC | none | 325 | 0.9231% |
| V | IC | reversed | 314 | 0.6369% |
| V | CI | first_only | 229 | 1.7467% |
| V | CI | second_only | 274 | 0.7299% |
| V | CI | none | 270 | 0.7407% |
| V | CI | reversed | 269 | 1.1152% |

All-query agreement, agreement on subsets uniquely distinguishing each candidate, and full-candidate-mapping agreement are saved separately in diagnostic summary.json and NPZ. Coincident candidates are not evidence of skipped/reversed operations. Reversing II/CC equals correct execution; IC/CI patterns are described only on predefined distinguishable subsets. Complete-query reconstruction uses existing readouts, without writing truth into hidden states or adding training losses.

Both actual operations and visible words are saved: IC is AB for r=0, BA for r=1; CI reverses this correspondence. Stratified CSVs retain counts, accuracy and CE for actual operations, context, visible order, six slot pairs, d, query type and descriptive crossed cells. Such patterns suggest further tests, not a unique internal mechanism.

T true-operation strata below the original 95% gate: none. Diagnostics save stimulus-offset/swap/final-query sensitivity, actual-graph gradients and gate distributions for failing strata, with at most 64 fixed-order trials per stratum and no L2/gate thresholds.
V true-operation strata below the original 95% gate: none. Diagnostics save stimulus-offset/swap/final-query sensitivity, actual-graph gradients and gate distributions for failing strata, with at most 64 fixed-order trials per stratum and no L2/gate thresholds.

| V true order | r | Visible order | n | Accuracy | CE nat |
|---|---:|---|---:|---:|---:|
| II | 0 | AA | 175 | 98.2857% | 0.0525578 |
| II | 1 | BB | 168 | 100.0000% | 0.0263348 |
| CC | 0 | BB | 174 | 98.8506% | 0.0372273 |
| CC | 1 | AA | 167 | 98.8024% | 0.0658897 |
| IC | 0 | AB | 175 | 95.4286% | 0.130842 |
| IC | 1 | BA | 193 | 97.9275% | 0.100858 |
| CI | 0 | BA | 149 | 96.6443% | 0.0617584 |
| CI | 1 | AB | 165 | 96.3636% | 0.111605 |

Nonzero content sensitivity establishes stimulus influence, not storage of the correctly updated complete content. Correct behavior could also result from history stored in control and computation at final readout; the original interchange analysis remains necessary.


### Stage 7 advancement/termination: fixed T/V two-step diagnostics

| Dataset | Two-step count | Single-query accuracy | Correct full forward mapping | Correct full reverse mapping | Both mappings correct | Forward bijective | Reverse bijective | Query mappings are mutual inverses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T | 2048 | 99.1211% | 93.7012% | 89.6484% | 84.5215% | 93.7012% | 89.6484% | 84.5215% |
| V | 2044 | 99.1194% | 92.3679% | 90.3131% | 84.3444% | 92.3679% | 90.3131% | 84.3444% |

Candidate answers are correct-order two-step, first-only, second-only, no update and reversed order. Every original trial retains all candidate answers/mappings and pairwise overlaps. The table uses only queries where candidate and correct answers differ; empty subsets support no judgment.

| Dataset | True order | Candidate | Distinguishable query count | Prediction agreement |
|---|---|---|---:|---:|
| T | II | first_only | 377 | 0.2653% |
| T | II | second_only | 377 | 0.2653% |
| T | II | none | 0 | Indistinguishable |
| T | II | reversed | 0 | Indistinguishable |
| T | CC | first_only | 457 | 0.2188% |
| T | CC | second_only | 457 | 0.2188% |
| T | CC | none | 0 | Indistinguishable |
| T | CC | reversed | 0 | Indistinguishable |
| T | IC | first_only | 423 | 0.2364% |
| T | IC | second_only | 367 | 0.2725% |
| T | IC | none | 437 | 0.2288% |
| T | IC | reversed | 430 | 0.2326% |
| T | CI | first_only | 385 | 1.5584% |
| T | CI | second_only | 442 | 0.0000% |
| T | CI | none | 448 | 0.4464% |
| T | CI | reversed | 449 | 1.3363% |
| V | II | first_only | 348 | 0.0000% |
| V | II | second_only | 348 | 0.0000% |
| V | II | none | 0 | Indistinguishable |
| V | II | reversed | 0 | Indistinguishable |
| V | CC | first_only | 440 | 0.0000% |
| V | CC | second_only | 440 | 0.0000% |
| V | CC | none | 0 | Indistinguishable |
| V | CC | reversed | 0 | Indistinguishable |
| V | IC | first_only | 438 | 0.0000% |
| V | IC | second_only | 374 | 0.0000% |
| V | IC | none | 431 | 0.0000% |
| V | IC | reversed | 436 | 0.2294% |
| V | CI | first_only | 397 | 1.2594% |
| V | CI | second_only | 449 | 0.2227% |
| V | CI | none | 449 | 0.8909% |
| V | CI | reversed | 450 | 1.1111% |

All-query agreement, agreement on subsets uniquely distinguishing each candidate, and full-candidate-mapping agreement are saved separately in diagnostic summary.json and NPZ. Coincident candidates are not evidence of skipped/reversed operations. Reversing II/CC equals correct execution; IC/CI patterns are described only on predefined distinguishable subsets. Complete-query reconstruction uses existing readouts, without writing truth into hidden states or adding training losses.

Both actual operations and visible words are saved: IC is AB for r=0, BA for r=1; CI reverses this correspondence. Stratified CSVs retain counts, accuracy and CE for actual operations, context, visible order, six slot pairs, d, query type and descriptive crossed cells. Such patterns suggest further tests, not a unique internal mechanism.

T true-operation strata below the original 95% gate: CIC, ICI. Save stimulus-offset/swap/final-query sensitivity, actual-graph gradients and gate distributions for each, selecting at most 64 trials in fixed order, without L2/gate thresholds.
V true-operation strata below the original 95% gate: CCC, CCI, CIC, ICI. Save stimulus-offset/swap/final-query sensitivity, actual-graph gradients and gate distributions for each, selecting at most 64 trials in fixed order, without L2/gate thresholds.

| V true order | r | Visible order | n | Accuracy | CE nat |
|---|---:|---|---:|---:|---:|
| II | 0 | AA | 220 | 100.0000% | 0.00662785 |
| II | 1 | BB | 271 | 99.6310% | 0.030283 |
| CC | 0 | BB | 244 | 99.5902% | 0.013588 |
| CC | 1 | AA | 277 | 99.6390% | 0.0163657 |
| IC | 0 | AB | 251 | 98.8048% | 0.0376635 |
| IC | 1 | BA | 258 | 100.0000% | 0.0127827 |
| CI | 0 | BA | 256 | 98.4375% | 0.0429707 |
| CI | 1 | AB | 267 | 97.0037% | 0.061419 |

Nonzero content sensitivity establishes stimulus influence, not storage of the correctly updated complete content. Correct behavior could also result from history stored in control and computation at final readout; the original interchange analysis remains necessary.

### Descriptive localization of two-step errors

At restoration, V true IC accuracy for r=0/1 is 82.86%/82.38%; true CI is 87.92%/69.09%. Visible AB means different actual operations across contexts, yielding 82.86%/69.09%. Operation and context/cue imbalances coexist; no single mathematical operation alone explains the pattern.

At restoration, two-step V accuracy for d=0/10 is 85.97%/83.91%; slots (1,2) yield 88.16%, and (2,4) 82.46%. Slot×d cells do not uniformly worsen with d, so these tables cannot establish a single retention-decay mechanism. Counts and all crossed cells remain saved.

At restoration, CI first-only agreement is 20.09% (46/229) on answer-distinguishable V queries. Yet no complete bidirectional predicted mapping equals that candidate on the full-mapping-distinguishable subset. This is a query-level pattern, insufficient to establish an internally skipped second operation.

On the same fixed stage 6 V set, two-step complete-bidirectional-mapping accuracy rose from 16.47% to 73.06% at advancement. Additional budget markedly improved two-step learning, but does not prove sequential updates. The final original-task gap concerns three-step operation factors, not forgetting k=0/1.

recovery_diagnostics/ also saves full T/V sensitivity, gradients and gate distributions for the first 64 fixed rows at all three diagnostic times. Weights, complete Adam state/parameter groups and all RNG were verified exactly again; checkpoints were neither optimized nor rewritten.

## Scientific analysis not started in this version

Failed pilot stage: 7. Reaching the cap means only that thresholds were not met within this budget, not that the default model cannot learn the function. G/B were not computed and formal 101–106 were not run; H1/H2/H3 remain untested. Complete 500-step curves, operation factors, latest/best weights, Adam, RNG and raw diagnostics are retained.

## Cost, reproduction and interpretation boundaries

Pre-continuation throughput measured 141.41 updates/second; all temporary weights/Adam/RNG from 200 benchmark steps were discarded and restored. Cumulative conservative accounting for training, validation, saving, caching and GPU analysis is 1080.079 seconds (0.3000 hours), including 605.515 prior seconds, against a 12-hour cap. Scientific CPU-bootstrap wall time is separate, not GPU time.

Command: `python -m experiments.memory_interchange.v1_2.run`. It resumes from completed summaries/checkpoints, verifies historical versions and frozen source, and stops by registered budgets/gates. Tests are in results/v1_2/tests.txt; independent restoration replay and actual source hash are in restore_consistency.json.

Configuration: revision_v1_2.yaml; changes: CHANGES.md. Datasets reuse results/v1_1/datasets without new V generation. Original 8,192-trial V and split hashes are frozen. Curves include stage 6's original 5,000 updates; every evaluation saves factor n, accuracy and CE.

m/c come from a designated architecture, not naturally independent memory brain regions. Passing behavior does not prove sequential reusable updating; interchange and later behavior still need testing. Mathematical properties do not establish learned group structure or biological mechanisms. Related work/originality was not reassessed; update it before submission.

**Final status: BLOCKED.**
