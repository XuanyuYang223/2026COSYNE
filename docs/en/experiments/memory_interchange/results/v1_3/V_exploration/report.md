> English edition · [Frozen original](../../../../../../../experiments/memory_interchange/results/v1_3/V_exploration/report.md) · [English index](../../../../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

# v1.3 V exploration report: EXPLORATORY

V is model-selection data, not formal held-out confirmation. The original v1.2 BLOCKED report remains unchanged; seed0 received no additional training.

Locked best stage step 49500, cumulative course 78000, original V 98.6084%. Old fine-factor gate passed: False; new coarse gate and preceding consecutive evaluation passed: True.

Checkpoint SHA: `da26c47754975bac62579b477eec95fcec5599fdbaa786687463a5ca6560df02`; weights SHA: `2925f0fb0ac74e0bae509a2b804450781b76b1f5d079f04ee863169a2c1375cf`; Adam SHA: `9d6df2f741667cb7f1dd96fea1212bee0ec2c0ae4efd9907a149892324368a6c`.

Geometry V: original first 52 orbits/392 contents. Behavioral V: original latter-half 6 orbits/34 contents. Membership was compared itemwise with original manifests, without output-based reselection.

| Treatment | Mean accuracy across all queries | All-query CE (nat) |
|---|---:|---:|
| Native | 97.9420% | 0.0582248 |
| sham | 97.9420% | 0.0582248 |
| Matched | 67.9051% | 5.08034 |
| Wrong content | 17.6837% | 21.1415 |

| mask | Retained fraction | Matched accuracy | Wrong-content accuracy | Matched L (nat) | Wrong-content L (nat) | Matched R | Wrong-content R | Repair of native errors | Wrong-content counterfactual agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 100.0000% | 67.9051% | 17.6837% | 5.05122 | 21.0912 | 31.0851% | 80.5391% | 1.0482% | 67.9051% |
| rec | 97.9420% | 68.0452% | 17.7693% | 5.03458 | 21.1016 | 31.9548% | 82.2307% | 0.0000% | 67.9201% |
| both | 95.9974% | 68.7218% | 17.8648% | 5.08487 | 21.1715 | 31.2782% | 82.1352% | 0.0000% | 67.9671% |
| three | 94.1326% | 68.7225% | 17.9810% | 5.11153 | 21.2895 | 31.2775% | 82.0190% | 0.0000% | 68.6846% |

Worst native condition×S×d×query-type accuracy is 84.8739%. Counts, CE, all actual operations and native-correctness quality strata are in native_strata.csv; overall means do not conceal poor strata.

rec uses queries originally correct for the receiver. both additionally requires the matched donor to answer the same S/query correctly under its own context. three additionally requires the wrong-content donor to be correct. Donor answers need not equal receiver answers. Matched/wrong-content treatments share one pre-frozen mask. Repair rates use that mode's query denominator and are therefore zero for rec/both. Every sample remains reported; shared-state/orbit queries are not independent statistical units.

Registered GPU continuation/sham/full-state controls passed. Strict CPU float32 logit tolerance passed: False; maximum difference 1.33514e-05, changed answers 0. With identical coefficients converted to float64, maximum difference under original 2e-7/2e-6 tolerances is 3.55271e-14. Actual interventions remain GPU float32; thresholds were not relaxed.

| Probe target | Per-item accuracy (equal conditions/delays) | Full-mapping accuracy |
|---|---:|---:|
| Current q | 80.5788% | 32.1854% |
| Initial p_a | 99.6882% | 98.0230% |

Both probes fit exactly the same T natural states, centering and alpha=1. Easier linear decoding of initial content shows readability differences, not which information the network uses.

Initial-input counterfactual is F_b(G_b(p_a)); current-content target is F_b(q). Raw NPZ saves full mappings, per-query answers and overlaps. directions.csv reports all/distinguishable subsets; mapping_diagnostics.csv reports complete bidirectional mapping agreement. Initial-counterfactual agreement is consistent with initial/history-dependent encoding, but proves neither exclusively delayed computation nor absence of history-dependent coordinates.

| Context relation | mask | Distinguishable query count | Current-target agreement | Initial-counterfactual agreement | Wrong-content counterfactual (distinct answers) |
|---|---|---:|---:|---:|---:|
| Same | all | 264960 | 97.5891% | 0.8764% | 97.8741% |
| Same | rec | 258520 | 97.6760% | 0.8576% | 97.8841% |
| Same | both | 252512 | 100.0000% | 0.0000% | 97.9625% |
| Same | three | 247234 | 100.0000% | 0.0000% | 100.0000% |
| Cross | all | 264960 | 36.2530% | 29.1478% | 40.7029% |
| Cross | rec | 258520 | 36.7894% | 29.1099% | 41.1397% |
| Cross | both | 252407 | 36.6915% | 29.3669% | 41.5634% |
| Cross | three | 247113 | 37.0923% | 29.2728% | 41.5308% |

| Donor r→receiver r | all L nat | All new errors | Signed CE change (nat) |
|---|---:|---:|---:|
| 0→0 | 0.0445843 | 1.5202% | -1.05549e-07 |
| 0→1 | 10.3132 | 56.8744% | 10.3027 |
| 1→0 | 8.67695 | 57.2021% | 8.66973 |
| 1→1 | 0.0591835 | 2.2555% | 1.10791e-08 |

All tables are descriptive summaries of prescribed pairs/directions. Complete d/S/H/query-type strata and counts are retained, without treating shared-state queries as independent observations.

These results describe interventions and check definitions. They supply no single-model large-sample p-values or cross-seed generalization claim. V outcomes select neither metrics/models nor whether formal execution proceeds. The formal primary label remains L_all, the main baseline controls D_input, and rec/both sensitivity analyses remain.

![CKA](../../../../../../../experiments/memory_interchange/results/v1_3/V_exploration/figures/cka_all_damage.png)

![Native-coordinate distance](../../../../../../../experiments/memory_interchange/results/v1_3/V_exploration/figures/raw_all_damage.png)

![Procrustes](../../../../../../../experiments/memory_interchange/results/v1_3/V_exploration/figures/proc_all_damage.png)

![Behavior and coverage](../../../../../../../experiments/memory_interchange/results/v1_3/V_exploration/figures/behavior_and_coverage.png)
