> English edition · [Frozen original](../../../../../../experiments/memory_interchange/results/v1_3/report.md) · [English index](../../../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

# v1.3 execution report: COMPLETED

Date: 2026-10-12 (America/Los_Angeles).

The v1.2 BLOCKED report is unchanged. v1.3 explicitly revises stage 7 advancement; original fine operation strata remain itemized, and overall accuracy does not replace conditional behavior. seed0 remains fixed at best 49,500, without continued training or replacement.

V exploration is separate in the [V exploration report](V_exploration/report.md). Scientific metrics and training/analysis rules were frozen before formal G/B opening; exploration does not determine whether formal execution proceeds.

Cumulative conservative GPU accounting: 3313.147 seconds (0.9203 hours), inheriting 1080.079 prior seconds, with a 12-hour cap. CPU-bootstrap wall time is separate.

## Six-seed original-task and fine-stratum status

| Seed | Course updates | New curriculum passed | Selected original V | New coarse factors passed | Old fine factors passed | Worst B condition stratum |
|---:|---:|---|---:|---|---|---:|
| 101 | 37000 | True | 98.5718% | True | False | 82.4094% |
| 102 | 61000 | True | 98.5229% | True | False | 90.6183% |
| 103 | 40000 | True | 98.3521% | True | False | 87.1002% |
| 104 | 47500 | True | 98.3643% | True | False | 86.4606% |
| 105 | 49500 | True | 98.6084% | True | False | 85.6077% |
| 106 | 48500 | True | 98.5474% | True | False | 60.6610% |

Every seed starts from a new initialization, never pilot fine-tuning. Failed seeds remain; no replacement or interchange-guided continuation occurs. Actual-operation words, k, r, d and query n/accuracy/CE are saved in selected_validation.json, every 500-step curves and native_strata.csv. Checkpoints include Adam/RNG. Initial CKA uses pre-curriculum weights.

Native B quality is fixed separately by receiver×S×d×query_type and bidirectional label-level mean accuracy: ≥95%, 90–95%, <90%. Tiers are descriptive only and do not filter fitting models or unfavorable test rows.

## Seed 101: interventions and counterfactuals

| Treatment | Mean accuracy across all queries | All-query CE (nat) |
|---|---:|---:|
| Native | 98.2923% | 0.0502797 |
| sham | 98.2923% | 0.0502797 |
| Matched | 70.2769% | 4.16661 |
| Wrong content | 19.0626% | 15.7444 |

| mask | Retained fraction | Matched accuracy | Wrong-content accuracy | Matched L (nat) | Wrong-content L (nat) | Matched R | Wrong-content R | Repair of native errors | Wrong-content counterfactual agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 100.0000% | 70.2769% | 19.0626% | 4.14313 | 15.7019 | 28.9503% | 79.5017% | 0.9349% | 70.2769% |
| rec | 98.2923% | 70.3619% | 19.1147% | 4.1275 | 15.6827 | 29.6381% | 80.8853% | 0.0000% | 70.2508% |
| both | 96.6499% | 71.0823% | 19.1579% | 4.15638 | 15.7323 | 28.9177% | 80.8421% | 0.0000% | 70.2681% |
| three | 95.1173% | 71.0952% | 19.1789% | 4.17383 | 15.814 | 28.9048% | 80.8211% | 0.0000% | 70.9884% |

Worst native condition×S×d×query-type accuracy is 82.4094%. Counts, CE, all actual operations and native-correctness quality strata are in native_strata.csv; overall means do not conceal poor strata.

rec uses queries originally correct for the receiver. both additionally requires the matched donor to answer the same S/query correctly under its own context. three additionally requires the wrong-content donor to be correct. Donor answers need not equal receiver answers. Matched/wrong-content treatments share one pre-frozen mask. Repair rates use that mode's query denominator and are therefore zero for rec/both. Every sample remains reported; shared-state/orbit queries are not independent statistical units.

Registered GPU continuation/sham/full-state controls passed. Strict CPU float32 logit tolerance passed: False; maximum difference 1.52588e-05, changed answers 0. With identical coefficients converted to float64, maximum difference under original 2e-7/2e-6 tolerances is 2.13163e-14. Actual interventions remain GPU float32; thresholds were not relaxed.

| Probe target | Per-item accuracy (equal conditions/delays) | Full-mapping accuracy |
|---|---:|---:|
| Current q | 71.6504% | 20.2292% |
| Initial p_a | 100.0000% | 100.0000% |

Both probes fit exactly the same T natural states, centering and alpha=1. Easier linear decoding of initial content shows readability differences, not which information the network uses.

Initial-input counterfactual is F_b(G_b(p_a)); current-content target is F_b(q). Raw NPZ saves full mappings, per-query answers and overlaps. directions.csv reports all/distinguishable subsets; mapping_diagnostics.csv reports complete bidirectional mapping agreement. Initial-counterfactual agreement is consistent with initial/history-dependent encoding, but proves neither exclusively delayed computation nor absence of history-dependent coordinates.

| Context relation | mask | Distinguishable query count | Current-target agreement | Initial-counterfactual agreement | Wrong-content counterfactual (distinct answers) |
|---|---|---:|---:|---:|---:|
| Same | all | 1069056 | 97.9984% | 0.7198% | 98.2493% |
| Same | rec | 1047694 | 98.0288% | 0.7117% | 98.2466% |
| Same | both | 1027042 | 100.0000% | 0.0000% | 98.3413% |
| Same | three | 1008881 | 100.0000% | 0.0000% | 100.0000% |
| Cross | all | 1069056 | 42.7823% | 26.1131% | 46.1510% |
| Cross | rec | 1047694 | 43.2518% | 26.1405% | 46.4799% |
| Cross | both | 1027273 | 43.4958% | 26.1302% | 46.7637% |
| Cross | three | 1008978 | 43.8398% | 26.1202% | 46.9827% |

| Donor r→receiver r | all L nat | All new errors | Signed CE change (nat) |
|---|---:|---:|---:|
| 0→0 | 0.0450499 | 1.5486% | 1.93508e-10 |
| 0→1 | 7.9891 | 53.2434% | 7.97902 |
| 1→0 | 7.58231 | 53.2627% | 7.57156 |
| 1→1 | 0.0454311 | 1.6710% | -1.67152e-09 |

All tables are descriptive summaries of prescribed pairs/directions. Complete d/S/H/query-type strata and counts are retained, without treating shared-state queries as independent observations.

## Seed 102: interventions and counterfactuals

| Treatment | Mean accuracy across all queries | All-query CE (nat) |
|---|---:|---:|
| Native | 98.3653% | 0.0465358 |
| sham | 98.3653% | 0.0465358 |
| Matched | 70.2995% | 4.63657 |
| Wrong content | 19.0227% | 16.7772 |

| mask | Retained fraction | Matched accuracy | Wrong-content accuracy | Matched L (nat) | Wrong-content L (nat) | Matched R | Wrong-content R | Repair of native errors | Wrong-content counterfactual agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 100.0000% | 70.2995% | 19.0227% | 4.61485 | 16.7384 | 28.9699% | 79.6179% | 0.9041% | 70.2995% |
| rec | 98.3653% | 70.3882% | 19.0570% | 4.61444 | 16.7391 | 29.6118% | 80.9430% | 0.0000% | 70.2739% |
| both | 96.8156% | 71.0375% | 19.0859% | 4.6511 | 16.7648 | 28.9625% | 80.9141% | 0.0000% | 70.2645% |
| three | 95.3104% | 71.0054% | 19.1300% | 4.66876 | 16.8482 | 28.9946% | 80.8700% | 0.0000% | 70.9460% |

Worst native condition×S×d×query-type accuracy is 90.6183%. Counts, CE, all actual operations and native-correctness quality strata are in native_strata.csv; overall means do not conceal poor strata.

rec uses queries originally correct for the receiver. both additionally requires the matched donor to answer the same S/query correctly under its own context. three additionally requires the wrong-content donor to be correct. Donor answers need not equal receiver answers. Matched/wrong-content treatments share one pre-frozen mask. Repair rates use that mode's query denominator and are therefore zero for rec/both. Every sample remains reported; shared-state/orbit queries are not independent statistical units.

Registered GPU continuation/sham/full-state controls passed. Strict CPU float32 logit tolerance passed: False; maximum difference 1.62125e-05, changed answers 0. With identical coefficients converted to float64, maximum difference under original 2e-7/2e-6 tolerances is 4.79616e-14. Actual interventions remain GPU float32; thresholds were not relaxed.

| Probe target | Per-item accuracy (equal conditions/delays) | Full-mapping accuracy |
|---|---:|---:|
| Current q | 90.8399% | 58.0906% |
| Initial p_a | 94.1517% | 78.9295% |

Both probes fit exactly the same T natural states, centering and alpha=1. Easier linear decoding of initial content shows readability differences, not which information the network uses.

Initial-input counterfactual is F_b(G_b(p_a)); current-content target is F_b(q). Raw NPZ saves full mappings, per-query answers and overlaps. directions.csv reports all/distinguishable subsets; mapping_diagnostics.csv reports complete bidirectional mapping agreement. Initial-counterfactual agreement is consistent with initial/history-dependent encoding, but proves neither exclusively delayed computation nor absence of history-dependent coordinates.

| Context relation | mask | Distinguishable query count | Current-target agreement | Initial-counterfactual agreement | Wrong-content counterfactual (distinct answers) |
|---|---|---:|---:|---:|---:|
| Same | all | 1069056 | 98.1827% | 0.6979% | 98.3959% |
| Same | rec | 1049848 | 98.2638% | 0.6637% | 98.3968% |
| Same | both | 1031621 | 100.0000% | 0.0000% | 98.4380% |
| Same | three | 1015208 | 100.0000% | 0.0000% | 100.0000% |
| Cross | all | 1069056 | 42.7733% | 25.9362% | 46.1589% |
| Cross | rec | 1049848 | 43.1702% | 25.8481% | 46.4111% |
| Cross | both | 1031620 | 43.3479% | 25.8863% | 46.6670% |
| Cross | three | 1015485 | 43.6095% | 25.8025% | 46.8286% |

| Donor r→receiver r | all L nat | All new errors | Signed CE change (nat) |
|---|---:|---:|---:|
| 0→0 | 0.0393738 | 1.4079% | 5.609e-08 |
| 0→1 | 9.96848 | 53.3449% | 9.95644 |
| 1→0 | 7.39308 | 53.4369% | 7.38367 |
| 1→1 | 0.0419784 | 1.5848% | 1.17432e-07 |

All tables are descriptive summaries of prescribed pairs/directions. Complete d/S/H/query-type strata and counts are retained, without treating shared-state queries as independent observations.

## Seed 103: interventions and counterfactuals

| Treatment | Mean accuracy across all queries | All-query CE (nat) |
|---|---:|---:|
| Native | 98.1822% | 0.0531965 |
| sham | 98.1822% | 0.0531965 |
| Matched | 70.1642% | 3.84737 |
| Wrong content | 19.0847% | 14.2131 |

| mask | Retained fraction | Matched accuracy | Wrong-content accuracy | Matched L (nat) | Wrong-content L (nat) | Matched R | Wrong-content R | Repair of native errors | Wrong-content counterfactual agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 100.0000% | 70.1642% | 19.0847% | 3.8225 | 14.1681 | 29.0270% | 79.3925% | 1.0090% | 70.1642% |
| rec | 98.1822% | 70.2450% | 19.1362% | 3.81202 | 14.1599 | 29.7550% | 80.8638% | 0.0000% | 70.1646% |
| both | 96.4501% | 71.0355% | 19.1640% | 3.8405 | 14.2031 | 28.9645% | 80.8360% | 0.0000% | 70.1973% |
| three | 94.8081% | 71.0183% | 19.1614% | 3.86063 | 14.2876 | 28.9817% | 80.8386% | 0.0000% | 71.0090% |

Worst native condition×S×d×query-type accuracy is 87.1002%. Counts, CE, all actual operations and native-correctness quality strata are in native_strata.csv; overall means do not conceal poor strata.

rec uses queries originally correct for the receiver. both additionally requires the matched donor to answer the same S/query correctly under its own context. three additionally requires the wrong-content donor to be correct. Donor answers need not equal receiver answers. Matched/wrong-content treatments share one pre-frozen mask. Repair rates use that mode's query denominator and are therefore zero for rec/both. Every sample remains reported; shared-state/orbit queries are not independent statistical units.

Registered GPU continuation/sham/full-state controls passed. Strict CPU float32 logit tolerance passed: False; maximum difference 1.38283e-05, changed answers 0. With identical coefficients converted to float64, maximum difference under original 2e-7/2e-6 tolerances is 2.84217e-14. Actual interventions remain GPU float32; thresholds were not relaxed.

| Probe target | Per-item accuracy (equal conditions/delays) | Full-mapping accuracy |
|---|---:|---:|
| Current q | 71.3536% | 18.0838% |
| Initial p_a | 100.0000% | 100.0000% |

Both probes fit exactly the same T natural states, centering and alpha=1. Easier linear decoding of initial content shows readability differences, not which information the network uses.

Initial-input counterfactual is F_b(G_b(p_a)); current-content target is F_b(q). Raw NPZ saves full mappings, per-query answers and overlaps. directions.csv reports all/distinguishable subsets; mapping_diagnostics.csv reports complete bidirectional mapping agreement. Initial-counterfactual agreement is consistent with initial/history-dependent encoding, but proves neither exclusively delayed computation nor absence of history-dependent coordinates.

| Context relation | mask | Distinguishable query count | Current-target agreement | Initial-counterfactual agreement | Wrong-content counterfactual (distinct answers) |
|---|---|---:|---:|---:|---:|
| Same | all | 1069056 | 97.9634% | 0.7957% | 98.1469% |
| Same | rec | 1047076 | 98.0147% | 0.7691% | 98.1520% |
| Same | both | 1026288 | 100.0000% | 0.0000% | 98.2342% |
| Same | three | 1007276 | 100.0000% | 0.0000% | 100.0000% |
| Cross | all | 1069056 | 42.7064% | 26.1003% | 45.9958% |
| Cross | rec | 1047076 | 43.1700% | 26.0365% | 46.3631% |
| Cross | both | 1026011 | 43.4722% | 25.9546% | 46.7194% |
| Cross | three | 1006951 | 43.7726% | 25.9399% | 47.0620% |

| Donor r→receiver r | all L nat | All new errors | Signed CE change (nat) |
|---|---:|---:|---:|
| 0→0 | 0.0466694 | 1.6732% | -5.08123e-09 |
| 0→1 | 7.30042 | 53.3072% | 7.28751 |
| 1→0 | 7.05702 | 53.3509% | 7.04602 |
| 1→1 | 0.0468182 | 1.7011% | -3.38849e-10 |

All tables are descriptive summaries of prescribed pairs/directions. Complete d/S/H/query-type strata and counts are retained, without treating shared-state queries as independent observations.

## Seed 104: interventions and counterfactuals

| Treatment | Mean accuracy across all queries | All-query CE (nat) |
|---|---:|---:|
| Native | 97.8614% | 0.061496 |
| sham | 97.8614% | 0.061496 |
| Matched | 69.8024% | 4.31318 |
| Wrong content | 19.0010% | 14.6925 |

| mask | Retained fraction | Matched accuracy | Wrong-content accuracy | Matched L (nat) | Wrong-content L (nat) | Matched R | Wrong-content R | Repair of native errors | Wrong-content counterfactual agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 100.0000% | 69.8024% | 19.0010% | 4.28607 | 14.6425 | 29.2964% | 79.2635% | 1.2374% | 69.8024% |
| rec | 97.8614% | 69.8561% | 19.0001% | 4.28946 | 14.6668 | 30.1439% | 80.9999% | 0.0000% | 69.8077% |
| both | 95.8608% | 70.9195% | 19.0040% | 4.32005 | 14.7109 | 29.0805% | 80.9960% | 0.0000% | 69.8615% |
| three | 93.9140% | 70.9455% | 19.0432% | 4.33468 | 14.8215 | 29.0545% | 80.9568% | 0.0000% | 70.9698% |

Worst native condition×S×d×query-type accuracy is 86.4606%. Counts, CE, all actual operations and native-correctness quality strata are in native_strata.csv; overall means do not conceal poor strata.

rec uses queries originally correct for the receiver. both additionally requires the matched donor to answer the same S/query correctly under its own context. three additionally requires the wrong-content donor to be correct. Donor answers need not equal receiver answers. Matched/wrong-content treatments share one pre-frozen mask. Repair rates use that mode's query denominator and are therefore zero for rec/both. Every sample remains reported; shared-state/orbit queries are not independent statistical units.

Registered GPU continuation/sham/full-state controls passed. Strict CPU float32 logit tolerance passed: False; maximum difference 2.09808e-05, changed answers 0. With identical coefficients converted to float64, maximum difference under original 2e-7/2e-6 tolerances is 3.01981e-14. Actual interventions remain GPU float32; thresholds were not relaxed.

| Probe target | Per-item accuracy (equal conditions/delays) | Full-mapping accuracy |
|---|---:|---:|
| Current q | 84.2084% | 41.8643% |
| Initial p_a | 99.0216% | 94.1396% |

Both probes fit exactly the same T natural states, centering and alpha=1. Easier linear decoding of initial content shows readability differences, not which information the network uses.

Initial-input counterfactual is F_b(G_b(p_a)); current-content target is F_b(q). Raw NPZ saves full mappings, per-query answers and overlaps. directions.csv reports all/distinguishable subsets; mapping_diagnostics.csv reports complete bidirectional mapping agreement. Initial-counterfactual agreement is consistent with initial/history-dependent encoding, but proves neither exclusively delayed computation nor absence of history-dependent coordinates.

| Context relation | mask | Distinguishable query count | Current-target agreement | Initial-counterfactual agreement | Wrong-content counterfactual (distinct answers) |
|---|---|---:|---:|---:|---:|
| Same | all | 1069056 | 97.5915% | 0.9254% | 97.9078% |
| Same | rec | 1043770 | 97.7013% | 0.8676% | 97.9129% |
| Same | both | 1019777 | 100.0000% | 0.0000% | 97.9811% |
| Same | three | 997421 | 100.0000% | 0.0000% | 100.0000% |
| Cross | all | 1069056 | 42.2785% | 26.2180% | 45.6347% |
| Cross | rec | 1043770 | 42.6483% | 26.2028% | 46.0182% |
| Cross | both | 1019678 | 43.2704% | 26.0780% | 46.4281% |
| Cross | three | 997307 | 43.6831% | 25.9763% | 47.0512% |

| Donor r→receiver r | all L nat | All new errors | Signed CE change (nat) |
|---|---:|---:|---:|
| 0→0 | 0.0473828 | 1.6717% | -2.62803e-09 |
| 0→1 | 8.67475 | 53.3648% | 8.65482 |
| 1→0 | 7.42174 | 53.8633% | 7.40712 |
| 1→1 | 0.0598636 | 2.2065% | -8.09596e-09 |

All tables are descriptive summaries of prescribed pairs/directions. Complete d/S/H/query-type strata and counts are retained, without treating shared-state queries as independent observations.

## Seed 105: interventions and counterfactuals

| Treatment | Mean accuracy across all queries | All-query CE (nat) |
|---|---:|---:|
| Native | 97.9636% | 0.0582796 |
| sham | 97.9636% | 0.0582796 |
| Matched | 70.0192% | 4.41392 |
| Wrong content | 19.0841% | 16.3189 |

| mask | Retained fraction | Matched accuracy | Wrong-content accuracy | Matched L (nat) | Wrong-content L (nat) | Matched R | Wrong-content R | Repair of native errors | Wrong-content counterfactual agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 100.0000% | 70.0192% | 19.0841% | 4.38811 | 16.2712 | 29.1056% | 79.2464% | 1.1613% | 70.0192% |
| rec | 97.9636% | 70.0976% | 19.1039% | 4.3802 | 16.2834 | 29.9024% | 80.8961% | 0.0000% | 70.0317% |
| both | 96.0339% | 71.0054% | 19.1341% | 4.41455 | 16.3366 | 28.9946% | 80.8659% | 0.0000% | 70.0728% |
| three | 94.1768% | 71.0059% | 19.1223% | 4.43716 | 16.4416 | 28.9941% | 80.8777% | 0.0000% | 71.0079% |

Worst native condition×S×d×query-type accuracy is 85.6077%. Counts, CE, all actual operations and native-correctness quality strata are in native_strata.csv; overall means do not conceal poor strata.

rec uses queries originally correct for the receiver. both additionally requires the matched donor to answer the same S/query correctly under its own context. three additionally requires the wrong-content donor to be correct. Donor answers need not equal receiver answers. Matched/wrong-content treatments share one pre-frozen mask. Repair rates use that mode's query denominator and are therefore zero for rec/both. Every sample remains reported; shared-state/orbit queries are not independent statistical units.

Registered GPU continuation/sham/full-state controls passed. Strict CPU float32 logit tolerance passed: False; maximum difference 1.90735e-05, changed answers 0. With identical coefficients converted to float64, maximum difference under original 2e-7/2e-6 tolerances is 4.26326e-14. Actual interventions remain GPU float32; thresholds were not relaxed.

| Probe target | Per-item accuracy (equal conditions/delays) | Full-mapping accuracy |
|---|---:|---:|
| Current q | 76.4437% | 26.8744% |
| Initial p_a | 99.9887% | 99.9210% |

Both probes fit exactly the same T natural states, centering and alpha=1. Easier linear decoding of initial content shows readability differences, not which information the network uses.

Initial-input counterfactual is F_b(G_b(p_a)); current-content target is F_b(q). Raw NPZ saves full mappings, per-query answers and overlaps. directions.csv reports all/distinguishable subsets; mapping_diagnostics.csv reports complete bidirectional mapping agreement. Initial-counterfactual agreement is consistent with initial/history-dependent encoding, but proves neither exclusively delayed computation nor absence of history-dependent coordinates.

| Context relation | mask | Distinguishable query count | Current-target agreement | Initial-counterfactual agreement | Wrong-content counterfactual (distinct answers) |
|---|---|---:|---:|---:|---:|
| Same | all | 1069056 | 97.7466% | 0.8898% | 97.9703% |
| Same | rec | 1044701 | 97.8113% | 0.8598% | 97.9806% |
| Same | both | 1021836 | 100.0000% | 0.0000% | 98.0338% |
| Same | three | 1000177 | 100.0000% | 0.0000% | 100.0000% |
| Cross | all | 1069056 | 42.6141% | 26.0778% | 45.9051% |
| Cross | rec | 1044701 | 43.0183% | 26.0207% | 46.2758% |
| Cross | both | 1021552 | 43.3716% | 25.9688% | 46.6249% |
| Cross | three | 999806 | 43.7029% | 25.9411% | 46.9856% |

| Donor r→receiver r | all L nat | All new errors | Signed CE change (nat) |
|---|---:|---:|---:|
| 0→0 | 0.0497989 | 1.7593% | -1.28211e-09 |
| 0→1 | 8.96219 | 53.2517% | 8.94695 |
| 1→0 | 7.52328 | 53.3714% | 7.50769 |
| 1→1 | 0.0535188 | 1.9886% | 2.71876e-10 |

All tables are descriptive summaries of prescribed pairs/directions. Complete d/S/H/query-type strata and counts are retained, without treating shared-state queries as independent observations.

## Seed 106: interventions and counterfactuals

| Treatment | Mean accuracy across all queries | All-query CE (nat) |
|---|---:|---:|
| Native | 98.3451% | 0.0464408 |
| sham | 98.3451% | 0.0464408 |
| Matched | 70.2208% | 5.32789 |
| Wrong content | 19.0863% | 19.2076 |

| mask | Retained fraction | Matched accuracy | Wrong-content accuracy | Matched L (nat) | Wrong-content L (nat) | Matched R | Wrong-content R | Repair of native errors | Wrong-content counterfactual agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 100.0000% | 70.2208% | 19.0863% | 5.30713 | 19.1695 | 29.0592% | 79.5603% | 0.9350% | 70.2208% |
| rec | 98.3451% | 70.2414% | 19.1011% | 5.27735 | 19.1474 | 29.7586% | 80.8989% | 0.0000% | 70.1964% |
| both | 96.7475% | 71.0302% | 19.1123% | 5.32156 | 19.2577 | 28.9698% | 80.8877% | 0.0000% | 70.2809% |
| three | 95.4541% | 71.0482% | 19.0861% | 5.35776 | 19.3892 | 28.9518% | 80.9139% | 0.0000% | 71.0078% |

Worst native condition×S×d×query-type accuracy is 60.6610%. Counts, CE, all actual operations and native-correctness quality strata are in native_strata.csv; overall means do not conceal poor strata.

rec uses queries originally correct for the receiver. both additionally requires the matched donor to answer the same S/query correctly under its own context. three additionally requires the wrong-content donor to be correct. Donor answers need not equal receiver answers. Matched/wrong-content treatments share one pre-frozen mask. Repair rates use that mode's query denominator and are therefore zero for rec/both. Every sample remains reported; shared-state/orbit queries are not independent statistical units.

Registered GPU continuation/sham/full-state controls passed. Strict CPU float32 logit tolerance passed: False; maximum difference 1.52588e-05, changed answers 0. With identical coefficients converted to float64, maximum difference under original 2e-7/2e-6 tolerances is 2.84217e-14. Actual interventions remain GPU float32; thresholds were not relaxed.

| Probe target | Per-item accuracy (equal conditions/delays) | Full-mapping accuracy |
|---|---:|---:|
| Current q | 67.9031% | 29.8385% |
| Initial p_a | 97.3287% | 88.1888% |

Both probes fit exactly the same T natural states, centering and alpha=1. Easier linear decoding of initial content shows readability differences, not which information the network uses.

Initial-input counterfactual is F_b(G_b(p_a)); current-content target is F_b(q). Raw NPZ saves full mappings, per-query answers and overlaps. directions.csv reports all/distinguishable subsets; mapping_diagnostics.csv reports complete bidirectional mapping agreement. Initial-counterfactual agreement is consistent with initial/history-dependent encoding, but proves neither exclusively delayed computation nor absence of history-dependent coordinates.

| Context relation | mask | Distinguishable query count | Current-target agreement | Initial-counterfactual agreement | Wrong-content counterfactual (distinct answers) |
|---|---|---:|---:|---:|---:|
| Same | all | 1069056 | 98.0844% | 0.7424% | 98.3249% |
| Same | rec | 1048614 | 98.1078% | 0.7338% | 98.3205% |
| Same | both | 1028777 | 99.9995% | 0.0004% | 98.6069% |
| Same | three | 1012960 | 99.9995% | 0.0004% | 99.9995% |
| Cross | all | 1069056 | 42.6167% | 26.1282% | 45.9778% |
| Cross | rec | 1048614 | 43.0115% | 26.1097% | 46.3250% |
| Cross | both | 1028739 | 43.3558% | 26.0086% | 46.5499% |
| Cross | three | 1013014 | 43.6321% | 26.0596% | 46.8631% |

| Donor r→receiver r | all L nat | All new errors | Signed CE change (nat) |
|---|---:|---:|---:|
| 0→0 | 0.0275262 | 0.8440% | 3.79117e-08 |
| 0→1 | 11.1631 | 53.0512% | 11.1489 |
| 1→0 | 8.81014 | 53.9271% | 8.80329 |
| 1→1 | 0.0578956 | 2.3072% | 2.82412e-06 |

All tables are descriptive summaries of prescribed pairs/directions. Complete d/S/H/query-type strata and counts are retained, without treating shared-state queries as independent observations.

## Held-out prediction: primary label L_all

Alpha=1 ridge, LOSO fitted only on the other five seeds' 91 pairs without AB/BA. The primary test is the held-out seed's 32 unseen cross-context pairs×2d×3S. Centering, scaling and missing-value handling are fold-local. D_input uses G initial one-hot input distances only, without B labels.

| Seed | Model | MAE nat | RMSE nat | MAE improvement over M0_v13 | MAE improvement over M0_old |
|---|---|---:|---:|---:|---:|
| 101 | M0_old | 2.46408 | 2.80531 | 0.000553963 | 0 |
| 101 | M0_old_CKA | 2.49986 | 2.84433 | -0.0352316 | -0.0357855 |
| 101 | M0_v13 | 2.46463 | 2.80533 | 0 | -0.000553963 |
| 101 | M_CKA_v13 | 2.51709 | 2.86928 | -0.0524588 | -0.0530127 |
| 101 | M_raw_v13 | 2.47077 | 2.81069 | -0.00614217 | -0.00669614 |
| 101 | M_proc_v13 | 2.50811 | 2.86028 | -0.0434775 | -0.0440315 |
| 101 | M_init_v13 | 2.48598 | 2.82293 | -0.0213523 | -0.0219063 |
| 101 | M_init_CKA_v13 | 2.56804 | 2.92369 | -0.103414 | -0.103968 |
| 102 | M0_old | 2.44636 | 2.79767 | -0.00426848 | 0 |
| 102 | M0_old_CKA | 2.46219 | 2.81862 | -0.0201005 | -0.015832 |
| 102 | M0_v13 | 2.44209 | 2.80027 | 0 | 0.00426848 |
| 102 | M_CKA_v13 | 2.49271 | 2.82313 | -0.0506225 | -0.046354 |
| 102 | M_raw_v13 | 2.43989 | 2.79623 | 0.00219275 | 0.00646123 |
| 102 | M_proc_v13 | 2.47347 | 2.82288 | -0.0313815 | -0.027113 |
| 102 | M_init_v13 | 2.46017 | 2.84052 | -0.0180859 | -0.0138174 |
| 102 | M_init_CKA_v13 | 2.52628 | 2.88157 | -0.0841959 | -0.0799274 |
| 103 | M0_old | 1.94613 | 2.29441 | 0.0064989 | 0 |
| 103 | M0_old_CKA | 2.01708 | 2.35812 | -0.064458 | -0.0709569 |
| 103 | M0_v13 | 1.95263 | 2.30169 | 0 | -0.0064989 |
| 103 | M_CKA_v13 | 2.02569 | 2.36345 | -0.0730675 | -0.0795664 |
| 103 | M_raw_v13 | 1.9626 | 2.31372 | -0.00997799 | -0.0164769 |
| 103 | M_proc_v13 | 1.98986 | 2.32397 | -0.0372377 | -0.0437366 |
| 103 | M_init_v13 | 1.91962 | 2.2577 | 0.0330106 | 0.0265117 |
| 103 | M_init_CKA_v13 | 2.03119 | 2.35188 | -0.0785591 | -0.085058 |
| 104 | M0_old | 2.86437 | 3.13079 | -0.00717623 | 0 |
| 104 | M0_old_CKA | 2.7236 | 3.04413 | 0.133585 | 0.140762 |
| 104 | M0_v13 | 2.85719 | 3.12401 | 0 | 0.00717623 |
| 104 | M_CKA_v13 | 2.66601 | 3.07943 | 0.191183 | 0.198359 |
| 104 | M_raw_v13 | 2.89305 | 3.16301 | -0.0358583 | -0.0286821 |
| 104 | M_proc_v13 | 2.56793 | 3.02938 | 0.289256 | 0.296433 |
| 104 | M_init_v13 | 2.92907 | 3.2184 | -0.0718762 | -0.0647 |
| 104 | M_init_CKA_v13 | 2.74251 | 3.13974 | 0.11468 | 0.121856 |
| 105 | M0_old | 2.43016 | 2.70255 | -0.00508456 | 0 |
| 105 | M0_old_CKA | 2.47305 | 2.74768 | -0.0479767 | -0.0428921 |
| 105 | M0_v13 | 2.42507 | 2.69891 | 0 | 0.00508456 |
| 105 | M_CKA_v13 | 2.52251 | 2.79277 | -0.0974422 | -0.0923576 |
| 105 | M_raw_v13 | 2.43687 | 2.70768 | -0.0118004 | -0.00671584 |
| 105 | M_proc_v13 | 2.51437 | 2.78436 | -0.0892973 | -0.0842127 |
| 105 | M_init_v13 | 2.36362 | 2.68681 | 0.0614528 | 0.0665373 |
| 105 | M_init_CKA_v13 | 2.46146 | 2.78416 | -0.0363932 | -0.0313087 |
| 106 | M0_old | 3.96249 | 4.7522 | 0.0044287 | 0 |
| 106 | M0_old_CKA | 3.97406 | 4.76085 | -0.00714143 | -0.0115701 |
| 106 | M0_v13 | 3.96692 | 4.7632 | 0 | -0.0044287 |
| 106 | M_CKA_v13 | 3.97387 | 4.75101 | -0.00695275 | -0.0113815 |
| 106 | M_raw_v13 | 3.96666 | 4.75083 | 0.000260418 | -0.00416829 |
| 106 | M_proc_v13 | 4.02811 | 4.87343 | -0.061192 | -0.0656207 |
| 106 | M_init_v13 | 3.96635 | 4.75461 | 0.000566407 | -0.0038623 |
| 106 | M_init_CKA_v13 | 3.98476 | 4.74018 | -0.0178427 | -0.0222714 |
| overall | M0_old | 2.68559 | 3.17936 | -0.000841284 | 0 |
| overall | M0_old_CKA | 2.69164 | 3.19055 | -0.00688714 | -0.00604586 |
| overall | M0_v13 | 2.68475 | 3.18174 | 0 | 0.000841284 |
| overall | M_CKA_v13 | 2.69965 | 3.20529 | -0.0148934 | -0.0140521 |
| overall | M_raw_v13 | 2.69497 | 3.18797 | -0.010221 | -0.00937967 |
| overall | M_proc_v13 | 2.68031 | 3.22052 | 0.00444507 | 0.00528636 |
| overall | M_init_v13 | 2.68747 | 3.19682 | -0.00271412 | -0.00187284 |
| overall | M_init_CKA_v13 | 2.71904 | 3.22649 | -0.0342875 | -0.0334462 |

| Analysis mode | Model | MAE-improvement 95% interval (new baseline) | RMSE-improvement 95% interval | MAE interval versus old baseline |
|---|---|---|---|---|
| all | M0_old | [-5.545642665395638, 1.5500332857520098] | [-19.13581296926499, 7.114525716932498] | [0.0, 0.0] |
| all | M0_old_CKA | [-23.63942261144139, 52.98612746229533] | [-101.17367614458752, 205.2315989774218] | [-22.734099369927517, 56.53857281774359] |
| all | M0_v13 | [0.0, 0.0] | [0.0, 0.0] | [-1.550033285752014, 5.545642665395636] |
| all | M_CKA_v13 | [-42.559522955466775, 87.77730863856183] | [-172.43769316081128, 320.18671944185155] | [-42.418882293200255, 89.53431801312125] |
| all | M_raw_v13 | [-14.926981044749056, 38.19852876542731] | [-57.23958334416661, 175.22322525783207] | [-15.664541784959036, 39.868490090055566] |
| all | M_proc_v13 | [-27.33319749586679, 89.20091766057593] | [-109.68162772955998, 323.25271090231814] | [-25.704464728784696, 92.30665386778871] |
| all | M_init_v13 | [-53.77393672266218, 139.9397290937885] | [-233.7348476996086, 520.7547906799223] | [-52.35807292584954, 144.93329854878246] |
| all | M_init_CKA_v13 | [-28.13767352744388, 117.71557239280156] | [-113.79288144837528, 461.5158989386201] | [-24.802497418048848, 118.56239915711177] |
| rec | M0_old | [-5.771737789709698, 1.5235225907023382] | [-19.782319764281766, 6.923202999639124] | [0.0, 0.0] |
| rec | M0_old_CKA | [-24.060406369006497, 54.88833374647412] | [-98.39892995170166, 210.9407037045315] | [-23.051212313393247, 58.37498095813898] |
| rec | M0_v13 | [0.0, 0.0] | [0.0, 0.0] | [-1.5235225907023386, 5.7717377897096975] |
| rec | M_CKA_v13 | [-42.350863883672766, 89.28615570854627] | [-177.27049806153886, 323.5943234505466] | [-44.56627551697699, 91.4963456140892] |
| rec | M_raw_v13 | [-14.87092587050125, 36.53051217028294] | [-57.262218206135195, 176.99985293270927] | [-15.620988531110193, 35.44515526898405] |
| rec | M_proc_v13 | [-27.77121388551691, 92.1659190731142] | [-108.3870747794287, 326.6879050053654] | [-25.96192131182421, 94.43792836942602] |
| rec | M_init_v13 | [-53.79677046964026, 139.07537606801006] | [-236.1919716364726, 522.6050670487729] | [-52.5548262324469, 144.08163797949032] |
| rec | M_init_CKA_v13 | [-26.92544471943545, 119.58279301484959] | [-110.63893330530547, 457.9141817992839] | [-23.557664039637544, 120.60176767434939] |
| both | M0_old | [-5.454126792934138, 1.5147406961015144] | [-18.75158809715838, 6.768916830448745] | [0.0, 0.0] |
| both | M0_old_CKA | [-23.311744057316663, 53.90722859517798] | [-95.20256288337619, 210.75323516862093] | [-23.010299625179098, 58.239011695361405] |
| both | M0_v13 | [0.0, 0.0] | [0.0, 0.0] | [-1.514740696101519, 5.4541267929341375] |
| both | M_CKA_v13 | [-43.0633391370169, 88.96925050798663] | [-175.12534781861802, 323.9251251075262] | [-44.53686986700489, 89.93178547448939] |
| both | M_raw_v13 | [-14.677433290298593, 37.30664922470366] | [-56.543211394854715, 181.3485544264472] | [-15.801329069265051, 36.26560024622669] |
| both | M_proc_v13 | [-27.619330287217643, 91.42057296296262] | [-110.32055311189835, 326.76586748102443] | [-25.302141920911755, 94.3049102504279] |
| both | M_init_v13 | [-53.994800386613086, 138.21003060490082] | [-237.52538314849275, 521.8374432721613] | [-53.258303408978165, 142.83841738650767] |
| both | M_init_CKA_v13 | [-26.83312606150859, 119.93213186960517] | [-111.91546051872585, 460.1935686785052] | [-24.24951713853633, 120.90292830132293] |

All fold/overall MAE/RMSE for rec/both sensitivity labels and three native-quality tiers are in scores.csv, without changing the primary label. Initial-CKA+trained-CKA comparisons and paired-resampling intervals remain. If CKA no longer improves after controlling D_input, initial-input structure has not been excluded. Seeds and B orbits are resampling blocks; each replicate refits. Six seeds represent a finite model set, not millions of independent query experiments.

The fixed quantile rule selected 288 high-CKA/high-new-error labels. Complete distributions and every qualifying case are delivered without story-driven content selection. Spearman supplements only fixed d/S/context categories; degenerate-variance strata are null.

CPU bootstrap/prediction wall time: 91.896 seconds. H1 uses matched/wrong-content and counterfactual behavior; H2/H3 use held-out improvement over the revised baseline, never curriculum or V exploration as substitutes.

![Curriculum](../../../../../../experiments/memory_interchange/results/v1_3/figures/formal_learning.png)

![Prediction errors](../../../../../../experiments/memory_interchange/results/v1_3/figures/held_out_prediction.png)

![CKA and primary damage](../../../../../../experiments/memory_interchange/results/v1_3/figures/cka_all_damage.png)

![Behavior and analysis modes](../../../../../../experiments/memory_interchange/results/v1_3/figures/behavior_and_coverage.png)

Primary-test MAE/RMSE-improvement intervals: [-42.559522955466775, 87.77730863856183] / [-172.43769316081128, 320.18671944185155]. Interpret only within actual behavioral quality and this finite model set; do not extrapolate brain mechanisms or abstract group structure.

All native/sham/matched/wrong-content/full-state outputs, natural truth, shared masks, orbit statistics and directions remain. Correct behavior alone does not prove sequential reusable updates. No publication or push was performed in this experiment.

**Final status: COMPLETED.**

## Interpretation and numerical-stability checks

With equal weights across six models and prescribed condition pairs, native accuracy is 98.1683%, matched 70.1305%, wrong-content 19.0569%. Per-seed matched-minus-wrong differences are [51.2143, 51.2768, 51.0795, 50.8014, 50.9351, 51.1345] percentage points. Together with distinguishable-answer counterfactual behavior, this supports H1's content-related intervention advantage in this experiment, not a unified neural coordinate system for current content.

Mean rec/both coverage is 98.1683%/96.4263%. They describe damage where native behavior is correct, not all states. The primary label remains all-sample L_all.

Primary-test M0_v13 MAE/RMSE is 2.68475/3.18174 nat; adding CKA gives 2.69965/3.20529 nat, improvements (baseline minus enhanced model) -0.0148934/-0.0235474 nat. MAE/RMSE-improvement 95% intervals are [-42.559522955466775, 87.77730863856183]/[-172.43769316081128, 320.18671944185155], both crossing zero. Stable incremental-prediction support for H2/H3 was not obtained.

Original M0_old+CKA improves MAE over original M0 by -0.00604586 nat; this comparison is retained. Procrustes MAE point improvement is 0.00444507 nat, but RMSE improvement -0.0387843 nat; this does not justify selecting it as the new primary metric. Additional MAE-improvement interval for trained CKA beyond initial CKA is [-37.7683292390621, 107.10608578689315], also without stable support.

rec sensitivity-analysis CKA MAE/RMSE point improvements are -0.0138959/-0.0199993 nat. Full intervals, folds, quality tiers and other generalization strata are retained without replacing the main test.
both sensitivity-analysis CKA MAE/RMSE point improvements are -0.0162002/-0.0234727 nat. Full intervals, folds, quality tiers and other generalization strata are retained without replacing the main test.

Wide-interval verification: assigning every orbit block weight 1 reproduces saved point labels within maximum 2.62e-06 nat (float32 reduction rounding). Argmax differences across all formal GPU caches/uninterrupted continuations are 0. No label or mask-denominator error was found.

Replay of saved extreme resampling replicates fits only a few model instances, with nearly constant control distance in the fitting set; held-out 106 lies far outside that covariate range. Frozen fold-local scaling and alpha=1 ridge therefore extrapolate to very large predictions. Prediction estimates are unstable for this finite six-model set. All replicates remain; no post hoc prediction clipping, seed deletion, variance-floor increase or alpha tuning. See bootstrap_integrity_audit.json.

Seed 106's worst B condition is r=1, H=AB, S=NA (true CIC), d=10, position queries: 938 queries, 60.6610% accuracy. At d=0, the same stratum is 60.7676%. Passing coarse factors does not guarantee mastery of this condition. Quality strata and interventions remain; these results do not establish a universal memory-state mechanism.

Retaining receiver c is an architecture-defined component intervention. m may still carry context, history or initial input. Even same-context reuse of identical q does not prove sequential updating into unified q coordinates or learned group structure.

The original unseen four-step extension was not triggered: 106 failed the original ctx×S×d×query-type behavioral-quality target. The extension was not executed or reported as a result. This version's prescribed formal main/sensitivity analyses are complete.

Full display: [all selected-checkpoint T/V factors](all_behavior_factors.md); quality-tier MAE/RMSE, counts and masks in the [quality/mask summary](quality_and_mask_summary.md); [methodological context](method_context.md).


Complete counterfactual display: [all queries, distinguishable queries and full mappings](counterfactual_all_and_distinct.md). Coincident truth candidates do not distinguish accounts; q is not selected by outcomes.

Task and intervention architecture follow the original protocol:

![Identical current content: replace m and retain c](../../../../../../experiments/memory_interchange/results/figures/01_task.png)
