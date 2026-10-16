> English edition · [Frozen original](../../../../../experiments/memory_interchange/Final_audit/report.md) · [English index](../../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

# v1.3 Final audit

Date: 2026-10-12 (America/Los_Angeles). Status: **COMPLETED: the three limited audits are finished and closed**. The original six RNNs were not retrained; original v1.3 point estimates, primary metrics, all bootstrap replicates, 106 and failed numerical checks remain intact. No publication or push was performed in that audit.

## Abstract

1. **The intervention advantage of content matching is retained.** Original equal-weight six-model accuracy was 98.1683% native, 70.1305% matched and 19.0569% wrong-content, a matched-minus-wrong advantage of 51.0736 percentage points. This supports a content-related intervention advantage in this design, not the claim that m contains only pure current content.
2. **Native m interchange depends on rule context.** Original mean new-error rates were approximately 1.697% within context and 53.398% across contexts. Retaining the physical c component does not establish retention of the receiver's functional rule.
3. **The donor-rule account receives behavioral confirmation on the new set.** Donor-rule agreement was 96.6383% on B1 queries with three pairwise-distinct candidates. On independent B_reserved, the frozen primary cross-context accuracy contrast was **+55.7907 percentage points**; same-context translation changed original-target accuracy by -55.7907 percentage points. This confirms compensation of the mixed state's rule interface through external cue translation, not success of the original direct swap.
4. **The original primary CKA test remains negative, with limited statistical precision.** Mean MAE improvement over the six fixed folds was -0.01489342 nat, with 1 positive and 5 negative folds; the conditional empirical interval was [-0.07402, 0.07351]. The original extremely wide refit-bootstrap intervals are retained. Typical/extreme replicates were reproduced, with no implementation error found within the scope of these checks. This does not establish that CKA can never predict function.

## Frozen scope, independent confirmation and dependence

Original B contains 496 contents / 70 orbits; previously opened B1 contains 134 / 19; the exact difference is **362 contents / 51 complete orbits**. Every remaining orbit is used, without selection by B1 outcomes. Frozen membership, hashes and the audit of historical natural-state-cache access are in [B_reserved_manifest.json](../../../../../experiments/memory_interchange/Final_audit/B_reserved_manifest.json), [membership NPZ](../../../../../experiments/memory_interchange/Final_audit/B_reserved_members.npz), [frozen_plan.json](../../../../../experiments/memory_interchange/Final_audit/frozen_plan.json) and the [freeze receipt](../../../../../experiments/memory_interchange/Final_audit/freeze_receipt.json). Preserved historical natural-state caches contain none of these remaining contents, and no record was found of their prior use for rule selection or tuning. This judgment is limited to retained local records; it does not reconstruct unrecorded external access.

All six models use their original formal initializations and selected checkpoints. T is used only for fixed probe fitting, G only for geometry. The new B1 mechanism analysis is **EXPLORATORY**; B_reserved independently confirms the frozen translation contrast. The original primary CKA test is not replaced.

| seed | Original checkpoint (relative to experiment directory) | stage7 step | SHA256 |
| --- | --- | --- | --- |
| 101 | runs/v1_3/seed_101/selected_stage7.pt | 13500 | 7d2d1e3cf60c94e80c1bc254df633d2a5ec7b3a1e771c6016c969f95252081e2 |
| 102 | runs/v1_3/seed_102/selected_stage7.pt | 32500 | 53b5751a8728919042eb0f3660c6036bb6c68fa4a860d94fdf2108eea1fa7109 |
| 103 | runs/v1_3/seed_103/selected_stage7.pt | 16500 | 2f879d7b568faea91b164a3fa4f2367030572407fddf4f7deef3ccb328644cbd |
| 104 | runs/v1_3/seed_104/selected_stage7.pt | 21000 | dcf5e758c955846c917826596ff27da54da3bec740a8f2145d2fbf608ffffb99 |
| 105 | runs/v1_3/seed_105/selected_stage7.pt | 26000 | 19b33dfa65ff86b6f8b1c72680d6f84e0eccd011cde01f5fcc27ef4fe4f0c49b |
| 106 | runs/v1_3/seed_106/selected_stage7.pt | 23000 | 5ad670aff11654c7c82975d5cfa55cb1c4c18d2ce60a814b3a385f9c390642d8 |

Coverage includes all 18 (r,H) conditions, 153 unordered pairs with 306 ordered directions, d=0/10, S=NN/NA/NB, and 7 item plus 7 position queries per content. The primary contrast weights the six seeds and both directions equally, retains the original d/S/query distribution, and includes NN in the main S mixture. Conditional results pool retained queries within the same direction/d/S mask and then weight those cells equally; queries are not treated as independent models or statistical units. Orbit/seed bootstrap resampling retains directions, query types and complete orbits.

Rule additions are limited to the third counterfactual, A/B suffix translation, native-defined E_sem, G mean decomposition, fixed context probes, replay of the original bootstrap and fixed-prediction descriptions. Original learning, thresholds, primary predictors, alpha, variance floor and samples are unchanged. See [revision_final_audit.yaml](../../../../../experiments/memory_interchange/Final_audit/revision_final_audit.yaml) for the full rules.

## Audit one: donor-rule counterfactual and cue translation

For donor a, receiver b and shared current full content q, the three targets are `Q_receiver=F_b(S)(q)`, `Q_initial=F_b(S)(G_b(p_a))` and `Q_donor_rule=F_a(S)(q)`. The third target is distinct from the initial-input account. Wrong-content q′ candidates are `F_b(S)(q′)` and `F_a(S)(q′)`; no initial-input candidate is invented for wrong content. B1 reuses every original matched/wrong log-probability output; initial-input mathematical mappings are generated and verified independently.

Exploratory cross-context B1 results (rates weight seeds equally; counts sum all six seeds and include dependent queries):

| Treatment | Candidate-distinguishable subset | Query count | Receiver rule | Initial-input account | Donor rule |
| --- | --- | --- | --- | --- | --- |
| matched | all | 10940832 | 45.2081% | 35.5153% | 98.1682% |
| matched | receiver_initial_distinct | 6414336 | 42.6286% | 26.0956% | 97.9649% |
| matched | receiver_donor_distinct | 6003072 | 0.9481% | 28.0260% | 97.4698% |
| matched | initial_donor_distinct | 7065984 | 38.8117% | 0.8494% | 97.8599% |
| matched | all_three_distinct | 2015232 | 0.9906% | 1.1680% | 96.6383% |
| wrong | all | 10940832 | 45.2081% | — | 98.1682% |
| wrong | receiver_donor_distinct | 6003072 | 0.9481% | — | 97.4698% |

All same/cross-context results, pairwise-distinguishable subsets, all-three-distinct subsets, query types and direction counts appear in the [overall candidate table](../../../../../experiments/memory_interchange/Final_audit/candidate_overall_summary.csv), [per-seed table](../../../../../experiments/memory_interchange/Final_audit/candidate_seed_summary.csv) and each `B1/seed_*/candidate_agreement.csv`. Full-mapping correctness, bijectivity, mutual inverse readouts and candidate-mapping agreement appear in the [mapping table](../../../../../experiments/memory_interchange/Final_audit/candidate_mapping_seed_summary.csv) and each seed's `mapping_agreement.csv`. Per-trial predictions, all candidates and overlaps are saved in `counterfactual_d*_S*.npz`.

For NN, donor and receiver rules coincide completely; some single-operation queries also coincide, so high all-query agreement cannot identify a mechanism by itself. Wrong-content q′ uses a fixed bijective reordering: at fixed (a,b,d,S), aggregate candidate agreement may exactly equal that for matched q; this is not additional independent evidence. Distinguishable subsets and the new behavioral intervention are more informative. Readability of initial p does not establish that q was never updated.

Translation exchanges A/B and leaves N unchanged. The cross-context identity `F_a(swap(S))(q)=F_b(S)(q)` was independently verified on all 5,040 permutations. Each state is evaluated under native+S, native+swap(S), matched m patch+S, matched m patch+swap(S), and sham. The **original semantic target** `F_b(S)(q)` and **new literal target** `F_b(swap(S))(q)` remain separate.

all/rec/both retain their original definitions. E_sem requires a correct receiver native answer for original S and a correct donor native execution of the same semantic operation: translated S for cross, original S for same. Native outputs and masks for all six models were saved before any new m patches ([all-model receipt](../../../../../experiments/memory_interchange/Final_audit/all_native_masks_frozen.json)). Original visible-S `both` requires each side to answer the same visible S correctly in its own context, which differs from executing the same semantic operation. All interventions share the original mask.

| Dataset | Context | mask | Coverage | Original-S accuracy | Translated-S accuracy on original target | Difference (pp) | CE difference (nat) | New-error difference (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | cross | all | 100.0000% | 45.2081% | 98.1682% | +52.9601 | -8.30775 | -51.6766 |
| B1 | cross | rec | 98.1683% | 45.2791% | 98.2435% | +52.9644 | -8.30163 | -52.9644 |
| B1 | cross | both | 96.3862% | 45.2571% | 98.4464% | +53.1893 | -8.35992 | -53.1893 |
| B1 | cross | E_sem | 96.4471% | 45.5504% | 99.9999% | +54.4495 | -8.32209 | -54.4495 |
| B1 | same | all | 100.0000% | 98.1682% | 45.2081% | -52.9600 | +8.30775 | +51.6966 |
| B1 | same | rec | 98.1683% | 98.2727% | 45.2834% | -52.9893 | +8.30280 | +52.9893 |
| B1 | same | both | 96.4714% | 99.9998% | 45.5525% | -54.4474 | +8.32236 | +54.4474 |
| B1 | same | E_sem | 96.4714% | 99.9998% | 45.5525% | -54.4474 | +8.32236 | +54.4474 |
| B_reserved | cross | all | 100.0000% | 42.0674% | 97.8582% | +55.7907 | -8.74779 | -54.2339 |
| B_reserved | cross | rec | 97.8582% | 42.1283% | 97.9453% | +55.8170 | -8.74363 | -55.8170 |
| B_reserved | cross | both | 95.7822% | 42.1169% | 98.1913% | +56.0744 | -8.80858 | -56.0744 |
| B_reserved | cross | E_sem | 95.8515% | 42.4285% | 99.9999% | +57.5715 | -8.76876 | -57.5715 |
| B_reserved | same | all | 100.0000% | 97.8581% | 42.0674% | -55.7907 | +8.74779 | +54.2530 |
| B_reserved | same | rec | 97.8582% | 97.9740% | 42.1320% | -55.8421 | +8.74586 | +55.8421 |
| B_reserved | same | both | 95.8746% | 99.9999% | 42.4297% | -57.5703 | +8.77030 | +57.5703 |
| B_reserved | same | E_sem | 95.8746% | 99.9999% | 42.4297% | -57.5703 | +8.77030 | +57.5703 |

New errors retain the definition: native is correct for original S and patch is wrong for the original target. rec/E_sem table values are conditional probabilities. CE and new-error differences are translated minus original S; negative values indicate reductions. Coverage is the shared sample fraction determined by native outputs; the two treatments do not select separate trials.

The paired seed/orbit empirical 95% interval for the frozen primary B_reserved all contrast is **[54.2157, 57.1861] percentage points**. This is a block-resampling description for these fixed networks and this dataset, not a guarantee covering retraining or an unknown model population. All mask intervals are in [translation_block_intervals.json](../../../../../experiments/memory_interchange/Final_audit/translation_block_intervals.json).

| seed | cross all(pp) | cross rec | cross both | cross E_sem | Cross-context CE difference | Cross-context new-error difference (pp) | same all(pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | +55.7619 | +55.7811 | +56.1333 | +57.3491 | -8.18071 | -54.2928 | -55.7619 |
| 102 | +55.4734 | +55.4293 | +55.7893 | +57.2161 | -8.97938 | -53.7575 | -55.4734 |
| 103 | +55.6982 | +55.7323 | +55.9838 | +57.5669 | -7.67756 | -54.0704 | -55.6982 |
| 104 | +55.6486 | +55.7116 | +55.9036 | +57.7697 | -8.24347 | -53.9203 | -55.6486 |
| 105 | +55.9283 | +55.9624 | +56.1802 | +57.6921 | -8.92589 | -54.4270 | -55.9283 |
| 106 | +56.2339 | +56.2854 | +56.4562 | +57.8350 | -10.47970 | -54.9353 | -56.2339 |

| S | Patch on original target | Translated patch on original target | Difference (pp) | Original-S donor rule | Donor rule on distinguishable queries | Translated receiver literal rule | Translated donor rule |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NN | 99.2529% | 99.2529% | +0.0000 | 99.2529% | — | 99.2529% | 99.2529% |
| NA | 13.5746% | 97.3523% | +83.7777 | 96.9692% | 96.8651% | 13.3747% | 97.3523% |
| NB | 13.3747% | 96.9692% | +83.5945 | 97.3523% | 97.3174% | 13.5746% | 96.9692% |

Original/translated NN patch log-probabilities are **elementwise identical**; native/sham are also elementwise identical. New B1 translated outputs match the saved corresponding S′ outputs within the original registered GPU tolerance. NA/NB, both d values, both directions and all H/query results are retained without selecting the best. Complete paired accuracy/CE/new-error differences, rule agreement and coverage are in the [all-seed summary](../../../../../experiments/memory_interchange/Final_audit/translation_seed_summary.csv), [overall summary](../../../../../experiments/memory_interchange/Final_audit/translation_overall_summary.csv) and each `translation_d*_S*.csv`. All 14 query readouts and candidates are saved in the corresponding NPZ files; full-mapping descriptions are in [translation_full_mapping_diagnostics.csv](../../../../../experiments/memory_interchange/Final_audit/translation_full_mapping_diagnostics.csv).

B_reserved native quality and the prescribed native-translation control:

| seed | Native original task | Translated native on original target | Translated native on literal target | Worst original-task stratum | Worst condition | n |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 98.0182% | 42.2562% | 98.0182% | 81.9653% | r=0, H=BA, S=NB, d=10, type=1 | 2534 |
| 102 | 97.7122% | 42.2391% | 97.7122% | 88.3978% | r=1, H=AB, S=NA, d=0, type=1 | 2534 |
| 103 | 97.7208% | 42.0226% | 97.7208% | 83.1492% | r=1, H=AB, S=NA, d=10, type=1 | 2534 |
| 104 | 97.5672% | 41.9184% | 97.5672% | 81.1760% | r=0, H=AB, S=NA, d=10, type=1 | 2534 |
| 105 | 97.8902% | 41.9619% | 97.8902% | 85.1223% | r=1, H=AB, S=NA, d=10, type=0 | 2534 |
| 106 | 98.2404% | 42.0063% | 98.2404% | 63.0624% | r=1, H=AB, S=NA, d=10, type=1 | 2534 |

Translated native outputs primarily follow the new literal target while accuracy on the original target decreases; translation is therefore not a general accuracy improvement. 106 and its poor conditions are retained in full. Native H×r×d×S×query-type accuracy and CE are in [native_all_strata.csv](../../../../../experiments/memory_interchange/Final_audit/native_all_strata.csv); native quality summaries are in [native_seed_summary.csv](../../../../../experiments/memory_interchange/Final_audit/native_seed_summary.csv).

Cross-context improvement, same-context disruption of the original target, and high B1 donor-rule agreement on distinguishable queries support the account that **the rule interface changes with the exchanged m**. This does not establish that r exists only in m or that c is useless, and does not exclude history-dependent content formats, initial inputs, joint encoding or a final one-shot computation. The additional treatment includes external semantic translation; it cannot be described as success of the original direct swap.

## Audit two: CKA, native means and context readability

G geometry is deduplicated to (seed,condition_pair,d); repetition of the same value across S is not counted as a new sample. There are 1,836 values.

| Context | d | n | min | q05 | q25 | median | q75 | q95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| same | 0 | 432 | 0.698615 | 0.759707 | 0.857789 | 0.891930 | 0.913618 | 0.998108 | 0.999986 |
| same | 10 | 432 | 0.702060 | 0.759483 | 0.857315 | 0.891830 | 0.913602 | 0.999881 | 0.999993 |
| cross | 0 | 486 | 0.672997 | 0.768475 | 0.856044 | 0.888664 | 0.912799 | 0.974673 | 0.992563 |
| cross | 10 | 486 | 0.676044 | 0.769509 | 0.854386 | 0.888701 | 0.913299 | 0.974789 | 0.992786 |

All 288 high-CKA/high-new-error cases selected by the original fixed rule are listed in [original_fixed_counterexamples.csv](../../../../../experiments/memory_interchange/Final_audit/original_fixed_counterexamples.csv), including actual CKA, original quantile thresholds, new errors, native/pair quality, worst direction/query-type native quality and the mean decomposition. A high quantile does not imply absolute CKA near 1. This is the complete fixed-rule list, not a favorable-looking selected swap.

Only the first eight rows in original saved order are shown below. Cases were not reselected using the new audit; the link contains all 288 rows:

| seed | (a,b) | d | S | Actual CKA | native(pair) | Worst direction/type native accuracy | New errors R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | (7,17) | 0 | NN | 0.971889 | 99.0938% | 98.9339% | 0.6930% |
| 101 | (8,16) | 0 | NN | 0.956281 | 98.3742% | 97.8678% | 1.2527% |
| 101 | (0,9) | 0 | NA | 0.988387 | 99.7868% | 99.5736% | 82.0629% |
| 101 | (3,11) | 0 | NA | 0.961657 | 99.2271% | 98.6141% | 81.6098% |
| 101 | (4,11) | 0 | NA | 0.962153 | 99.2271% | 98.6141% | 81.6098% |
| 101 | (5,9) | 0 | NA | 0.975455 | 99.4936% | 99.1471% | 81.9829% |
| 101 | (6,9) | 0 | NA | 0.965994 | 99.1205% | 97.7612% | 81.6365% |
| 101 | (12,14) | 0 | NA | 0.928670 | 98.2143% | 96.9083% | 1.7324% |

The native-coordinate mean decomposition uses the original D_raw denominator: `D_mean=||mu_a-mu_b||²/den`, `D_center=mean_q||(m_a-mu_a)-(m_b-mu_b)||²/den`. Intervention states are not translated, rotated or standardized. Independent termwise computation verifies `D_raw=D_mean+D_center`, with maximum absolute error 7.77e-16. Full values and denominators are in [geometry_decomposition.csv](../../../../../experiments/memory_interchange/Final_audit/geometry_decomposition.csv).

| Context | d | Median D_raw | Median D_mean | Median D_center | Median D_mean/D_raw |
| --- | --- | --- | --- | --- | --- |
| same | 0 | 1.2409 | 0.832682 | 0.374766 | 0.670441 |
| same | 10 | 1.23457 | 0.837089 | 0.375992 | 0.670513 |
| cross | 0 | 1.33554 | 0.964595 | 0.388655 | 0.723725 |
| cross | 10 | 1.32662 | 0.957122 | 0.386523 | 0.72427 |

Centering removes a translation shared by every q: if `m_b(q)=m_a(q)+v`, centered linear CKA=1, although v may affect interpretation of future A/B cues. An independent synthetic check gives CKA≈1. This is a possible mechanism; real RNN D_center is nonzero, so the networks cannot be described as differing only by a translation. Original D_raw, which retains uncentered information, also did not yield stable improvement in the original held-out test (M_raw MAE approximately 2.69497 nat, M0 2.68475). These decompositions are descriptive and do not replace the primary predictor.

Context probes are fixed linear ridge models with alpha=1, r→±1, threshold 0, fitted only on the original 512 T contents' natural states, balanced over r/H/d and excluding AB/BA as history holdouts. Coordinate means and standard deviations are fitted on T only; only exactly-zero standard deviations are replaced by 1. m gives equal weight to contents within each condition; c does not treat repeated q as independent states: pooled fitting uses 28 (r,H,d) states and single-d fitting 14, with exact unique-vector counts separately saved. Pooled, fit-d0 and fit-d10 variants are all reported, without score-based selection of directions or more complex probes.

Pooled probes on G AB/BA history/content holdouts:

| seed | m d0 | m d10 | c d0 | c d10 | m margin d0 | m margin d10 | c margin d0 | c margin d10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 100.0000% | 100.0000% | 50.0000% | 50.0000% | 0.655663 | 0.646789 | 0 | 0 |
| 102 | 100.0000% | 100.0000% | 50.0000% | 50.0000% | 0.839875 | 0.852983 | -5.18397e-11 | 0 |
| 103 | 75.0000% | 75.0000% | 50.0000% | 50.0000% | 0.515704 | 0.504857 | 0 | 0 |
| 104 | 100.0000% | 100.0000% | 50.0000% | 50.0000% | 0.761336 | 0.763045 | 0 | 0 |
| 105 | 98.7297% | 98.8821% | 50.0000% | 50.0000% | 0.813203 | 0.813379 | 0 | 0 |
| 106 | 86.2805% | 85.9248% | 50.0000% | 50.0000% | 0.659651 | 0.653723 | 2.52056e-11 | 0 |

| seed | Independent c fitting states | Exactly unique c vectors | Minimum raw c L2 difference | Maximum raw c L2 difference | Minimum raw m L2 difference | Maximum raw m L2 difference | Smallest nonzero T std of c |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | 28 | 8 | 0 | 0 | 3.32556 | 6.87996 | 2.735738043515572e-07 |
| 102 | 28 | 21 | 0 | 6.20247e-07 | 4.46476 | 7.64141 | 0.0007883686603691337 |
| 103 | 28 | 9 | 0 | 0 | 4.27202 | 7.25293 | 5.293840577035602e-07 |
| 104 | 28 | 10 | 0 | 0 | 5.12643 | 7.55172 | 3.382160888705043e-05 |
| 105 | 28 | 7 | 0 | 0 | 4.47785 | 6.79545 | 5.902125266736281e-06 |
| 106 | 28 | 15 | 0 | 9.11086e-07 | 3.14669 | 7.63425 | 0.0017138775762968641 |

Complete T/G, same/different-d, nonmixed/AB/BA and per-H margins and accuracies are in [context_probe_results.csv](../../../../../experiments/memory_interchange/Final_audit/context_probe_results.csv); raw L2 contrasts in [context_raw_contrasts.csv](../../../../../experiments/memory_interchange/Final_audit/context_raw_contrasts.csv); unique-state counts and standard deviations in [context_probe_fit_inventory.csv](../../../../../experiments/memory_interchange/Final_audit/context_probe_fit_inventory.csv). Fitted parameters and scores, margins and labels for each natural state are saved in `context_probes/seed_*/`. Standardization can amplify tiny residual signals; margin size is not a direct measure of neural functional contribution.

Readability of r from m and cue translation are consistent with the rule-interface account. Chance performance of this fixed AB/BA c probe does not prove that c contains no information or has no causal role. Decoding r does not establish causal use of r by the network; finding a signal in c would not establish that it controls future rules. Geometry, probes and semantic-translation compensation describe these artificial networks and cannot directly establish real-brain mechanisms. Correct native behavior also does not exclude retaining history in m and computing the answer only at the end.

## Audit three: refitting instability versus fixed-prediction results

The original primary test has M0_v13 MAE=2.6847533452 nat and M_CKA_v13=2.6996467606 nat; mean improvement from CKA is negative. Original refit-bootstrap MAE-improvement 95% interval **[−42.55952, 87.77731] nat** and RMSE-improvement interval **[−172.43769, 320.18672] nat** are retained. These wide intervals do not provide adequate precision and are not evidence that CKA is universally ineffective.

Replicates 0, 164, 285, 991 were replayed (original-data-defined typical 164, minimum 991 and maximum 285 MAE improvement, plus original 0/285), giving 72 all/rec/both×held-out-model×M0/CKA fits. Every copy of the held-out original seed is jointly excluded. Saved diagnostics include model weights, number of unique training models, feature mean/scale/raw range/test-z range, ridge coefficients, maximum prediction, associated condition and finiteness. Paired M0/CKA training-label and unit hashes agree. Maximum replayed-versus-saved MAE difference is 0.

For replicate 285 holding out 106, fitting seeds are `[101,101,105]`, only two unique models. Nearly constant control distance produces extreme test z-score extrapolation. Replayed maximum |z| is 342629.284376 and maximum |prediction| 87359.894091 nat; all predictions remain finite. No seed-copy leakage or unpaired resampling was found; numerical extrapolation alone is not an implementation bug. 106 was not removed, predictions were not clipped, and neither alpha nor the variance floor was changed. See [bootstrap_replay_metadata.json](../../../../../experiments/memory_interchange/Final_audit/bootstrap_replay_metadata.json), the [feature table](../../../../../experiments/memory_interchange/Final_audit/bootstrap_feature_diagnostics.csv), [replay scores](../../../../../experiments/memory_interchange/Final_audit/bootstrap_replay_scores.csv) and original [bootstrap replicates](../../../../../experiments/memory_interchange/results/v1_3/prediction/bootstrap.csv).

| rep285 held106 | Fitting-model weights | Original control training range | Original training std / scale | Original test range | max |test z| | max |Prediction| nat |
| --- | --- | --- | --- | --- | --- | --- |
| M0_v13 | 101×2,105×1 | [0, 3.21e-06] | 9.06e-07 / 9.06e-07 | [0, 0.31] | 342629 | 87359.9 |
| M_CKA_v13 | 101×2,105×1 | [0, 3.21e-06] | 9.06e-07 / 9.06e-07 | [0, 0.31] | 342629 | 38877.5 |

| Held-out seed | M0 MAE | M_CKA MAE | Paired MAE improvement (nat) | Paired RMSE improvement |
| --- | --- | --- | --- | --- |
| 101 | 2.46463021 | 2.51708897 | -0.05245876 | -0.06394987 |
| 102 | 2.44208739 | 2.49270992 | -0.05062253 | -0.02285060 |
| 103 | 1.95262642 | 2.02569391 | -0.07306750 | -0.06176421 |
| 104 | 2.85718927 | 2.66600606 | +0.19118321 | +0.04457927 |
| 105 | 2.42507087 | 2.52251304 | -0.09744217 | -0.09385756 |
| 106 | 3.96691592 | 3.97386867 | -0.00695275 | +0.01219553 |

Fixed-prediction results: mean -0.01489342 nat, range [-0.09744217, +0.19118321], only 104 positive and the remaining 5 negative. Paired resampling of the six observed fold scores 100,000 times yields a conditional empirical interval of [-0.07401677, 0.07351177] nat. This conditions on these six trained models/predictors and the original test data. It includes no refitting or retraining, no unknown-model-population or training uncertainty, and does not replace the original refit bootstrap. See [fixed_prediction_summary.json](../../../../../experiments/memory_interchange/Final_audit/fixed_prediction_summary.json).

Refitting instability on degenerate fitting sets and lack of average improvement for fixed predictions are separate findings. Under this design, prescribed baseline and original held-out conditions, no stable incremental predictive gain from CKA was observed. This does not establish that CKA can never predict function.

## Numerical checks, budget and stopping

The original numerical note is retained: **registered GPU controls passed; the original strict CPU float32 logit tolerance failed; the same coefficients in float64 agreed and original compared argmax answers were unchanged.** COMPLETED means the limited experiments/audits finished, not that every precision check passed. New GPU native caches versus uninterrupted computation pass the original atol=rtol=1e−5, with unchanged argmax and exact sham/NN agreement. New looser tolerances do not erase the original CPU failure.

Inherited conservative GPU accounting was 3313.147 seconds. The final cumulative total is **4043.451 seconds (1.1232 GPU hours)** against a 12-hour cap; this audit adds 730.304 seconds. Cost definitions and original timings are in [budget.json](../../../../../experiments/memory_interchange/Final_audit/budget.json). Hash checks for original checkpoints, source, curves, primary predictions and all bootstrap replicates are in [preservation_check.json](../../../../../experiments/memory_interchange/Final_audit/preservation_check.json).

Reproduction commands and artifact indices are in [README.md](README.md); audit source is in [final_audit/](../../../../../experiments/memory_interchange/final_audit); frozen rules and membership are in this directory. [artifact_manifest.json](../../../../../experiments/memory_interchange/Final_audit/artifact_manifest.json) records all new artifact hashes. Outputs for all six seeds are delivered. Work stops after these three audits: no new models, parameter scans, adapters, neuron ablations or four-step extensions. Negative results are retained.

## Exportable figures

![Cue-translation effect on the independent set](../../../../../experiments/memory_interchange/Final_audit/figures/translation_gain.png)

![Deduplicated geometry and mean contribution](../../../../../experiments/memory_interchange/Final_audit/figures/geometry.png)

![Six folds of the original fixed predictions](../../../../../experiments/memory_interchange/Final_audit/figures/fixed_prediction_scores.png)

Each figure is also provided as PDF/SVG.
