> English edition · [Frozen original](../../../../../experiments/memory_interchange/donor_consistency/report.md) · [English index](../../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

# State interchange and donor-native output consistency audit

Date: 2026-10-13 (America/Los_Angeles). Status: **COMPLETED**. This audit only reads B1/B_reserved caches from the existing six frozen models. There is no network forward pass, retraining, additional seed or probe/predictor refitting; no push, publication or submission was performed in that audit.

## Pairing coverage and provenance

The actual project directory is `experiments/memory_interchange/`. The root README, v1.3 README and Final audit were read; no applicable AGENTS.md was found. B1=134 contents/19 complete orbits; B_reserved=362/51; membership exactly matches the original manifests and the sets are disjoint. **Both datasets had already been opened. This is an additional descriptive audit; B_reserved is no longer an unopened independent confirmation set.**

| Dataset | Expected nonself query records | Actually pairable | Missing | Coverage |
| --- | --- | --- | --- | --- |
| B1 | 41332032 | 41332032 | 0 | 100.00000000% |
| B_reserved | 111658176 | 111658176 | 0 | 100.00000000% |

Primary averages retain equal weighting of both directions of 153 unordered pairs (306 nonself directions), six seeds, d, original S and query type. Arrays also retain 18 self-directions as controls, totaling 8999424 query records; these are not new cross-condition evidence. The nonself cell for same context and same history is empty by definition, not because of missing data.

Native keys include dataset, seed, donor (r,H), complete content ID, d/swap event, actual effective_suffix, query type and item. Patch keys additionally include receiver (r,H), treatment and original S. Condition/content axes are verified against actual state-cache metadata and preserved source. Every native key and patch key is unique. Each native key expands in a controlled way to 17 receivers×two treatments, cardinality 34, without unintended many-to-many joins. Coverage is detailed in [pairing_coverage.csv](../../../../../experiments/memory_interchange/donor_consistency/pairing_coverage.csv) and [pairing_audit.json](../../../../../experiments/memory_interchange/donor_consistency/pairing_audit.json).

Untranslated pairs compare `(m_a(q),c_b),S` with `(m_a(q),c_a),S`; translated pairs compare `(m_a(q),c_b),T(S)` with `(m_a(q),c_a),T(S)`. Thus an original S=NA translated patch uses **donor NB**, and original S=NB uses **donor NA**. Receiver native outputs are not substituted, and translated outputs are not paired with donor original S. Donor history, initial input `p_a=G_a^{-1}(q)` and shared current q were verified.

The actual post-swap continuation lasts 16 steps with no external context reminder. Cues occupy relative steps 1/5; each slot has a subsequent 3-step hold, followed by a final 5-step hold; queries occupy relative steps 14–16. The swap event is 21+d, first query 35+d, final readout 37+d. Saved `native_logp` is the donor's native continuation from its natural state and matches the patch's effective_suffix inputs, holds and query timing. Uninterrupted native outputs are separate joining/numerical controls. No condition was excluded for mismatched external inputs.

## Answer agreement and probability differences

The primary comparison uses all pairable queries, with no selection by post-treatment correctness or agreement. Both argmax operations choose the lowest tied class index, with class order 0–6. Probabilities use float64 `exp(logp)`, checking finiteness and original row-sum error ≤2e−6 before row normalization. Each query is compared using maximum absolute class-probability difference and TV, not raw logits with an arbitrary common shift. Exact equality, tolerance closeness and answer agreement are distinct descriptions, not new scientific success thresholds.

Rates below weight six models/directions equally; probability means use the same definition. Bracketed intervals jointly resample orbit blocks for the fixed six models only:

| Dataset | Treatment | Context | Answer agreement | Empirical 95% interval (%) | All 14 queries agree | Mean maximum probability difference | Mean TV | Largest maximum probability difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | original | same | 99.99987661% | [99.99969606, 100.00000000] | 99.99827253% | 8.132789e-07 | 8.3374504e-07 | 0.0067318028 |
| B1 | original | cross | 99.99989032% | [99.99972983, 100.00000000] | 99.99846447% | 7.231154e-07 | 7.4131131e-07 | 0.0067318028 |
| B1 | translated | same | 99.99987661% | [99.99969606, 100.00000000] | 99.99827253% | 8.132789e-07 | 8.3374504e-07 | 0.0067318028 |
| B1 | translated | cross | 99.99989032% | [99.99972983, 100.00000000] | 99.99846447% | 7.231154e-07 | 7.4131131e-07 | 0.0067318028 |
| B_reserved | original | same | 99.99990865% | [99.99982503, 99.99997741] | 99.99872110% | 6.1041822e-07 | 6.2369657e-07 | 0.0081950501 |
| B_reserved | original | cross | 99.99991880% | [99.99984447, 99.99997992] | 99.99886320% | 5.4268076e-07 | 5.5448527e-07 | 0.0081950501 |
| B_reserved | translated | same | 99.99990865% | [99.99982503, 99.99997741] | 99.99872110% | 6.1041822e-07 | 6.2369657e-07 | 0.0081950501 |
| B_reserved | translated | cross | 99.99991880% | [99.99984447, 99.99997992] | 99.99886320% | 5.4268076e-07 | 5.5448527e-07 | 0.0081950501 |

Query numerators/denominators differ conceptually from equally weighted means. Complete all cells have equal direction sizes, so pooled query rates equal equally weighted rates; donor-correct/error subsets or missingness can differ, and both are saved. Six-seed `answer_agreement_equal_cells` first equally weights each seed's nonempty direction×d/S×type cells, then the six seeds. `answer_agreement_pooled` divides the total numerator by total denominator. Probability means retain this distinction. Medians and 95/99 percentiles describe pooled observed queries, without treating queries as independent or substituting averages of model medians for pooled medians.

## Six seeds, same/different histories and numerical details

| Dataset | seed | Treatment | Matching answers | Query count | Answer disagreements | Answer agreement | All 14 agree | Maximum probability difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 101 | original | 3444336 | 3444336 | 0 | 100.00000000% | 100.00000000% | 9.0487477e-06 |
| B1 | 101 | translated | 3444336 | 3444336 | 0 | 100.00000000% | 100.00000000% | 9.0487477e-06 |
| B1 | 102 | original | 3444336 | 3444336 | 0 | 100.00000000% | 100.00000000% | 0.00096037615 |
| B1 | 102 | translated | 3444336 | 3444336 | 0 | 100.00000000% | 100.00000000% | 0.00096037615 |
| B1 | 103 | original | 3444336 | 3444336 | 0 | 100.00000000% | 100.00000000% | 3.7718091e-05 |
| B1 | 103 | translated | 3444336 | 3444336 | 0 | 100.00000000% | 100.00000000% | 3.7718091e-05 |
| B1 | 104 | original | 3444336 | 3444336 | 0 | 100.00000000% | 100.00000000% | 0.00014114112 |
| B1 | 104 | translated | 3444336 | 3444336 | 0 | 100.00000000% | 100.00000000% | 0.00014114112 |
| B1 | 105 | original | 3444336 | 3444336 | 0 | 100.00000000% | 100.00000000% | 3.5675602e-06 |
| B1 | 105 | translated | 3444336 | 3444336 | 0 | 100.00000000% | 100.00000000% | 3.5675602e-06 |
| B1 | 106 | original | 3444312 | 3444336 | 24 | 99.99930320% | 99.99024485% | 0.0067318028 |
| B1 | 106 | translated | 3444312 | 3444336 | 24 | 99.99930320% | 99.99024485% | 0.0067318028 |
| B_reserved | 101 | original | 9304842 | 9304848 | 6 | 99.99993552% | 99.99909724% | 1.5635247e-05 |
| B_reserved | 101 | translated | 9304842 | 9304848 | 6 | 99.99993552% | 99.99909724% | 1.5635247e-05 |
| B_reserved | 102 | original | 9304836 | 9304848 | 12 | 99.99987103% | 99.99819449% | 0.0012415008 |
| B_reserved | 102 | translated | 9304836 | 9304848 | 12 | 99.99987103% | 99.99819449% | 0.0012415008 |
| B_reserved | 103 | original | 9304848 | 9304848 | 0 | 100.00000000% | 100.00000000% | 0.00013489272 |
| B_reserved | 103 | translated | 9304848 | 9304848 | 0 | 100.00000000% | 100.00000000% | 0.00013489272 |
| B_reserved | 104 | original | 9304836 | 9304848 | 12 | 99.99987103% | 99.99819449% | 0.00012381079 |
| B_reserved | 104 | translated | 9304836 | 9304848 | 12 | 99.99987103% | 99.99819449% | 0.00012381079 |
| B_reserved | 105 | original | 9304848 | 9304848 | 0 | 100.00000000% | 100.00000000% | 7.1279598e-06 |
| B_reserved | 105 | translated | 9304848 | 9304848 | 0 | 100.00000000% | 100.00000000% | 7.1279598e-06 |
| B_reserved | 106 | original | 9304830 | 9304848 | 18 | 99.99980655% | 99.99729173% | 0.0081950501 |
| B_reserved | 106 | translated | 9304830 | 9304848 | 18 | 99.99980655% | 99.99729173% | 0.0081950501 |

| Dataset | Treatment | Context | History | n | Answer agreement | All 7 item queries agree | All 7 position queries agree | All 14 agree | Mean TV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | original | same | same | 0 | — | — | — | — | — |
| B1 | original | same | different | 9725184 | 99.99987661% | 99.99827253% | 100.00000000% | 99.99827253% | 8.3374504e-07 |
| B1 | original | cross | same | 1215648 | 100.00000000% | 100.00000000% | 100.00000000% | 100.00000000% | 1.8653286e-09 |
| B1 | original | cross | different | 9725184 | 99.99987661% | 99.99827253% | 100.00000000% | 99.99827253% | 8.3374205e-07 |
| B1 | translated | same | same | 0 | — | — | — | — | — |
| B1 | translated | same | different | 9725184 | 99.99987661% | 99.99827253% | 100.00000000% | 99.99827253% | 8.3374504e-07 |
| B1 | translated | cross | same | 1215648 | 100.00000000% | 100.00000000% | 100.00000000% | 100.00000000% | 1.8653286e-09 |
| B1 | translated | cross | different | 9725184 | 99.99987661% | 99.99827253% | 100.00000000% | 99.99827253% | 8.3374205e-07 |
| B_reserved | original | same | same | 0 | — | — | — | — | — |
| B_reserved | original | same | different | 26272512 | 99.99990865% | 99.99968027% | 99.99904082% | 99.99872110% | 6.2369657e-07 |
| B_reserved | original | cross | same | 3284064 | 100.00000000% | 100.00000000% | 100.00000000% | 100.00000000% | 7.9654343e-10 |
| B_reserved | original | cross | different | 26272512 | 99.99990865% | 99.99968027% | 99.99904082% | 99.99872110% | 6.2369636e-07 |
| B_reserved | translated | same | same | 0 | — | — | — | — | — |
| B_reserved | translated | same | different | 26272512 | 99.99990865% | 99.99968027% | 99.99904082% | 99.99872110% | 6.2369657e-07 |
| B_reserved | translated | cross | same | 3284064 | 100.00000000% | 100.00000000% | 100.00000000% | 100.00000000% | 7.9654343e-10 |
| B_reserved | translated | cross | different | 26272512 | 99.99990865% | 99.99968027% | 99.99904082% | 99.99872110% | 6.2369636e-07 |

| Dataset | Treatment | Probability metric | Mean | Median | q95 | q99 | Maximum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | original | max_abs_probability_difference | 7.6554528e-07 | 0 | 2.2650791e-07 | 5.88446e-06 | 0.0067318028 |
| B1 | original | TV | 7.8480953e-07 | 0 | 2.2992529e-07 | 5.955147e-06 | 0.0068133535 |
| B1 | translated | max_abs_probability_difference | 7.6554528e-07 | 0 | 2.2650791e-07 | 5.88446e-06 | 0.0067318028 |
| B1 | translated | TV | 7.8480953e-07 | 0 | 2.2992529e-07 | 5.955147e-06 | 0.0068133535 |
| B_reserved | original | max_abs_probability_difference | 5.7455721e-07 | 0 | 2.9534424e-07 | 7.3155736e-06 | 0.0081950501 |
| B_reserved | original | TV | 5.8705529e-07 | 0 | 2.9996927e-07 | 7.4369976e-06 | 0.0083350462 |
| B_reserved | translated | max_abs_probability_difference | 5.7455721e-07 | 0 | 2.9534424e-07 | 7.3155736e-06 | 0.0081950501 |
| B_reserved | translated | TV | 5.8705529e-07 | 0 | 2.9996927e-07 | 7.4369976e-06 | 0.0083350462 |

| Dataset | Treatment | Probability-comparable queries | All 7 probabilities exactly equal | Elementwise exact equality | ≤1e−6 | ≤1e−5 | ≤1e−4 | Patch argmax ties | Donor argmax ties |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | original | 20666016 | 51.15958006% | 51.67967892% | 97.63404809% | 99.23930670% | 99.86599256% | 0 | 0 |
| B1 | translated | 20666016 | 51.15958006% | 51.67967892% | 97.63404809% | 99.23930670% | 99.86599256% | 0 | 0 |
| B_reserved | original | 55829088 | 52.08585352% | 52.50610680% | 97.26588942% | 99.16328563% | 99.89566729% | 0 | 0 |
| B_reserved | translated | 55829088 | 52.08585352% | 52.50610680% | 97.26588942% | 99.16328563% | 99.89566729% | 0 | 0 |

Full-vector agreement means all 14 argmax answers agree, with both 7-query vectors separately retained. Predictions need not be valid. Nonbijective predictions and readouts that are not mutual inverses remain included; identical wrong/invalid vectors do not establish full-permutation correctness. Counts, all S/d/query types, empty crossed cells and all six seeds are in [seed_summary.csv](../../../../../experiments/memory_interchange/donor_consistency/seed_summary.csv) and [strata_summary.csv](../../../../../experiments/memory_interchange/donor_consistency/strata_summary.csv). S strata use original receiver literal S; effective_suffix is saved separately.

## Are donor errors reproduced?

Donor correctness/error is defined against its own actual-effective_suffix truth `F_a(effective_suffix)(q)`, not original S or receiver targets. Truth labels actual donor outputs; it never replaces missing predictions. Expanded query counts reuse the same native outputs across multiple receivers and are not independent error samples.

| Dataset | Treatment | Donor subset | n | Reproduced answers | Disagreements | Pooled query agreement | Equal-weight nonempty-cell agreement | Mean maximum difference | Mean TV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | original | correct | 20287477 | 20287453 | 24 | 99.99988170% | 99.99985802% | 5.4849102e-07 | 5.5927516e-07 |
| B1 | original | error | 378539 | 378539 | 0 | 100.00000000% | 100.00000000% | 1.2398384e-05 | 1.2872133e-05 |
| B1 | translated | correct | 20287477 | 20287453 | 24 | 99.99988170% | 99.99985802% | 5.4849102e-07 | 5.5927516e-07 |
| B1 | translated | error | 378539 | 378539 | 0 | 100.00000000% | 100.00000000% | 1.2398384e-05 | 1.2872133e-05 |
| B_reserved | original | correct | 54633325 | 54633295 | 30 | 99.99994509% | 99.99993149% | 4.3849057e-07 | 4.4597533e-07 |
| B_reserved | original | error | 1195763 | 1195745 | 18 | 99.99849468% | 99.99798712% | 6.7913185e-06 | 7.0328705e-06 |
| B_reserved | translated | correct | 54633325 | 54633295 | 30 | 99.99994509% | 99.99993149% | 4.3849057e-07 | 4.4597533e-07 |
| B_reserved | translated | error | 1195763 | 1195745 | 18 | 99.99849468% | 99.99798712% | 6.7913185e-06 | 7.0328705e-06 |

When the donor is wrong, an identical patch answer reproduces the same error; it does not mean agreement with donor-rule truth. This metric, the previous donor-rule truth agreement and translation recovery of the original target are three distinct quantities. Empty donor-error subcells remain empty; all strata retain counts and missingness, and 106 is not removed.

## Control states and implementation checks

| Dataset | History | Seed×d×direction count | Exactly equal raw c pairs | Mean L2 | Maximum L2 |
| --- | --- | --- | --- | --- | --- |
| B1 | same | 216 | 192 | 3.2434037e-08 | 9.1108606e-07 |
| B1 | different | 3456 | 840 | 0.1066239 | 2.0257148 |
| B_reserved | same | 216 | 192 | 3.2434037e-08 | 9.1108606e-07 |
| B_reserved | different | 3456 | 840 | 0.1066239 | 2.0257148 |

Raw c contrasts use saved natural states without generating new states; per-model/direction results are in [control_state_differences.csv](../../../../../experiments/memory_interchange/donor_consistency/control_state_differences.csv). If c_a=c_b and actual future inputs coincide, the initial state equals the donor-native initial state; identical deterministic mathematical outputs are expected and are not an independent mechanism discovery. Different GPU batch implementations may still produce small probability differences. Same-history results cannot be generalized to every different history, nor can identical answers establish that c universally lacks a causal role.

The exhaustive NN check covers 24 dataset×seed×d cells. S and T(S) use the same donor source, and patch outputs and primary comparison results agree for all 324 cached directions. Argmax agrees for native/sham, uninterrupted native, and saved B1 full-state controls with identical subsequent inputs. Available control probability differences are in [supplemental_controls.csv](../../../../../experiments/memory_interchange/donor_consistency/supplemental_controls.csv). B_reserved has no saved full-state control; this optional control is not fabricated and does not affect complete primary pairing.

The independent direct-read small sample was frozen before inspecting outputs. It covers six seeds, two datasets, same/cross context, every feasible same/different history, all S/d/treatments and both query types. Source indices were searched afresh using actual q/(r,H) for 864 keys; recomputed predictions/probabilities were checked against vectorized joins. See [direct_index_validation.json](../../../../../experiments/memory_interchange/donor_consistency/direct_index_validation.json).

## Uncertainty, artifacts and limitations

Fixed RNG seed=20261013, with 10,000 orbit-block replicates. B1's 19 and B_reserved's 51 orbits are sampled separately; each draw is shared across all six models, retaining every content, direction, treatment, d, S and query. Models stay fixed: no model resampling or retraining. Intervals condition on these six models and do not cover an unknown model population. See [block_intervals.json](../../../../../experiments/memory_interchange/donor_consistency/block_intervals.json); original orbit weights and draws are retained.

Frozen definitions apply only to this post hoc analysis; they do not retrospectively preregister previous findings. Input locations/hashes, model hashes, axes, pairing keys, class order, aggregation, tolerances, resampling and code hashes are in [analysis_plan.json](../../../../../experiments/memory_interchange/donor_consistency/analysis_plan.json). Before/after hashes agree for all 2055 registered historical files; original reports, results, models, members and failure records are unchanged.

The original numerical note remains: **registered GPU controls passed; the original strict CPU float32 logit tolerance failed; the same coefficients in float64 agreed and argmax was unchanged.** The present probability tolerances do not replace or erase the original logit failure.

Predictions/donor truth, flags and continuous probability differences for every pairable query are saved in compact tensors. Complete semantic keys and source-probability locations are in [input_index.json](../../../../../experiments/memory_interchange/donor_consistency/input_index.json). Answer disagreements or maximum probability difference >1e−5 additionally save both full seven-class probability vectors, complete keys, truth and error flags. Other small nonzero differences remain locatable, with both probability vectors recoverable using `inspect_record`, avoiding duplication of the existing probability library. Commands and index formats are in [README.md](README.md). No mechanism explanation is invented for every small difference.

English result sentence suitable for an abstract:

> Across the six frozen RNNs and tested continuations, memory-state patches reproduced donor-native outputs (B1: 99.99988387% of query answers and 99.99837414% of complete 14-query prediction vectors agreed, with mean total variation 7.8481e-07 and maximum probability difference 0.0067318; B_reserved: 99.99991402% of query answers and 99.99879633% of complete 14-query prediction vectors agreed, with mean total variation 5.87055e-07 and maximum probability difference 0.00819505), including donor-native errors; these are descriptive comparisons on previously opened data.

These results describe output agreement under the tested continuations. High answer agreement still requires reporting the actual probability differences. It does not establish pure-content encoding in m, rules exclusively in m, no causal role for c, equivalence on all future tasks, separated subspaces or any disease-repair mechanism. The original primary CKA test and translation-recovery conclusions remain unchanged. Work stops after this cache audit.

Read-only CPU analysis took 112.53985 seconds (comparison stage only). Additional GPU time is **0**; inherited cumulative conservative accounting remains 4043.4506 seconds. No new network calls or external publication.
