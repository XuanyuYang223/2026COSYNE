> English edition · [Frozen original](../../../../../experiments/memory_interchange/Final_audit/README.md) · [English index](../../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

# v1.3 Final audit

These are the three limited audits authorized by the user. The six formal RNNs are not trained again; the original v1.3 metrics, point estimates, all bootstrap replicates, seed 106, and the failed strict CPU float32 tolerance check are retained. The main report is [report.md](report.md).

## Reproduction / recovery

From the repository root, use the original Python/Torch environment:

```bash
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m experiments.memory_interchange.final_audit.common
python -m experiments.memory_interchange.final_audit.behavior native >> experiments/memory_interchange/Final_audit/native.log 2>&1
python -m experiments.memory_interchange.final_audit.validate
python -m experiments.memory_interchange.final_audit.behavior CF
python -m experiments.memory_interchange.final_audit.geometry_probe
python -m experiments.memory_interchange.final_audit.statistics
python -m experiments.memory_interchange.final_audit.behavior patch >> experiments/memory_interchange/Final_audit/translation.log 2>&1
python -m experiments.memory_interchange.final_audit.behavior summary
python -m experiments.memory_interchange.final_audit.summarize
python -m experiments.memory_interchange.final_audit.finish check
python -m experiments.memory_interchange.final_audit.finish cost
python -m experiments.memory_interchange.final_audit.report
python -m experiments.memory_interchange.final_audit.finish inventory
```

`native` must finish saving native outputs and masks for every model before `patch` is allowed. Completed GPU outputs are not recomputed. `CF` reuses all original B1 matched/wrong outputs; new translated B1 patches are replayed separately and checked against the saved corresponding S′ outputs. Both original and translated B_reserved patches are newly computed. Exact reproduction requires an isolated workspace with complete original inputs; do not overwrite the original reports or checkpoints. The cumulative GPU budget inherits prior accounting and remains capped at 12 hours.

## Frozen rules and provenance

- `frozen_plan.json`, `revision_final_audit.yaml`, `freeze_receipt.json`: the new hypothesis, two targets, all conditions, masks, primary contrast, and fixed probe/statistics rules.
- `B_reserved_members.npz` / `B_reserved_manifest.json`: the exact original B-minus-B1 set, with 362 contents / 51 complete orbits; all remaining orbits are used. B1 is exploratory; the reserved set confirms the translation contrast.
- `all_native_masks_frozen.json` and per-seed `mask_freeze.json`: native masks saved before new patches.
- `preservation_manifest.json` / `preservation_check.json`: hashes of the original v1.0–v1.3 source, configurations, checkpoints and reports.
- `artifact_manifest.json`: complete hashes of new code and artifacts; volatile Python bytecode is excluded.
- `budget.json` / `cost_windows.json`: GPU accounting and conservative process windows including initialization, saving, CPU reduction/IO and completion-observation latency; actual busy GPU time is lower.

## Arrays and aggregation

All `B1/seed_{101..106}` and `B_reserved/seed_{101..106}` outputs are retained. Conditions follow the original `truth.CONDITIONS`: nine histories H for r=0, followed by the same nine for r=1; H is NN, NA, AN, NB, BN, AA, BB, AB, BA.

The two patch log-probability arrays in `translation_d*_S*.npz` have axes `[donor,receiver,content,query_type,query_id,class]` and shape `[18,18,Q,2,7,7]`. query_type=0 means item→position and 1 means position→item; argmax over classes reconstructs forward/reverse mappings using the two query axes. `Q_original=F_b(S)(q)` and `Q_literal=F_b(swap(S))(q)` have shape `[condition,Q,2,7]`; donor candidates use the donor condition and receiver candidates use the receiver condition. Both directions of each unordered condition pair are saved; diagonal entries are stored but excluded from the main descriptions.

`counterfactual_d*_S*.npz` is B1-only: all single-query and full-mapping predictions, three matched candidates, two wrong-content candidates, pairwise-distinct flags, an all-three-distinct flag, contents/orbits and fixed mismatch indices. Flag complements encode candidate overlaps. Wrong content has no initial-input candidate; the corresponding CSV column is empty.

The all/rec/both/E_sem arrays in `masks_d*_S*.npz` all have shape `[18,18,Q,2,7]`. The same mask is used for both patches; samples are not selected by treatment outcomes. E_sem requires the receiver native output to be correct for original S and the donor native output to execute the same semantic operation correctly: cross uses translated donor S; same uses original donor S. `both` only requires each model state to answer correctly for the same visible S.

`blocks_d*_S*.npz`: `sums[306 directions,4 masks,2 treatments,orbits,8 metrics]`, `counts[306,4,orbits]`. Metrics include original-target accuracy, CE, new errors, positive damage, new-literal-target accuracy/CE, and receiver/donor-rule agreement. Complete orbits retain every query and direction. The primary all contrast weights seeds and ordered directions equally and retains the original d/S/query distribution. Conditional descriptions first aggregate within each direction's fixed mask, then weight directions, d, S and seeds equally. Counts are observations, not independent sample sizes.

## Tables and checks

- `translation_seed_summary.csv` / `translation_overall_summary.csv`: every mask, same/cross context, S and d; original/translated accuracy, CE, new errors and paired differences. Detailed query types, directions and counts are in each `translation_d*_S*.csv`.
- `translation_full_mapping_diagnostics.csv`: bijectivity, mutual inverses, and full-mapping agreement with original/literal/donor candidates, for all mappings and distinguishable subsets.
- `candidate_*summary.csv`: counts, CE, single-query and full-mapping agreement for pairwise-distinguishable and all-three-distinct subsets; original strata are in each B1 seed directory.
- `native_all_strata.csv`: original and new literal targets for all H/r/d/S/query types, including native, translated native and sham; `native_seed_summary.csv` retains the poor conditions of 106.
- `geometry_decomposition.csv` / `geometry_distributions.csv`: deduplicated G CKA distributions, the D_raw mean/center decomposition and its independent numerical check; `original_fixed_counterexamples.csv` retains all 288 cases selected by the original fixed rule.
- `context_probe_results.csv`, `context_raw_contrasts.csv`, `context_probe_fit_inventory.csv`: fixed T-only fitting on nonmixed H with AB/BA held out, pooled-d and both single-d models; c does not replicate q. Per-seed parameters, scores and margins are in `context_probes/`.
- `bootstrap_replay_metadata.json`, `bootstrap_feature_diagnostics.csv`, `bootstrap_replay_scores.csv`: complete audits of typical/extreme original replicates; no held-seed-copy leakage; M0/CKA share resampling, labels and test units. Alpha, variance floor and predictions are unchanged.
- `fixed_prediction_summary.json`: paired fold scores and conditional empirical intervals for the original six fixed LOSO models/test predictions; this does not replace the refit bootstrap or include uncertainty from retraining or an unknown model population.
- `pre_patch_checks.json` / `post_patch_checks.json`: the translation identity on all 5,040 permutations, mask freezing and semantics, exact NN invariance, independent point reductions and block-denominator checks.

Geometry, linear readability, compensation by external semantic translation and mechanisms in real brains are distinct levels of evidence. Retaining physical c does not establish retention of the receiver's functional rule. Work stops after this checklist: no parameter scans, adapters, ablations or four-step extensions.
