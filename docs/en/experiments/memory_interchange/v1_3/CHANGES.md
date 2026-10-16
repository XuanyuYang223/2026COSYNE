> English edition · [Frozen original](../../../../../experiments/memory_interchange/v1_3/CHANGES.md) · [English index](../../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

# v1.3 registration and implementation notes

Lock seed0 to stage 7 best by total-V accuracy (49,500), without continued training. Original v1.2 BLOCKED/latest strata remain intact. V uses the old first 52 geometry orbits and latter-half 6 behavioral orbits for exploration only.

Formal stages 1–6 and fine-factor gates are unchanged. Stage 7 advancement uses total ≥97% and each k/context/d/query_type ≥95% twice consecutively. Every actual-operation fine stratum remains recorded. Seeds 101–106 initialize afresh, failures remain, and budgets/Adam parameters do not change.

Primary label remains L_all. Add fixed natural-output rec/both/three masks shared by matched/wrong treatments. R_all uses all queries, R_rec originally correct queries. Save repairs, directions, current/initial-input counterfactuals and mapping overlaps. Two T probes with identical samples/centering/alpha=1 read current/initial content.

Prediction retains LOSO/history-pair holdouts. Main baseline M0_v13=M0_old+D_input; the main model adds CKA. D_input uses G initial one-hot inputs only. Retain old M0+CKA, native distance, Procrustes and initial CKA. rec/both are sensitivity analyses; quality tiers are descriptive.

The additional strict CPU float32 logit check shows near-zero-logit numerical differences on trained weights. Save every failure, maximum difference and changed-answer count; exactly convert identical weights to float64 and recheck original 2e-7/2e-6 tolerances. Actual GPU float32 interventions retain registered logp 1e-5/1e-5; sham must agree elementwise. No relaxed thresholds, weight changes or outcome-selected contents.

Freeze source, configuration, contents, mask/quality definitions, model and prediction rules before formal G/B opening. V outcomes do not determine formal execution. The cumulative 12-GPU-hour cap inherits exactly 1080.078962639178 prior seconds.
