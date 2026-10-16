> English edition · [Frozen original](../../../../../experiments/memory_interchange/v1_3/README.md) · [English index](../../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

# v1.3: can memory states with identical content be interchanged?

Run `python -m experiments.memory_interchange.v1_3.run`. First lock seed0 stage 7 best=49,500, replay T/V and complete old fixed-V exploration. Do not continue seed0 training. V outcomes do not determine formal execution.

Stages 1–6 follow v1.2. Stage 7 uses original 65 words/8192 V, requiring total ≥97% and every k/r/d/query_type ≥95% twice consecutively, with all fine operation strata reported. Seeds 101–106 initialize afresh; failures are not replaced. Caps stay fixed; cumulative 12 GPU hours inherit 1080.078962639178 seconds.

Primary L_all is unchanged. rec/both/three masks are fixed by native outputs before patches and shared by both treatments. Save native-error repairs, current/initial-input counterfactuals and overlaps. The two T alpha=1 probes are diagnostic only.

G measures geometry/D_input; B measures behavior. Main held-out M_CKA_v13 is compared with M0_old+D_input. LOSO, 91 fitting/62 unseen pairs and 32 unseen cross-context primary pairs remain unchanged. Retain rec/both, quality tiers, old baselines and initial CKA. Freeze source/hashes before opening formal G/B.

Artifacts: `../results/v1_3/`, `../runs/v1_3/`. V exploration and formal reports remain separate. Original v1.2 BLOCKED is not rewritten. Scientific null results remain; no publication/push.
