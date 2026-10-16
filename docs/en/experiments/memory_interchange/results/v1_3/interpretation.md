> English edition · [Frozen original](../../../../../../experiments/memory_interchange/results/v1_3/interpretation.md) · [English index](../../../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

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

Full display: [all selected-checkpoint T/V behavior factors](all_behavior_factors.md); quality-tier MAE/RMSE, counts and masks are in quality_and_mask_summary.md; [methodological context](method_context.md).
