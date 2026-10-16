> English edition · [Frozen original](../../../../../experiments/memory_interchange/v1_2/CHANGES.md) · [English index](../../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

# v1.2 registered changes

1. Prefer and continue only candidate A with default initialization; retain B for diagnosis.
2. Restore A stage 6 latest (stage 5,000; course 18,500) weights, Adam and all RNG. Validate fixed T/V first; absolute accuracy tolerance 1e-12, CE tolerance 1e-6.
3. Stage 6/7 caps are 50,000 each, including existing stage 6's 5,000. Formal stages 1–5 allow 20,000 each; total per model ≤200,000. Do not retrain pilot stages 1–5.
4. Evaluate every 500 updates; total V ≥97% and original applicable factors ≥95% must pass twice consecutively. Stage 7 uses original 8,192-trial V; checkpoints use original-task V only.
5. Fixed T/V two-step diagnostics add actual/visible operations, slots, candidate overlaps/distinguishable subsets, complete item/position mappings, and state sensitivity/gradients/gates for failing operations.
6. Once the pilot passes the full task, freeze curriculum/science rules and automatically run new initializations 101–106, retaining failed seeds and continuing original G/B science. G/B remain sealed beforehand.
7. Keep cumulative 12-GPU-hour cap, including exactly 605.5145923811942 prior seconds and all validation, caching, saving and analysis costs.

Mathematical truth, 128/32/64 architecture, inputs, final-query CE, Adam parameters, clipping, balanced-k sampling, content orbits, G/B contents and original predictor holdouts are unchanged.
