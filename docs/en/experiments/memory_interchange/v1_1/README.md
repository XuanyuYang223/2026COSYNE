> English edition · [Frozen original](../../../../../experiments/memory_interchange/v1_1/README.md) · [English index](../../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

# v1.1: task-learning revision

Current status: **BLOCKED; both A/B failed the stage 6 two-step gate.** The first five stages passed; stage 7 was not run. This is neither TASK_READY nor a formal scientific result.

The [complete report](../results/v1_1/report.md) retains both versions, training-path audits, every stage, actual operations, gradient/content/gate diagnostics and costs. [Changes](CHANGES.md) describes differences from v1.0.

## Scope and preservation

v1.0 source, reports, checkpoints, data and original splits remain in place; 101 original files are separately archived with SHA-256. The 134 formal B contents are reused exactly. This version runs no G/B model, geometry/damage analysis or scientific predictor.

New v1.1 source is here; results are in `results/v1_1/`, checkpoints in `runs/v1_1/`. Prior GPU cost is inherited from immutable v1.0 records; the cumulative cap including this version is 12 GPU hours.

## Reproduction entry point

Run from repository root using original v1.0 torch 2.11.0+cu128, NumPy and dependencies.

```bash
python -m experiments.memory_interchange.v1_1.run
```

The entry point verifies old artifacts, fixed data and implementation, reusing completed audit/scratch/course records. Rerunning the current failed version adds no training and does not automatically enter stage 7 or formal seeds.

Stage-specific entry points:

```bash
python -m experiments.memory_interchange.v1_1.audit
python -m pytest experiments/memory_interchange/tests experiments/memory_interchange/v1_1/tests -q
python -m experiments.memory_interchange.v1_1.train scratch --candidate A
python -m experiments.memory_interchange.v1_1.train curriculum --candidate A
python -m experiments.memory_interchange.v1_1.train curriculum --candidate B
```

B's curriculum is allowed only after A's curriculum fails, or A tiny-fit fails while B tiny-fit passes. Do not try B if A passes. Consecutive advancement points, failure caps and optimizer/RNG recovery are constrained by checkpoint state.

## Fixed methods

Task truth, encoding, unidirectional m/c architecture and final-query CE reuse v1.0; this version fixes dimensions at 128/32/64. Context and inverse/complement semantics remain only in generation/independent oracles, not network-weight equations.

| Stage | Timing | Training words/context | Cap |
|---:|---|---|---:|
| 1 | 9-step short | NNNN, r=0 | 5,000 |
| 2 | 23-step mid | NNNN, r=0/1 | 5,000 |
| 3 | 37/47-step original | NNNN, r=0/1, d=0/10 | 5,000 |
| 4 | original | k≤1, r=0 | 5,000 |
| 5 | original | k≤1, r=0/1 | 5,000 |
| 6 | original | k≤2, r=0/1 | 5,000 |
| 7 | Original v1.0 distribution | Full 65 words, uniform k=0/1/2/3 | 20,000 |

A uses default initialization. Sole fallback B initializes both GRUs' effective update-gate bias to +3, leaving every parameter trainable. Each candidate starts from a new seed=0 initialization. Scratch weights stay separate; advancement restores matching weights, Adam and all RNG.

Evaluate fixed V every 500 updates. Stages 1–6 each use 4,096 trials; stage 7 must use unchanged original 8,192-trial V. Total accuracy ≥97% and every context/actual-operation/delay/query-type/non-N factor ≥95% must pass twice consecutively. No arbitrary fine crossed-cell gates or reduced behavioral thresholds.

Generate all new evaluation sets before training, saving actual collated inputs and oracle labels. Additional balanced T/V uses seed=20261013; stage V seed=20261014. Balanced V does not replace original V. `datasets/manifest.json` saves hashes and factor counts.

`audit.py` checks old best/latest T/V behavior, stimulus removal/shuffling, padding and actual training forward. `trace.py` retains gradients on actual stepwise GRU nodes, compares fused logits/parameter/input derivatives and saves None/finiteness/norms. Short CPU float64 finite differences are independent of long-sequence tiny derivatives. `data.py`'s oracle reconstructs semantics from actual inputs without calling label-update functions.

Each stage's `curve.json` saves T/V strata/counts every 500 steps; `diagnostics.json` saves full evaluation responses, 64-trial state sensitivity, four-phase memory/control update gates and gradients. Diagnostic `logit/CE/gradient/L2` quantities are separate from behavioral gates and do not select geometry or manufacture swap damage.

Only successful stage 7 permits `ready.py` to fit T-only ridge content probes and stratify native V behavior. Even TASK_READY does not automatically launch formal science in this revision. Future initial CKA must use pre-curriculum initialization. Scientific cache times, G/B, swaps and prediction definitions remain v1.0.
