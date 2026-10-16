> English edition · [Frozen original](../../../../../experiments/memory_interchange/v1_2/README.md) · [English index](../../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

# Representational similarity and functional interchangeability of memory states: v1.2

This is an authorized v1.0/v1.1 budget revision. Task p[i]=position of item i; I inverts, C mirrors positions. A=I/B=C for r=0, reversed for r=1. The 128/32/64 dual-GRU architecture and final-query supervision remain unchanged.

Execute:

```bash
python -m experiments.memory_interchange.v1_2.run
```

Restore candidate A stage 6 latest and first verify fixed T/V. Stages 6/7 allow 50,000 updates each, evaluated every 500; advancement requires total accuracy ≥97% and prescribed factors ≥95% twice consecutively. Stage 7 must use original 65 words and 8,192-trial V. Do not retrain pilot stages 1–5.

After pilot success, freeze and automatically run new initializations 101–106. Formal early stages allow 20,000 each, final two 50,000, maximum 200,000 per model. Science reuses v1.0 functions: fixed G/B, m-only swaps retaining c, controls, geometry, alpha=1 held-out ridge and 1,000 seed/orbit bootstrap replicates. Failed seeds remain. The cumulative 12-GPU-hour budget includes 605.5145923811942 prior seconds.

Results: `../results/v1_2/`; checkpoints: `../runs/v1_2/`; report: `report.md`; original data: `../results/v1_1/datasets/`. Restore/diagnose separately with `python -m experiments.memory_interchange.v1_2.prepare`. Stage 6 latest includes old 5,000 updates and does not restart from zero.

H1 checks content-matched intervention; H2/H3 test CKA's held-out incremental prediction. Curriculum progress, decodability and correct behavior neither replace interventions nor prove sequential updating. Budget-limited failure means only missing thresholds within this budget. Original artifacts remain hash-protected; no publication/push.
