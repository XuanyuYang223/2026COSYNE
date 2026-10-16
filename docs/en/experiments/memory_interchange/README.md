> English edition · [Frozen original](../../../../experiments/memory_interchange/README.md) · [English index](../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

# Memory-state interchange experiment

Current version status: **BLOCKED — neither of the two prescribed n=7 pilot models learned the original task.**
The default 128-dimensional and sole fallback 256-dimensional models each completed 20,000 updates. Best fixed-validation accuracies were 14.7827% and 14.8926%, against a 97% threshold. Failures are retained under the stopping rule; the formal six-seed experiment, H2/H3 held-out predictions and unseen four-step-word extension were not run.

See [report.md](report.md). Pilot geometry and interchange data serve only implementation, throughput and failure diagnosis; they are not scientific conclusions.

## Task and research question

Test whether geometric similarity of natural RNN states with identical complete memory content but different formation histories predicts functional interchangeability when the receiver's control state is retained.

Permutation `p[i]` denotes the position of item i; `I(p)[p[i]]=i` represents the inverse bijection, and `C(p)=n-1-p` mirrors positions. A/B mean I/C in context 0, and C/I in context 1; N always preserves the state. Inversion does not describe physical object movement.

Context is shown for 3 steps, the complete stimulus for 5, then a blank for 5. Each of four rule slots has a 1-step cue and 3-step blank. After the first two slots, retain for d∈{0,10} and interchange at the end of step 21+d. After the remaining slots and final 5-step blank, the query first appears at step 35+d and is read out at step 37+d. For each content, enumerate all 7 item and position queries.

The unidirectional control GRU receives neither the permutation nor m; the memory GRU receives the initial stimulus, current c and phase flags. Training uses only final-query cross-entropy. m/c are designated modules; their names do not establish natural brain specialization or spontaneous group structure in the network.

## Execution

Run from repository root `/home/yangx/2026COSYNE`. Use a PyTorch CUDA wheel supporting the local GPU; this version used torch 2.11.0+cu128.

```bash
python -m pip install -r experiments/memory_interchange/requirements.txt
python -m experiments.memory_interchange.truth
python -m pytest experiments/memory_interchange/tests -q
python -m experiments.memory_interchange.run
```

The last command reuses completed pilots with matching training configurations, generates reports and uses the thresholds to determine entry into the formal six-seed experiment. Rerunning this version retains BLOCKED and does not automatically try other configurations. Incomplete training resumes from `latest.pt`, including optimizer and all RNG states. Without a GPU, only mathematical checks and a short CPU smoke test run.

Stage-specific reproduction:

```bash
python -m experiments.memory_interchange.train --seed 0 --out experiments/memory_interchange/runs/pilot_128 --resume
python -m experiments.memory_interchange.train --seed 0 --memory-dim 256 --out experiments/memory_interchange/runs/pilot_256 --resume
python -m experiments.memory_interchange.diagnose --run experiments/memory_interchange/runs/pilot_128
python -m experiments.memory_interchange.intervene --run experiments/memory_interchange/runs/pilot_128 --pilot
python -m experiments.memory_interchange.geometry --run experiments/memory_interchange/runs/pilot_128 --pilot
```

Formal entry points are gated by the pilot, implementation checks, budget and frozen source. `predict` requires complete labels from all six prescribed seeds and rejects formal analysis using a pilot or missing seeds. `extension` requires the formal original-behavior thresholds to pass. Neither entry point needs to run for this historical version.

## Data and analysis constraints

split_seed=20261012; enumerate I/C closure orbits before splitting. T/V/G/B contain 451/104/69/70 orbits and 3274/778/492/496 permutations. The formal B subset uses complete orbits in fixed order until first reaching 128 permutations, yielding 134; selection uses no model outcomes.

Primary CKA, native-coordinate distance and centered orthogonal Procrustes are computed on all G contents; the main intervention uses fixed B. Wrong content uses a fixed cross-orbit derangement. Original answers and donor-content counterfactual answers under receiver rules are both saved. Sham and same-context full-state swaps are implementation controls only.

Both `L=mean(max(0,ell_swap-ell_native))` and `R=mean(native correct and swap incorrect)` are reported; damage in opposite directions cannot cancel. Ridge alpha=1, with standardization and missing-value handling confined to the fitting fold. Primary prediction compares M0 with M_CKA; native distance, Procrustes and initialization CKA are prescribed comparisons. The outer split holds out seeds; fitting excludes 62 pairs involving AB/BA, with the 32 cross-context pairs as the main test. The held-out quantity is the predictor's interchange labels; task-network training includes those histories.

Counterexamples use a fixed rule within each seed/d/S/context category (0/0, 0/1, 1/1): both CKA and R at or above the 75th percentile and R≥0.5%. All qualifying cases and complete distributions are retained. Content probes use alpha=1, fitted only on balanced T condition/delay caches. Training categories are sampled uniformly at random, with random within-batch count fluctuations allowed. Every formal seed is retained, including seeds failing the original task thresholds.

## Files and array indices

| File | Contents |
|---|---|
| `protocol_v1.txt` / `config.yaml` | Original protocol and complete parameters |
| `split_manifest.json` / `.sha256.json` | All orbits, members, splits, B subset and checksums |
| `truth.py` / `task.py` | Array truth, independent matrix verification, fixed timing and sampling |
| `model.py` / `train.py` | Learned unidirectional m/c GRUs, checkpoints and resumable training |
| `cache.py` / `intervene.py` | Natural states, all controls, truth and orbit statistics |
| `geometry.py` / `predict.py` | float64 geometry, T probes, LOSO and hierarchical bootstrap |
| `diagnose.py` / `figures.py` / `report.py` | V failure diagnostics, exportable figures, reports and artifact hashes |
| `run.py` / `extension.py` | Finite budget, pilot stopping, freezing and conditional extensions |

Condition indices 0–8 use r=0 and 9–17 use r=1; history order is `NN,NA,AN,NB,BN,AA,BB,AB,BA`. Cached m axes are `[d,condition,q,neuron]`, and c axes `[d,condition,neuron]`. Every array saves contents and orbit_id, traceable to the manifest.

In each `raw_d*_S*.npz`, native/sham axes are `[receiver,q,query_type,query_item,output_class]`; matched/wrong axes are `[donor,receiver,q,query_type,query_item,output_class]`. query_type=0/1 denotes item/position queries. `logp` contains float32 natural-log probabilities. Full-state swaps separately save donor/receiver indices and are interpretable only as same-context controls.

In `orbit_blocks.npz`, `metric_sums` axes are `[d/S,donor,receiver,orbit,query_type,metric]`; `query_counts` gives each orbit's count per query type. On answer-distinct subsets, `wrong_*_different` is a sum of correct responses and must be divided by the sum of `answers_differ`. Raw NPZ files recover individual queries; CSV files summarize directions, query types and pairs. Initial, selected and latest checkpoints, learning curves and per-trial V predictions remain in `runs/`.

Large generated files are excluded by `.gitignore` but remain in the complete local delivery. `results/artifact_manifest.json` records each artifact's SHA-256. Copying source alone does not deliver this version's experimental data.

12 GPU hours is the cumulative cap. There is no automatic budget expansion, noise tuning to manufacture damage, architecture expansion or post-failure seed selection. Statistics from 6 seeds describe only this finite model set. Missing formal data are never filled with invented predictive performance or significance.
