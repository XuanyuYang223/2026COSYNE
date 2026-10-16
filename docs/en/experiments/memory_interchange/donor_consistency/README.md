> English edition · [Frozen original](../../../../../experiments/memory_interchange/donor_consistency/README.md) · [English index](../../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

# Donor-native output consistency

This directory contains the read-only descriptive audit added on 2026-10-13. Both B1 and B_reserved had already been opened, so this is not an independent confirmation. No RNN is run, trained or refitted; original files remain intact. The main report is [report.md](report.md).

## Execution

Run from `/home/yangx/2026COSYNE`:

```bash
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m experiments.memory_interchange.donor_consistency.test_math
python -m experiments.memory_interchange.donor_consistency.prepare
python -m experiments.memory_interchange.donor_consistency.analyze
python -m experiments.memory_interchange.donor_consistency.summarize
python -m experiments.memory_interchange.donor_consistency.validate
python -m experiments.memory_interchange.donor_consistency.report
```

`prepare` reads only axes, members, source, input timing and hashes, saving the plan, code hashes and fixed direct-read sample before consistency computation. Existing plans are not overwritten; subsequent commands first verify unchanged code. `analyze` reads existing outputs and joins all pairable records using actual condition/content keys. `summarize` aggregates and jointly resamples whole orbits 10,000 times for the fixed six models; models are neither resampled nor retrained. `validate` independently reads original caches by q and (r,H), recomputes the fixed small sample and verifies old-file hashes. Freezing applies only to this analysis, not retrospectively to earlier findings.

All six seeds 101–106 are retained. The main comparison contains 306 nonself directions; the arrays retain an additional 18 self-directions as implementation controls. Conditions (r,H) and contents come from actual natural-state caches, not assumed row correspondence.

## Actual donor provenance

- B1 patch: `../Final_audit/B1/seed_*/translation_d*_S*.npz`.
- B1 native: `../runs/v1_3/seed_*/analysis/baseline_d*_S*.npz`; members/conditions come from `behavior_states.npz` in the same directory.
- B_reserved patch/native: `../Final_audit/B_reserved/seed_*/translation_d*_S*.npz` and `baseline_d*_S*.npz`; members/conditions come from `states.npz` in the same directory.
- Untranslated patches with original S pair with donor S; translated patches pair with donor T(S): NA→NB, NB→NA, NN→NN. The donor's own history and p_a are checked using state caches; q matches the original manifest.

The primary native field is saved `native_logp`, the native continuation from (m_a,c_a). `natural_logp` is an uninterrupted control and is not substituted for the primary array. Missing donor predictions are not filled with truth and networks are not run to replace missing caches. Files, fields, hashes and actual q join positions are in `analysis_plan.json` / `input_index.json`.

## Compact per-query records and difference lookup

`data/{B1,B_reserved}/seed_*/` retains all query comparisons:

- `flags.npy`: uint16, axes `[d/S cell, treatment, nonself pair, canonical content, query_type, query_item]`. Cell order is d0/NN,NA,NB followed by d10/NN,NA,NB; treatments are original,translated. Pairs are explicitly listed in `analysis_plan.json`; contents are uniquely identified by membership dictionaries and lexicographic permutation IDs.
- `probability_differences.npy`: the same axes plus `[max_absolute_probability_difference,TV]` as the final axis, float64; unavailable values are NaN.
- `answers.npy`: the same axes plus `[patch argmax,donor argmax,donor own truth,number of exactly equal class probabilities]`, uint8; missing values are 255. Exact element counts are interpretable only when probabilities are comparable.
- `orbit_blocks.npz`: orbit numerators and denominators for every direction/d/S/treatment/type; source queries are not treated as independent Bernoulli observations.
- `differences_d*_S*_{original,translated}.npz`: explicit records of answer disagreements or maximum probability difference >1e−5, including complete keys, both predictions and all 7 class probabilities, donor truth/error flags, maximum difference and TV. Smaller floating-point differences remain fully available in compact tensors and source-probability indices.

Flag bits: available=1, probability_available=2, agree=4, donor_correct=8, exact=16, patch_tie=32, donor_tie=64, close_1e6=128, close_1e5=256, close_1e4=512, patch_tie_known=1024, donor_tie_known=2048. Unavailable tie detection is not counted as absence of ties.

A reproducible decoder reconstructs complete semantic keys from each tensor's flat index and retrieves both probability vectors from existing source caches, without duplicating the probability library. For example:

```bash
python -m experiments.memory_interchange.donor_consistency.inspect_record \
  --dataset B1 --seed 101 --flat-index 0
```

The disagreement NPZ's `flat_index` can be used directly with this command. Other small nonzero differences can also be located using `probability_differences.npy`. `input_index.json` records each treatment's source files, fields, effective suffixes, condition/content joins and tensor positions. Probabilities use float64 exp(logp), checks of original row sums and row normalization; argmax consistently chooses the lowest tied class index.

## Tables and statistical definitions

- `seed_summary.csv`: per-seed and equal-weight six-seed summaries, original/translated, same/cross context, same/different history, and donor-correct/error subsets; also a combined-treatment description.
- `strata_summary.csv`: fixed full crossing of S/d/query type/context/history/treatment/donor subset, including impossible or unobserved empty cells; both original S and effective_suffix are saved.
- `answer_agreement_equal_cells`: equally weighted query rates within each direction×d/S×query-type cell of the fixed subset, followed by equal weighting of six seeds. `answer_agreement_pooled`: total matching answers / total queries. They agree for complete equal-sized all cells, but may differ for error subsets or missingness.
- Probability means retain pooled and equal_cells definitions. Medians and q95/q99 describe the pooled observed-query distribution, not averages of model medians. Complete 14/two 7-query-vector agreement measures identical predictions, not permutation correctness; invalid predictions are retained.
- `block_intervals.json` and two bootstrap NPZ files: resample 19/51 complete orbits, sharing each draw's orbit weights across all six models and retaining directions, treatments, d/S/query. Fixed seed=20261013; the generator is reset independently for each dataset; 10,000 replicates. Intervals condition on the six models and exclude retraining and unknown-model-population uncertainty.
- `pairing_audit.json` / `pairing_coverage.csv`: unique keys, join cardinality, present/missing/diagonal controls, input timing and model/membership coverage.
- `normalization_audit.json` / `NN_audit.json` / `supplemental_controls.csv`: finiteness/normalization, identical NN sources/results, and existing sham/native/full-state controls.
- `control_state_differences.csv`: raw c differences computed only from saved states; no new states are generated.
- `direct_index_validation.json`: semantic direct reads of the frozen small sample checked against vectorized joins.
- `preservation_check.json`: hashes of old models, members, reports, results and failure records; `artifact_manifest.json` hashes all new deliverables.

The original numerical note remains: GPU controls passed; the original strict CPU float32 logit tolerance failed; identical coefficients in float64 agreed and argmax was unchanged. The 1e−6/1e−5/1e−4 probability thresholds only describe differences; they neither replace the logit check nor define scientific success. Work stops after the cache audit; no external publication.
