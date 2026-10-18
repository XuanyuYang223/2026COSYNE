# Wrong-content recombination audit

This is a descriptive, cache-only rescoring of the six frozen formal models' original v1.3/B1 wrong-content interventions. It does not run networks, train models, refit predictors, change historical outputs or use B_reserved. The Chinese result report is `report.md`.

From the repository root, using the existing NumPy/PyYAML environment:

```bash
python experiments/memory_interchange/wrong_content_recombination/analysis.py prepare
python experiments/memory_interchange/wrong_content_recombination/analysis.py score
python experiments/memory_interchange/wrong_content_recombination/report.py
python experiments/memory_interchange/wrong_content_recombination/analysis.py verify
python experiments/memory_interchange/wrong_content_recombination/validate.py
```

`prepare` inspects file hashes, NPZ headers and non-output metadata only. It freezes `analysis_plan.json`, code hashes and mathematical-candidate-defined direct-index samples before reading saved wrong/matched/native output values. Existing plans are immutable; do not rerun `prepare` over a completed audit. `score` requires all six primary input sets and verifies their provenance; it never substitutes mathematical truth for native predictions. `verify` checks frozen code/plan and pre/post hashes of all tracked historical files plus input caches.

For recipient content index j, actual donor index is k=`mismatch_index[j]`. The three mathematical candidates are donor-rule truth for q[k], receiver-rule truth for q[j], and receiver-rule truth for q[k]. The primary subset contains cross-context queries where those three answers are pairwise distinct. It is unrelated to the original correctness-based `three` mask. All cross-context queries are also reported, with overlapping candidate agreement explicitly nonexclusive.

Rates pool eligible queries within each seed, then weight the six seed rates equally. Counts and pooled rates are separate. Empty strata have n=0 and rate=null. If a stratum is empty for any seed, its strict six-seed mean is null; an available-seed mean is separately identified. No confidence intervals, p-values or resampling are computed.

The core outputs are `overall.csv`, `seed_summary.csv`, `strata_summary.csv`, `candidate_overlaps.csv`, `index_checks.csv`, `consistency_checks.json`, `direct_index_checks.json`, `control_state_pairs.csv`, `control_state_summary.csv`, `donor_query_reuse.csv`, `summary.json` and the preservation checks. Donor-query usage NPZs have axes `[d,S,D,donor_content_k,query_type,query_item]`; counts specify how many recipients reuse each native prediction. They are counts, not newly computed network outputs.

`report.py` renders the Chinese report and English result sentences from saved summaries. Its category-by-donor table is an algebraic cross-check using the frozen category counts and the complete list of native-disagreement records, without rescoring or changing selection. The numerical count of unique donor errors uses the saved donor-query reuse histogram. The scoring source and plan remain frozen.

`validate.py` independently reads delivered CSVs to check partitions, denominators, strict six-seed means, empty NN strata, and category/native cross-counts; it also verifies the fixed direct-index result, complete-report status, links and new-artifact hashes. It performs no model execution or further scientific estimation.

The same-start implementation comparison uses saved wrong[D,R,j] versus matched[D,R,k], over all stored directions and cross-context directions separately. Probabilities are computed in float64 from log probabilities, checked and row-normalized. Probability thresholds describe differences; they do not replace the original failed strict CPU float32 logit check. Original GPU controls passed, identical float64 coefficients agreed and original compared argmax answers were unchanged.

All new files remain in this directory. No Git commit, push or manuscript modification is part of this audit.
