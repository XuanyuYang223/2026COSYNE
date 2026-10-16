# Memory-state interchange: English documentation

Complete English editions of all versioned reports, protocols and instructions are indexed here. Each links its frozen original. Numerical results, commands, program identifiers, data and figures retain their definitions. Historical dates, execution scope, status and publication statements describe the corresponding original version.

## Latest reports and instructions

| Work | Report | Instructions |
| --- | --- | --- |
| State interchange versus donor-native outputs | [Consistency audit](experiments/memory_interchange/donor_consistency/report.md) | [Execution and data guide](experiments/memory_interchange/donor_consistency/README.md) |
| v1.3 three-part Final audit | [Final audit report](experiments/memory_interchange/Final_audit/report.md) | [Execution and artifacts](experiments/memory_interchange/Final_audit/README.md) |
| v1.3 formal experiment | [Formal report](experiments/memory_interchange/results/v1_3/report.md) | [Version guide](experiments/memory_interchange/v1_3/README.md) |

Donor-output agreement compares predictions with actual donor predictions. Donor-rule agreement compares predictions with mathematical truth. Cue-translation recovery evaluates the original receiver target. These quantities remain separate. Both B datasets had already been opened before the latest consistency audit.

## Detailed results

- [Interpretation and numerical stability](experiments/memory_interchange/results/v1_3/interpretation.md)
- [All formal-model native behavioral factors](experiments/memory_interchange/results/v1_3/all_behavior_factors.md)
- [Native-quality and conditional-mask sensitivity](experiments/memory_interchange/results/v1_3/quality_and_mask_summary.md)
- [Current-content versus initial-input counterfactuals](experiments/memory_interchange/results/v1_3/counterfactual_all_and_distinct.md)
- [Methodological context](experiments/memory_interchange/results/v1_3/method_context.md)
- [Pilot validation-set exploration](experiments/memory_interchange/results/v1_3/V_exploration/report.md)
- [Donor-consistency abstract sentence](experiments/memory_interchange/donor_consistency/english_result.txt)
- [Terminology and data-field glossary](GLOSSARY.md)

## Historical versions and protocols

| Version | Guide | Report | Protocol and changes |
| --- | --- | --- | --- |
| v1.0 | [Original experiment](experiments/memory_interchange/README.md) | [Historical BLOCKED report](experiments/memory_interchange/report.md) | [Original protocol](experiments/memory_interchange/protocol_v1.txt) |
| v1.1 | [Version guide](experiments/memory_interchange/v1_1/README.md) | [Historical BLOCKED report](experiments/memory_interchange/results/v1_1/report.md) | [Protocol](experiments/memory_interchange/v1_1/protocol_v1_1.txt); [changes](experiments/memory_interchange/v1_1/CHANGES.md) |
| v1.2 | [Version guide](experiments/memory_interchange/v1_2/README.md) | [Stage-seven BLOCKED report](experiments/memory_interchange/results/v1_2/report.md) | [Changes](experiments/memory_interchange/v1_2/CHANGES.md) |
| v1.3 | [Version guide](experiments/memory_interchange/v1_3/README.md) | [Formal report](experiments/memory_interchange/results/v1_3/report.md) | [Protocol](experiments/memory_interchange/v1_3/protocol_v1_3.txt); [changes](experiments/memory_interchange/v1_3/CHANGES.md) |

## Provenance and maintenance

Original Chinese audit documents and report-generator strings are retained as frozen scientific records. English editions translate their complete saved reports and instructions without changing scientific code, hashes or results. JSON/CSV schemas and values remain at their original paths; terminology is explained in the glossary.

[translation_manifest.json](translation_manifest.json) records original and English file hashes. [The builder](../../tools/build_english_docs.py) uses a reviewed [English translation dictionary](../../tools/english_translations.json) keyed by source-text hashes. Numerical placeholders are checked before restoration. Unknown or changed text fails explicitly rather than silently leaving untranslated content.

```bash
python tools/build_english_docs.py
python tools/build_english_docs.py --check
```

See the [repository README](../../README.md) for binary-artifact availability. Historical commands may train or run networks; this documentation conversion runs no model and recomputes no scientific result.
