# 2026COSYNE: memory-state interchange

This repository contains the experimental code, protocols, configuration, learning records, reports, summary tables, figures and integrity manifests for the memory-state interchange study.

The latest completed work is the descriptive comparison of memory patches with donor-native outputs in six frozen RNNs. B1 and B_reserved were both previously opened; this audit is not an independent confirmation experiment. The original CKA prediction test and previous numerical failures remain unchanged.

## Reports

- [Donor-native output consistency, 2026-10-13](experiments/memory_interchange/donor_consistency/report.md)
- [v1.3 Final audit](experiments/memory_interchange/Final_audit/report.md)
- [v1.3 formal experiment](experiments/memory_interchange/results/v1_3/report.md)
- [v1.3 interpretation](experiments/memory_interchange/results/v1_3/interpretation.md)
- [v1.2 report](experiments/memory_interchange/results/v1_2/report.md)
- [v1.1 report](experiments/memory_interchange/results/v1_1/report.md)

Earlier BLOCKED reports are historical results, not the status of the latest audit. All six prescribed seeds, including 106, are retained in the reports and run metadata. The registered GPU controls passed, but the original strict CPU float32 logit check failed; the same coefficients in float64 agreed and the compared argmax outputs did not change.

## Code and reproduction

The experimental package is [experiments/memory_interchange](experiments/memory_interchange). Versioned protocols and execution instructions are preserved alongside each version. The latest read-only audit has its own [README and commands](experiments/memory_interchange/donor_consistency/README.md); the prior Final audit has separate [execution instructions](experiments/memory_interchange/Final_audit/README.md).

**This Git repository is not the complete binary data delivery.** It includes all six formal models' course-start `initial.pt` and frozen `selected_stage7.pt` checkpoints (9.4 MiB combined), together with the source, text metadata, available tables and figures, including the original bootstrap records.

Approximately 9.57 GiB of other checkpoints, natural-state caches, original probability arrays, per-query comparison tensors and archived snapshots remain in the source workspace. They are excluded using `.gitignore` (`*.pt`, `*.npz`, `*.npy`, `*.tar.gz`, with explicit exceptions for the twelve formal initial/selected checkpoints). Historical files are preserved byte-for-byte, including CSV line endings.

To reproduce an audit of the frozen outputs, restore the exact binary artifacts at their original relative paths and verify their SHA-256 hashes before running the audit. Their filenames, members, input mappings and hashes are recorded in:

- [Donor consistency analysis plan](experiments/memory_interchange/donor_consistency/analysis_plan.json) and [input index](experiments/memory_interchange/donor_consistency/input_index.json)
- [Final audit artifact manifest](experiments/memory_interchange/Final_audit/artifact_manifest.json)
- [v1.3 artifact manifest](experiments/memory_interchange/results/v1_3/artifact_manifest.json)
- [Local artifacts excluded from Git](LOCAL_ARTIFACTS.json)

A fresh clone contains the twelve formal checkpoints but does not contain the required cached outputs and archived training states; it cannot, by itself, replay the cached-output audits. No binary artifact was deleted from the local workspace during upload. Uploading this snapshot does not authorize or trigger new training or additional scientific analyses.
