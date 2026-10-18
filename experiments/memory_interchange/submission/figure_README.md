# Stage 1: independent figures for review

This stage redraws four independent figures only. The manuscript, Summary, references, combined figure and two-page PDF remain unchanged. The new 297-word Summary and six-reference revision belong to the later LaTeX stage and have not been applied. No Git commit/push or conference submission was performed in this stage.

## Review files

| Logical figure | PDF for later LaTeX | Editable SVG | PNG preview | Design size (inches) |
| --- | --- | --- | --- | --- |
| A: task and exchange | [PDF](figures/panel_A.pdf) | [SVG](figures/panel_A.svg) | [PNG](figures/panel_A.png) | 7.20 × 7.50 |
| B: exchange errors | [PDF](figures/panel_B.pdf) | [SVG](figures/panel_B.svg) | [PNG](figures/panel_B.png) | 3.45 × 3.50 |
| C: cue translation | [PDF](figures/panel_C.pdf) | [SVG](figures/panel_C.svg) | [PNG](figures/panel_C.png) | 3.45 × 4.10 |
| D: geometry distribution | [PDF](figures/panel_D.pdf) | [SVG](figures/panel_D.svg) | [PNG](figures/panel_D.png) | 3.45 × 4.45 |

[captions.txt](captions.txt) gives four separate English captions, symbol definitions, statistical units and necessary methods. There are no burned-in A–D identifiers or long panel titles. A uses short numbered steps; C uses the descriptive dataset label “Held-out permutations”. Source identifiers remain in the captions and this document.

A is a new three-step drawing with a separate context-rule table and a direct donor-memory-to-receiver-slot arrow. Numerical permutation examples are in the caption and are checked by pure list operations in the plotting code. There is no outer network frame or architecture-input cartoon. B uses identical circular seed points and short black mean lines. C uses identical circular endpoints, fine paired network traces and thick mean trends, without seed-identity marker codes. D places its legend above the axes, so it does not cover either cumulative curve. Blue/orange consistently denote within/cross context in B/D.

## Generation commands

Use the existing Python environment recorded in [environment.json](environment.json), with NumPy, Matplotlib, CairoSVG, PyMuPDF and the Cairo library. No LaTeX tool is used during this stage. From the repository root:

```bash
python experiments/memory_interchange/submission/plot_figure.py --panel A
python experiments/memory_interchange/submission/plot_figure.py --panel B
python experiments/memory_interchange/submission/plot_figure.py --panel C
python experiments/memory_interchange/submission/plot_figure.py --panel D
```

`--panel all` regenerates all four independent figures, with no assembly or manuscript compilation. The functions `schematic`, `behavior`, `translation`, and `geometry` each accept `width` and `height` in inches. To change one design size, for example:

```bash
python experiments/memory_interchange/submission/plot_figure.py --panel B --width 3.45 --height 3.8
```

Font sizes remain fixed in physical points when dimensions change; text is never scaled down. The script rejects clipped or overlapping text. Re-export and review the layout at any new target width before LaTeX inclusion. The large A design is intentionally independent; later placement must use an appropriate redesign/export rather than shrinking this PDF below the minimum font size.

Use this script for the current stage. The original [README.md](README.md), [build.py](build.py), [validation.json](validation.json) and [artifact_manifest.json](artifact_manifest.json) describe the earlier combined manuscript and have deliberately not been updated. Running the legacy build commands would enter manuscript work, which is outside the present authorization. Current figure validation is in [figure_validation.json](figure_validation.json).

## Physical typography and rendering

All ordinary diagram text, axes, ticks, legends and annotations are **12 pt or larger at the recorded design sizes**. A step headings are 13.5 pt; mathematical base characters are 12–14 pt and true subscripts use normal reduced sizes (8.64–10.08 pt). B mean labels are 13 pt. B/C/D ordinary text and C/D annotations are 12 pt. The common font is DejaVu Sans, embedded in each vector PDF. All SVG text remains editable. Node outlines/axes are approximately 0.9 pt; flow arrows are 1.1 pt, the A memory arrow 1.6 pt, distribution curves 1.7 pt, and B/C mean marks 2.1/2.3 pt. C network traces are 0.7 pt.

PDFs contain vector paths and embedded fonts, with no image objects. PNGs are independent 144-dpi previews rendered from the final vector PDFs. They are not embedded back into the PDFs. All ordinary text was checked at physical design size and at enlarged view; mathematical subscripts were inspected separately. Anonymity checks cover visible text, metadata and absent XMP. The metadata uses generic creator/producer fields and empty author/date/keywords fields.

## Data and unchanged statistics

The existing data exports are read, not regenerated or edited. The script verifies all 118 prior source hashes in [data/source_manifest.json](data/source_manifest.json), and verifies the plot-input hashes from the legacy manifest. [data/AUDIT.md](data/AUDIT.md) retains the complete earlier source/filter/denominator audit and six-seed tables. The visual redraw does not change those tables.

| Figure | Immediate input | Selection and aggregation | Actual displayed aggregate |
| --- | --- | --- | --- |
| B | `data/panel_B_seed.csv` | All six seeds; cross=0/1; equal network weights; original pair/delay/suffix/direction/query weights | 1.696975604780331% / 53.39794999137177% |
| C | `data/panel_C_seed.csv` | Six paired networks per treatment; original/translated cues; original receiver target; actual native control; all suffixes including NN | Exchange 42.06742012335935% → 97.85815176967725%; native 97.85817206972824% → 42.06742012335935% |
| D | `data/panel_D_d0.csv` | d=0, a<b; unique (seed,a,b,d); 432 within / 486 cross measurements; pooled distribution | Medians 0.8919304489849997 / 0.8886644081851174 |

B primary saved results are `../runs/v1_3/seed_{101..106}/analysis/labels.csv`, columns `matched_R_all` and `n_all`, checked against the original direction-level summaries and `../results/v1_3/prediction/full_distribution.csv`. There are 72/81 unordered within/cross condition pairs, two delays, three suffixes, both exchange directions, 134 B1 contents, and 14 queries per content. The all-query denominators per network are 1,620,864 within and 1,823,472 cross. New errors count native-correct/exchange-wrong answers over these full denominators. No post-exchange selection is applied. All six points, including seed 106, are retained in each group.

C primary saved exchange CSVs are `../Final_audit/B_reserved/seed_{101..106}/translation_d{0,10}_S{NN,NA,NB}.csv`, selecting `cross=1, mode=all, query_type=-1`, with no self-directions. Use `original_accuracy_original` / `translated_accuracy_original`; 972 equal-weight direction×delay×suffix cells, 4,926,096 expanded queries per seed and cue treatment. The native control comes from the corresponding `native_d*_S*.csv`, selecting `treatment=native/native_translated, query_type=-1`, column `original_target_accuracy`; 108 equal-weight receiver×delay×suffix cells and 547,344 native queries per seed and cue treatment. Per-seed values were rechecked against `translation_seed_summary.csv` and `native_seed_summary.csv`. No same-context exchange is substituted for native activity. The cohort has 362 permutations / 51 complete orbits. The native/exchanged exact differences are retained; no seed points or intervals are invented.

D primary saved geometry is `../Final_audit/geometry_decomposition.csv`, already verified against each seed's `analysis/geometry.csv`, `geometry_summary.json` and `geometry_distributions.csv`. There is one d=0 value per network and unordered condition pair, calculated earlier over 492 G contents. The 72/81 pair measurements per network are dependent; they do not constitute 432/486 independent networks. Delays are not pooled and repeated suffix entries are not counted.

Network markers in B/C use deterministic horizontal offsets in seed order 101–106; the offsets improve visibility and do not encode a new marker category or change measured y-values. C has twelve fine paired traces because the same six networks appear in two treatments, with 24 circular endpoints. No error bars, refitting, bootstrap, training or model forward passes are performed.

## Preservation and stage boundary

The preceding plotting script, panel vector assets, previous panel previews and manuscript-era manifests were copied byte-for-byte to [archive/pre_stage1_2026-10-17](archive/pre_stage1_2026-10-17). The 30 pre-existing files outside the allowed panel/script edits have SHA-256 protection recorded in [protected_files.json](archive/pre_stage1_2026-10-17/protected_files.json). This includes submission.tex, summary.txt, summary.tex, submission.pdf, figures/figure_1.pdf, existing data, manuscript previews and the prior README/validation. The script verifies them before and after rendering. The old combined figure and two-page PDF are unchanged and are not the current single-figure preview delivery.

No second-stage edits were made. The next step is user review of these four separate figures; Summary synchronization, architecture prose, the six verified references, standard LaTeX citations, final panel numbering, and two-page arrangement remain pending that review.
