# Anonymous COSYNE submission

The [two-page PDF](submission.pdf) and [LaTeX source](submission.tex) reuse saved results from the six frozen networks. No training, model forward passes, scientific-output changes, predictor refits, or new bootstrap were performed. No conference submission was made.

## Files

- [Four-panel vector figure](figures/figure_1.pdf); individual vector PDFs: [A](figures/panel_A.pdf), [B](figures/panel_B.pdf), [C](figures/panel_C.pdf), [D](figures/panel_D.pdf).
- [Editable schematic SVG](figures/panel_A.svg); [Matplotlib and SVG plotting script](plot_figure.py).
- [Summary, plain text](summary.txt): **288 words**, one paragraph, unchanged from the supplied version. [summary.tex](summary.tex) only escapes typography for LaTeX.
- [Full source audit, filters, denominators and six-seed tables](data/AUDIT.md); [source paths and SHA-256 hashes](data/source_manifest.json); [exact aggregate values](data/actual_values.json).
- [Automated validation](validation.json), [software versions](environment.json), and rendered [page 1](preview/page_1.png) / [page 2](preview/page_2.png). PNG files are previews only; all publication artwork is vector.

## Reproduce

Run from the repository root. Required Python packages are NumPy, Matplotlib, CairoSVG, pypdf and PyMuPDF; the recorded versions are in environment.json. CairoSVG also requires the Cairo library. Install a Tectonic executable on PATH (the original build used Tectonic 0.17.0 in an external temporary conda environment).

```bash
# Read existing summaries, verify source hashes and export figure inputs.
python experiments/memory_interchange/submission/audit_data.py
# Draw all panels and assemble the vector figure.
python experiments/memory_interchange/submission/plot_figure.py
# Complete build: regenerates inputs/figure, compiles, anonymizes and renders.
python experiments/memory_interchange/submission/build.py
```

If Tectonic is outside PATH, pass `--tectonic /path/to/tectonic` to build.py. The exact executable used here was `/tmp/cosyne-publication-tools/bin/tectonic`. The internal compilation command is `tectonic --keep-logs submission.tex` in this directory. Use build.py for the final PDF because it also checks page count, removes optional metadata, and renders both pages. The delivered summary.tex permits direct LaTeX compilation as well.

Plotting requires only the existing CSV/JSON inputs listed in source_manifest.json, all included in the repository. Large model/output caches are not needed for this publication build. It does not import project model modules or torch. Tectonic may download its standard TeX support files on first compilation.

## Values and statistical units

| Panel | Actual plotted aggregate |
| --- | --- |
| B, all-query new errors | Within 1.69697560%; cross 53.39794999% |
| C, cross-context exchange | 42.06742012% → 97.85815177% |
| C, native no exchange | 97.85817207% → 42.06742012% |
| D, d=0 pooled CKA median | Within 0.891930448985; cross 0.888664408185 |

These agree with the requested rounded values. The small exact differences between native and exchanged results are preserved. B/C show six independently trained networks and equal-seed means; no confidence intervals or synthetic seed points are added. D pools dependent condition-pair measurements, 432 within / 486 cross, across those six networks, deduplicated across suffixes at **d=0 only**. See data/AUDIT.md for all source CSVs, selections, per-seed values and denominators. C uses actual unpatched native trials for the no-exchange control and includes NN. Donor-output agreement and the uncertain predictor result appear in text only; wrong-content rescoring is not presented as an independent new experiment.

## Print and anonymity checks

The final PDF is exactly **two US Letter pages**, with **0.6-inch margins**. Body type is 12.1 TeX pt (12.055 PDF pt); captions/references are 10.1 TeX pt (10.062 PDF pt). Mathematical subscripts use normal script sizing. At its printed width of 7.3 inches, the figure is 6.333 inches high; panel titles are 11.3 pt, axes/ticks 10 pt, and annotations/legends 9–9.5 pt. All figure text remains vector, and the schematic SVG keeps editable text.

Both final pages were rendered at 144 dpi and visually inspected at print scale and enlarged. The final review checked clipping, overlapping labels, legends, the schematic's single-network framing, all six seed markers, and reference readability. D's legend/median annotation and x-label/footer overlaps were corrected. The Summary's curly apostrophes are emitted through equivalent TeX typography to avoid missing glyphs; the plain-text Summary is untouched. The final build has **0 overfull boxes, 0 underfull boxes, 0 missing glyph warnings**, and no raster image objects. Additional Detail contains 298 whitespace tokens in normalized PDF text (including mathematical tokens); the submission limit applies to the 288-word Summary.

PDF body, figure text, document metadata and XMP were checked for author names, affiliations, acknowledgments, user paths, and email addresses. Author/keywords/date fields are empty, metadata has only a generic creator/producer, and XMP is absent. Names in the two verified scholarly references identify cited authors only. The manuscript contains no repository or personal links. Reference metadata was verified against the primary PMLR and arXiv/ICLR pages linked in data/AUDIT.md.

The prior wrong-content update was independently validated with its existing read-only validate.py before the requested GitHub update. That historical audit's source/results were preserved; the present authoring work lives here.
