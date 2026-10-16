> English edition · [Frozen original](../../../../../experiments/memory_interchange/v1_1/CHANGES.md) · [English index](../../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

# v1.1 changes

v1.0 source, reports, configurations, checkpoints and data remain in place. The 101 original artifacts are also archived with SHA-256. v1.1 adds only this package, `runs/v1_1/` and `results/v1_1/`; original splits and 134 formal B contents are unchanged.

- Add an oracle on actual collated inputs, reconstructing answers only from stimuli, visible context/rules and final queries; independently check labels, input repetition, effective steps, query duration and padding.
- Complete old best/latest behavior on fixed T, balanced V and unchanged original V, including stimulus-removal/shuffling responses.
- Add 128-trial actual original-sampler path checks, retaining eager recurrent-node gradients and comparing actual fused-forward logits/stimulus/parameter derivatives. Add short-trial CPU float64 autograd/finite-difference checks. No original training-forward bug was found.
- Add three timing schedules and seven curriculum stages, final-query CE only, fixed 128/32/64 architecture. A uses default initialization; sole fallback B initializes both GRUs' effective update biases to +3, with all parameters still trainable.
- Generate/save fixed stage T/V and tiny-fit sets before any new training. Reuse every field of original 8,192-trial V. Additional balanced V is a separate diagnostic and cannot replace stage 7's original V.
- Save scratch separately without curriculum weight transfer. Advancement restores matching checkpoint weights, Adam and all RNG. Fix interruption recovery potentially continuing after an advancement point/failure cap; this fix did not change this version's uninterrupted optimization path.
- Add state sensitivity, gate distributions, stratum counts, per-stage diagnostics and costs. Audit additions only observe/check consistency; they do not change optimizer, loss, sampling or already-run model parameters.
- Keep original 97% total/95% factor gates, requiring two consecutive evaluations spaced 500 updates apart. Operation factors use the actual I/C sequence after removing N, alongside context, d, query type and non-N count; sparse fine crossed cells are not gates.
- Formal G/B inputs, geometry–damage correlations and scientific predictors remain inaccessible throughout. Only a future TASK_READY pilot could freeze configuration/source and permit a separately authorized formal experiment; this version does not run one automatically.

Official [GRU documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html) r/z/n order and retention equations were checked against local torch 2.11.0+cu128 source. The original PyTorch version was not upgraded. Gradient-mode review follows [Autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html) and verifies actual forward/derivatives.
