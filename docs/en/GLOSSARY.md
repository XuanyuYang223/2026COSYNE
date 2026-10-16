# Terminology, data fields and statistical definitions

English editions translate prose and display labels. Actual paths, interfaces, JSON/CSV columns, array values and scientific definitions remain unchanged.

## States and treatments

| Identifier | Meaning |
| --- | --- |
| `m` / `memory` | Designated memory component; may also carry context, history or initial input. |
| `c` / `control` | Designated control component; retaining it does not establish retention of the receiver's functional rule. |
| `donor` / `a` | Condition supplying the exchanged memory state. |
| `receiver` / `recipient` / `b` | Condition whose control component is retained. |
| `native` | Continuation from a saved native state. |
| `natural` | Uninterrupted output, distinct from the saved-state `native` continuation. |
| `patch` / `matched` | Memory replacement using a donor with matching complete content. |
| `wrong` | Replacement using incorrect donor content. |
| `sham` | Receiver self-replacement control. |
| `full` | Full-state control; interpretation requires identical actual future inputs. |
| `original` | Untranslated continuation in the consistency audit; original timing in curricula. |
| `translated` | A/B exchanged and N unchanged: NN→NN, NA→NB, NB→NA. |
| `same` / `cross` | Same/cross context in context strata; same history in history strata. |
| `different` | Different histories. |

## Content, cues and queries

| Identifier | Meaning |
| --- | --- |
| `r` / `context` | Rule context, 0 or 1. |
| `H` / `history` | Visible cue history before interchange. |
| `S` / `suffix` | Original receiver literal suffix, retained as the stratum label. |
| `effective_suffix` | Actual continuation; translated NA uses NB and vice versa. |
| `q` / `content_id` | Shared current complete task content and identifier. |
| `p_a` | Donor initial permutation, `p_a=G_a^{-1}(q)`. |
| `d` / `delay` | Extra retention, 0 or 10. |
| `I` / `inverse` | Inverse permutation. |
| `C` / `complement` | Position mirroring. |
| `N` / `KEEP` | No update. |
| `true_operation_sequence` | Actual I/C order after removing N, such as IC or CI. |
| `non_N` | Number of non-N operations. |
| `query_type=0` / `item` | Given an item, predict its position. |
| `query_type=1` / `position` | Given a position, predict its item. |
| `query_item` / `query_id` | Queried item or position index. |
| `orbit_id` / `orbit` | Complete I/C closure orbit, the block-resampling unit. |

In context 0, A/B mean I/C; in context 1 they mean C/I. Visible AB cannot be interpreted as IC without context. Conditions list nine histories for context 0, then the same order for context 1: `NN,NA,AN,NB,BN,AA,BB,AB,BA`.

## Samples and masks

| Identifier | Meaning |
| --- | --- |
| `all` | Every available query; primary description. |
| `rec` | Receiver native correct for original S. |
| `both` | Receiver/matched donor each correct for the same visible original S under their own rules. |
| `three` | `both`, additionally requiring the wrong-content donor to be correct on its own task; answers need not coincide. |
| `E_sem` | Receiver correct for original S; donor correct for the same semantic operation, using translated donor S across contexts or original S within context. |
| `mask` | Sample mask frozen from native outputs before replacement. |
| `T` / `V` / `G` / `B` | Training, task validation, geometry and intervention-behavior orbit splits. |
| `B1` | Original selected 134 contents / 19 complete B orbits. |
| `B_reserved` | Remaining B-minus-B1: 362 contents / 51 orbits; already opened before the latest consistency audit. |

Use each version's own frozen masks. One native donor output may pair with multiple receivers; expanded counts are not independent native-error samples.

## Outputs and statistics

| Identifier | Meaning |
| --- | --- |
| `logp` | Natural-log probabilities; the consistency audit converts to float64 probabilities, checks finite values/row sums and normalizes. |
| `logit` | Unnormalized score; a common shift does not change probabilities. |
| `argmax` | Highest-scoring class; lowest class index on ties. |
| `CE` / `nat` | Cross-entropy / natural-log unit. |
| `TV` | Half the sum of absolute class-probability differences. |
| `max_abs_probability_difference` | Largest absolute class-probability difference per query. |
| `answer_agreement_equal_cells` | Equal nonempty direction×d/S×query-type rates, then equal model weights. |
| `answer_agreement_pooled` | Matching-answer total / pairable-query total. |
| `q95` / `q99` | 95th / 99th percentiles. |
| `item7` / `position7` | All seven item / position predictions agree. |
| `MAE` / `RMSE` | Mean absolute error / root mean squared error. |
| `CKA` | Centered kernel alignment. |
| `D_raw` | Normalized native-coordinate distance. |
| `D_mean` / `D_center` | Mean-offset / centered components of native distance. |
| `Procrustes` | Orthogonal Procrustes alignment. |
| `ridge` / `alpha` | Ridge regression / regularization coefficient. |
| `LOSO` / `fold` | Leave-one-seed-out evaluation / held-out fold. |
| `bootstrap` | Resampling; latest consistency intervals fix six models and jointly resample complete orbits. |
| `refit bootstrap` | Refit predictors per resample, distinct from fixed-prediction conditional intervals. |
| `pp` | Percentage points. |

Answer agreement, probability closeness and exact equality remain separate. Full fourteen-query agreement does not require bijections or correct complete permutations. Donor errors use the donor's own truth under its actual effective suffix.

## Files and status

| Identifier | Meaning |
| --- | --- |
| `seed` | Model-initialization or resampling RNG seed, according to context. |
| `pilot` | Pre-formal model establishing task readiness. |
| `initial` / `best` / `latest` / `advance` | Initial / best / latest / advancement checkpoint. |
| `selected_stage7.pt` | Selected frozen formal stage-seven checkpoint. |
| `RNG` | Random-number-generator state. |
| `manifest` / `SHA-256` | Membership/artifact inventory / integrity hash. |
| `float32` / `float64` | Single / double precision. |
| `TASK_READY` | Pilot passes its version's full task; scientific analysis remains incomplete. |
| `COMPLETED` | Prescribed scope finished; actual gates and numerical failures must still be reported. |
| `EXPLORATORY` | Exploratory findings or insufficient formal samples/checks. |
| `PARTIAL` | Incomplete scope with documented missing items. |
| `BLOCKED` | A version-specific implementation, learning or budget stop prevents completion. |

Retained numerical note: GPU controls passed; original strict CPU float32 logit tolerance failed; identical coefficients in float64 agreed and original compared argmax outputs were unchanged. Later probability tolerances do not replace that check.
