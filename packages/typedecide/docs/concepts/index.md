# Concepts

The library is small, and most of it follows from seven ideas. Each page below states
the idea, shows where it lives in the code, and says what goes wrong without it.

| Concept | One-line version |
|---|---|
| [The typed logit readout](readout.md) | Read a softmax over answer-letter token ids at one position. Do not generate. |
| [Answer-slot discovery](answer-slots.md) | The letter token ids are probed from the real tokenizer at the real prompt boundary, never hard-coded. |
| [Shared-prefix amortisation](shared-prefix.md) | The state is identical across criteria, so its KV cache can be computed once. |
| [Position priors and debiasing](position-priors.md) | Small models favour a slot. Averaging over orderings repairs the readout, not the model. |
| [Group-aware splitting](group-splitting.md) | Split on the thing that must not leak, which is the state, not the row. |
| [Answer-token-only loss](answer-token-loss.md) | One supervised token per example; long states are truncated from the left. |
| [Prompt parity](prompt-parity.md) | Python and JavaScript must build byte-identical prompts. |

## Vocabulary

State
:   The evidence a decision is about. Rendered under `Evidence:` in the prompt.

Criterion
:   A question plus its closed set of options. Identified by a `key`.

Decision
:   One criterion asked about one state, optionally with the gold `answer_id`.

Head / tail
:   The prompt is `head_text(state) + tail_text(criterion)`. The head is shared by
    every criterion asked about a state. The tail is the criterion's own text and
    ends in `Answer:`.

Answer slot
:   The token id that an answer letter encodes to at the position right after
    `Answer:` for a particular tokenizer.

Readout
:   The softmax over a criterion's answer slots at that one position.

Ordering
:   A permutation of a criterion's options. The `answer_id` does not change when the
    ordering does.
