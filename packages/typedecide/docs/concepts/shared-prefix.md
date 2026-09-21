# Shared-prefix amortisation

## The idea

Every criterion asked about one state shares the same opening text: the instruction,
the `Evidence:` header and the state. The prompt is built as two pieces for exactly
this reason:

```text
head_text(state)      "<INSTRUCTION>\n\nEvidence:\n<state>\n"          shared by every criterion
tail_text(criterion)  "\nCriterion: ...\nOptions:\nA. ...\nAnswer:"     this criterion only
```

A decoder's key/value cache for the head depends only on the head. So it can be
computed once (the *prefill*) and reused for every tail. The eleventh question about a
ticket then costs a short suffix, not another full prompt.

```text
                    +-- "which queue?"      --> logits[A,B,C]   --> softmax
state (prefilled    +-- "how urgent?"       --> logits[A,B,C,D] --> softmax
 once, reused)      +-- "is this angry?"    --> logits[A,B]     --> softmax
   KV cache         +-- "needs approval?"   --> logits[A,B,C]   --> softmax
```

## The one-token trim

The cached prefix is the head's token ids **minus the last one**.
[`state_prefix_ids`](../reference/api/prompt.md) drops it because the character that
follows the state can merge into the head's final token. If it did, the head would no
longer be an exact token-level prefix of the full prompt and the cache would describe
different tokens from the ones being scored. `web/engine.js` performs the same trim,
then verifies that the prefix really is a token-level prefix of each full prompt and
throws if it is not.

## Where this is implemented, precisely

!!! warning "The Python evaluator does not reuse the prefix"
    Shared-prefix reuse is implemented in the **browser engine**
    (`web/engine.js`: `prefill()` then `score()` per criterion). The Python
    `typedecide.evaluation` path scores every (decision, ordering) pair as a complete
    prompt in left-padded batches. That is the right trade on a GPU with batching
    available, and it means `typedecide evaluate` timings say nothing about browser
    latency. `prompt.state_prefix_ids` exists so Python code can compute the same
    prefix the browser caches; the evaluator itself does not call it.

## What has been measured

**Reuse is exact in fp32.** `bench/shared_check.mjs` scores the same criteria twice,
once from a full prompt each time and once against a prefilled state, and reports the
maximum probability drift. The repository README records `0.000e+0` for
`onnx-community/Qwen3-0.6B-ONNX`. (This site quotes that figure from the README; no
result file for it is committed under `bench/results/`.)

**Per-criterion cost is linear in suffix length.** From
`bench/results/logit-cost-Qwen3-0.6B-q4f16.json` (onnxruntime-node on CPU, prefix held
fixed at 35 tokens): the fit is 28.4 ms fixed plus 3.23 ms per suffix token, which
projects to about 174 ms for a typical 45-token criterion on that backend. The file's
own note says the point is the shape, not the intercept, and that it is not comparable
to WebGPU figures.

**Batching the criteria is not possible in the browser.** ONNX Runtime Web rejects it:

```text
GroupQueryAttention: batch_size must be 1 when sequence_length > 1 and past context is given.
```

So in the browser, criteria are scored one after another. The prefill buys a shorter
prompt per decision, not a single forward pass for all of them.

## Why this makes debiasing cheap in the browser

Reordering a criterion's options changes only the tail. Against a prefilled state,
each extra ordering costs one short suffix. A generation-based system would need a
complete request per ordering. This is the reason cyclic averaging is affordable
there. See [Position priors and debiasing](position-priors.md) and, for the
performance levers, [Making the readout fast](../SPEED.md).
