# Making the readout fast

This is the inference-speed guide for `typedecide.export`. It is written for
someone who has to decide what to spend a week on, so every claim carries a label
saying how much it is worth.

| Label | Means |
|---|---|
| **MEASURED** | A number this project actually observed on hardware. The source file is named. |
| **DERIVED** | Exact arithmetic from a published model config. No timing claim — FLOPs and bytes, not milliseconds. |
| **PROJECTED** | An estimate. The assumptions are stated and the arithmetic is shown. It may be wrong. |

All MEASURED numbers come from `bench/` and `bench/results/`: MacBook,
`onnx-community/Qwen3-0.6B-ONNX`, `q4f16`, unless the line says otherwise.
Nothing in this document is a measured wall-clock benchmark **of pruning or of
trimming** — those levers are derived and projected only. Say so if you quote
them.

## The short version

| Lever | Effort | What it buys | Evidence |
|---|---|---|---|
| Trim logits to the last position | one graph edit | 45x less logit compute and readback on a 45-token suffix | DERIVED |
| Prune the LM head to the answer letters | this module | Kills the remaining readback; 0 download saving on a tied model, ~209 MiB on Qwen3-4B | DERIVED |
| Keep the KV cache on the GPU | one session option | 63.9 MB of round-tripping per criterion at a 256-token prefix | DERIVED |
| Quantise to `q4f16` | one command | 4x smaller than fp16 | DERIVED size, **unmeasured accuracy cost** |
| Shorten the per-criterion suffix | prompt work | Linear in tokens; the whole per-criterion cost is the suffix | MEASURED shape, PROJECTED size |
| Batch the criteria | — | **Impossible in the browser.** Do not plan around it. | MEASURED |

Do them in that order. Trimming is the cheapest and largest single win;
pruning finishes the job; everything else is smaller.

---

## 1. The export returns logits for every position

A stock Optimum export of a decoder returns `logits` shaped
`[batch, sequence, vocab]` — the whole sequence, not the position you asked
about. A typed readout reads exactly one position (`web/engine.js` reads
`suffix.length - 1`). Everything else is computed, serialised, copied out of the
accelerator, and discarded.

**DERIVED.** Qwen3-0.6B, `vocab = 151936`, a 45-token criterion suffix (the
figure `bench/logit_cost.mjs` uses as typical), fp32 logits:

```
untrimmed:  45 x 151,936 x 4 B  =  27,348,480 B  =  26.1 MiB   per criterion
trimmed:     1 x 151,936 x 4 B  =     607,744 B  =   0.58 MiB  per criterion
```

Over one pass of the 144-row `authored144` fixture that is **3.94 GB** of
GPU-to-CPU copy that nobody reads. The unembedding matmul scales the same way:

```
untrimmed head:  45 x 2 x 1024 x 151,936  =  14.0 GFLOP  per criterion
rest of model:   45 x 2 x 440,467,456     =  39.6 GFLOP  per criterion
                 -> the head is 26.1% of the per-criterion FLOPs
trimmed head:     1 x 2 x 1024 x 151,936  =   0.31 GFLOP  (0.78% of the body)
```

So trimming alone removes about a quarter of the arithmetic and 98% of the
readback. It is the single best thing you can do, and it is one node.

**MEASURED.** `bench/logit_cost.mjs`, Qwen3-0.6B q4f16 on onnxruntime-node (CPU),
prefix prefilled once and held fixed so only the suffix length varies. Result in
`bench/results/logit-cost-Qwen3-0.6B-q4f16.json`:

```
suffix tokens     1      8     16     32     64     96
median ms      32.1   54.0   80.0  132.4  233.6  339.6

fit: 28.4 ms fixed + 3.23 ms per suffix token
```

Cost is linear in suffix length, as the arithmetic above predicts, and a typical
45-token criterion lands at ~174 ms on this backend. (Not comparable to the
browser's WebGPU path, which measures ~91 ms for the same shape — the point here
is the slope, not the intercept.)

What that slope bounds:

- **Lower bound, 1.27x.** The unembedding is 26.1% of per-token FLOPs. Removing
  it for 44 of 45 positions and nothing else: `174 -> 137 ms`.
- **Upper bound, 5.5x.** If the entire per-token term were logits-related:
  `174 -> 32 ms`. It is not — attention and MLP are in there too.

The truth is between, and closer to the bottom than the top for FLOPs alone. What
the arithmetic cannot bound is the 0.61 MB-per-token readback, whose cost depends
entirely on the backend's transfer path; on WebGPU that copy crosses a device
boundary and on CPU it does not. **A trimmed export still has not been built or
timed.** Until it is, this is a well-founded reason to expect a win, not a win.

### How to trim

`typedecide.export.trim_logits_to_last_position` does it by graph surgery: it
renames the `logits` output to `logits_all_positions`, appends a
`Slice(starts=[-1], ends=[INT64_MAX], axes=[1])`, and binds that node's output
back to `logits`, so the graph's interface is unchanged apart from the middle
dimension becoming 1.

```python
from typedecide.export import export_onnx, ExportConfig, trim_logits_to_last_position

directory = export_onnx(ExportConfig("Qwen/Qwen3-0.6B", None, Path("out")))
trim_logits_to_last_position(directory)
```

Run it right after the export and before quantisation. It passes
`onnx.checker` or it writes nothing.

**Two caveats, both real:**

- Decode steps already have `sequence == 1`, so this changes nothing there. It
  is a prefill/suffix fix. In this project the per-criterion suffix *is* the
  prefill, which is why it matters so much here and so little for a chatbot.
- **Any consumer must read the last position from the tensor's own shape.**
  Indexing a trimmed graph's output by `suffix.length - 1` runs off the end of a
  one-position tensor and produces garbage, not an error. `web/engine.js`
  already does this correctly: `readSlots` computes the offset from
  `logits.dims.at(1) - 1`, which is right for both the full and the trimmed
  graph. If you write your own runtime, do the same.

---

## 2. Vocabulary pruning — slicing the LM head to the answer letters

This is the idea the module is built around. The readout compares at most
`len(LETTERS) == 20` logits. Every other row of the unembedding exists to
support generation the model is never asked to do, so slice them off.

`typedecide.export.prune_lm_head(model, keep)` does it, and
`slot_map.json` records what each surviving row used to be.

### The arithmetic, Qwen3-0.6B

`vocab_size = 151936`, `hidden_size = 1024`, 28 layers, `head_dim = 128`,
16 query heads, 8 KV heads, `intermediate_size = 3072`. **DERIVED**, and pinned
in `tests/test_export_prune.py::test_qwen3_0_6b_lm_head_arithmetic` so it cannot
drift:

```
embedding / unembedding table   151,936 x 1024  =   155,582,464 params
everything else                                 =   440,467,456 params
total                                           =   596,049,920 params
                                       the table is  26.10% of the model
```

Pruning to K rows:

```
kept        K x 1024                 K = 4  ->        4,096 params
removed     (151,936 - K) x 1024     K = 4  ->  155,578,368 params
```

Per-position effects at K = 4:

```
unembedding matmul   2 x 1024 x 151,936  =  311,164,928 FLOP  ->  8,192 FLOP
logits tensor, fp32   151,936 x 4 B      =      607,744 B     ->     16 B
logits tensor, fp16   151,936 x 2 B      =      303,872 B     ->      8 B
```

Combine with the trim from §1 and one criterion's logit traffic goes from
26.1 MiB to **12 bytes** (K = 3, the `authored144` option count) — a factor of
2.28 million. That is the number worth remembering, and it is arithmetic, not a
stopwatch.

### The download: tied vs untied, and the honest answer

**This is the part that is usually stated wrong, including in this project's own
CONTRACTS.md gloss.** Whether pruning shrinks the *download* depends entirely on
`config.tie_word_embeddings`.

- **Tied** (Qwen3-0.6B and 1.7B): one table serves both the input embedding and
  the unembedding. The input side still needs a row for every token id in the
  prompt, so the table **cannot leave the file**. In the ONNX graph the same
  initializer feeds the `Gather` and the final `MatMul`, so it is stored once
  and pruning removes nothing from it. Pruning adds a new `K x hidden` head:
  20 x 1024 x 4 B = 80 KiB at fp32. Net download change: **approximately zero,
  very slightly up.** `lm_head_savings(..., tied=True)` returns
  `weight_bytes_removed == 0.0` for exactly this reason. What you get on a tied
  model is the matmul and the tensor copy, which is still most of the win.
- **Untied**: the head is its own matrix and pruning deletes it outright.

> **MEASURED, from each checkpoint's published `config.json` (2026-09-20):**
> Qwen3-0.6B, 1.7B and **4B** set `tie_word_embeddings: true`; Qwen3-8B and 14B set
> it to `false`. So the download saving starts at 8B, not 4B. Only Qwen3-0.6B has
> actually been run in this repository. Read `tie_word_embeddings` for the exact
> checkpoint you ship, and pass `--tied` to `typedecide savings` accordingly.

**DERIVED**, at `q4f16` (4 bits per weight plus one fp16 scale per block of 32 =
4.5 bits = 0.5625 bytes):

| Model | hidden | Tied | Head params | Head share of total | Download removed by pruning |
|---|---:|---|---:|---:|---:|
| Qwen3-0.6B | 1024 | yes | 155,582,464 | 26.10% of 596M | **0** (table still feeds the input embedding) |
| Qwen3-1.7B | 2048 | yes | 311,164,928 | 18.08% of 1.72B | **0** (same reason) |
| Qwen3-4B | 2560 | yes | 388,956,160 | ~9.7% of ~4.02B | **0** (same reason) |
| Qwen3-8B | 4096 | no | 622,329,856 | ~7.6% of ~8.19B | 622,313,472 x 0.5625 = **350,051,328 B (333.8 MiB)** |

Head params and the removal bytes are exact. Total parameter counts for 4B and
8B are the published model-card figures, so the percentage column is
approximate.

**One loose end, PROJECTED.** `web/models.json` annotates Qwen3-0.6B `q4f16` as
`~450 MB`, but 596,049,920 x 0.5625 = 335.3 MB. The 115 MB gap is most likely
the embedding table not being 4-bit at all: `Gather` cannot index int4 data, so
quantisers commonly leave the table at int8 or fp16. If that is what is
happening, then on a tied model the shared table is sitting in the file at
int8 (155.6 MB) or fp16 (311 MB) purely because the input side needs it — and
after untying and pruning, the table is *only* an embedding table and can be
quantised on its own terms without any concern for what the readout matmul
needs. That is a real second-order download win, and it is unverified. Check it
before you promise it:

```python
import onnx, collections
graph = onnx.load("out/onnx/model.onnx").graph
consumers = collections.defaultdict(list)
for node in graph.node:
    for name in node.input:
        consumers[name].append(node.op_type)
for init in graph.initializer:
    if len(init.dims) == 2 and 150_000 < init.dims[0] < 160_000:
        print(init.name, init.dims, onnx.TensorProto.DataType.Name(init.data_type),
              consumers[init.name])
```

One entry consumed by both `Gather` and `MatMul` means the tied table is shared
and pruning will not shrink the file. Two entries means the exporter duplicated
it and pruning removes one whole copy.

### The correctness trap: untie before you slice

Qwen3 ties at small sizes, which means `lm_head.weight` **is**
`model.embed_tokens.weight` — the same `Parameter` object, not a copy. Slicing
it in place leaves the *input* embedding with K rows, and every prompt token id
above K then indexes out of bounds or, worse, silently reads the wrong row. The
model still runs. The accuracy just quietly collapses.

`prune_lm_head` handles it in four steps, and refuses to return a model that
fails the last one:

1. Build a **fresh** `K x hidden` tensor from the kept rows and rebind
   `lm_head.weight` to it. Rebind, never mutate — mutation is what reaches
   through to the input side.
2. Set `config.tie_word_embeddings = False`, or loading the checkpoint back
   re-ties the K-row head onto the input embedding and destroys it.
3. Clear `model._tied_weights_keys`. Left set, `save_pretrained` treats
   `lm_head.weight` as a duplicate of the input embedding, omits it from the
   state dict, and the reloaded model has no head at all. This failure is
   invisible until inference.
4. Verify: the input embedding still has its original row count, its values at
   the probed row are byte-identical, and the two tensors no longer share
   storage. Any of those failing raises `ExportError`.

`config.vocab_size` is deliberately **not** changed. The input side still needs a
row per token id, and lowering it makes Optimum generate out-of-range dummy
inputs during the trace.

`tests/test_export_prune.py` exercises all of this against a stub model whose
head and embedding are literally the same Python list, so the trap is tested in
CI with no torch installed.

### What a pruned model can no longer do

Two things, both permanent, both worth deciding about before you ship:

- **It cannot generate text.** There is no row for any token that is not an
  answer letter. Sampling, beam search, perplexity, and any qualitative "what
  would it have said?" debugging are gone. Keep the unpruned checkpoint.
- **It cannot report what share of total vocabulary mass landed on the answer
  letters.** The full-vocabulary softmax denominator does not exist any more.
  `web/engine.js:readSlots` currently computes exactly this as `allowedMass`,
  and it is a genuinely useful diagnostic — it is the evidence that the model
  has learned the output format rather than being forced into it by the
  argmax-over-letters. A pruned model gives you the softmax over the K kept rows
  and nothing else. That restricted distribution is what the readout uses
  anyway, so accuracy, ECE and Brier over the answer set are unaffected; what
  you lose is the ability to notice that the model wanted to say something that
  was not on the list.

  There is a consolation: computing `allowedMass` costs two full passes over
  151,936 float16 values in JavaScript per criterion (303,872 element decodes,
  151,936 `Math.exp` calls). **PROJECTED**: at an optimistic 50M of those
  branchy half-decode operations per second that is order 5–10 ms per criterion
  of pure JavaScript, on top of the model. Pruning removes both the metric and
  its cost. Measure it with `performance.now()` around `readSlots` before you
  decide which you would rather have.

### slot_map.json

After pruning, the readout indexes **rows 0..K-1, not vocabulary ids.** Nothing
in the weights records which row was which token, so the export writes it:

```json
{
  "version": 1,
  "model_id": "Qwen/Qwen3-0.6B",
  "separator": " ",
  "letters": "ABCD",
  "rows": 4,
  "original_vocab_size": 151936,
  "keep": [362, 425, 356, 422],
  "token_to_row": {"362": 0, "425": 1, "356": 2, "422": 3},
  "can_generate_text": false
}
```

`keep[row] -> token id` and `token_to_row[token id] -> row` are both written so
the consumer does not have to invert anything, and `SlotMap.from_dict` refuses a
file whose two directions disagree. `separator` records whether the letters were
probed as `" A"` or `"A"`, because the browser has to reproduce the same prompt
boundary or it is scoring different tokens. It is `null` when the export could
not reproduce the supplied ids by probing — treat that as a warning, not a
detail.

The order in `keep` is the letter order and is never sorted. Sorting it would
permute the answer letters and produce a model that is confidently wrong rather
than one that crashes.

---

## 3. Quantisation tiers

`typedecide.export.quantize(path, mode=...)`. Sizes are **DERIVED** from the
596,049,920-parameter total; the accuracy column is deliberately empty because
this project has not measured it.

| Mode | Bits/weight | Qwen3-0.6B weights | Compute | Accuracy cost |
|---|---:|---:|---|---|
| `fp32` | 32 | 2,384.2 MB | fp32 | none (it is the reference) |
| `fp16` | 16 | 1,192.1 MB | fp16 | **unmeasured**; expected negligible for an argmax over 20 logits |
| `q8` | 8 | 596.0 MB | fp32 | **unmeasured** |
| `q4` | 4 + 16/32 = 4.5 | 335.3 MB | fp32 | **unmeasured** |
| `q4f16` | 4.5 | 335.3 MB | fp16 | **unmeasured**; the tier most likely to cost something |

The 4.5 bits is 4 bits of weight plus one fp16 scale per block of 32. Change
`block_size` and the arithmetic moves: `4 + 16/block_size`.

**Do not measure this with perplexity.** Perplexity is what quantisation
write-ups report and it is the wrong instrument here. A typed readout never
samples; it compares at most 20 logits and takes an argmax. What matters is
whether the *margin* between the top two answer letters survives the rounding.
Score the same eval set at each tier with `typedecide.evaluation` and compare:

- `balanced_accuracy` — did the answers change?
- `order_consistency` — this is the one that catches the real failure. A tier
  that keeps accuracy but drops order consistency has not preserved the model;
  it has degraded it into a position-guesser that happens to score the same on
  this fixture. `bench/rotation_probe.mjs` exists for exactly this reason, and
  the README's model-comparison table is what it looks like when a model is
  answering by position.
- `ece` / `brier` — quantisation usually shifts calibration before it shifts
  argmaxes, so these move first and are the early warning.

**Order matters: quantise after pruning.** Quantising first spends bits encoding
151,932 rows that are about to be deleted, and on a tied model it forces the
quantiser to pick one representation for a table that the `Gather` and the
`MatMul` will no longer share.

---

## 4. Keep the KV cache on the GPU

**DERIVED.** Qwen3-0.6B has 28 layers, 8 KV heads, `head_dim = 128`. Per token,
per layer, K and V at fp16:

```
2 x (8 x 128) x 2 B          =   4,096 B per layer
4,096 x 28 layers            = 114,688 B = 112 KiB per token
```

For a 256-token evidence prefix that is **29.4 MB** of cache. If it lives in a
CPU-side tensor, every `score()` call uploads all of it and downloads the
`present.*` outputs, which are the prefix plus the 45-token suffix:

```
upload    256 x 114,688        = 29,360,128 B = 29.4 MB
download  (256 + 45) x 114,688 = 34,521,088 B = 34.5 MB
                       total    = 63.9 MB of round-trip per criterion
```

That is **9.2 GB** over a 144-criterion run, before any arithmetic happens. At a
512-token prefix it is 122.6 MB per criterion.

`web/engine.js` already does the right thing structurally — it prefills once and
passes `this.prefix.kv` back in, never mutating or re-sending the text. What it
does not do is pin those tensors to the GPU. In transformers.js that is a
session option:

```js
AutoModelForCausalLM.from_pretrained(repo, {
  device: 'webgpu',
  dtype: 'q4f16',
  session_options: {
    preferredOutputLocation: Object.fromEntries(
      Array.from({ length: 28 }, (_, i) => [
        [`present.${i}.key`, 'gpu-buffer'],
        [`present.${i}.value`, 'gpu-buffer'],
      ]).flat(),
    ),
  },
});
```

With `present.*` produced as GPU buffers, the prefill's cache never leaves the
device and is handed back in by handle. **PROJECTED**: the saving is the copy
time for ~57 MB per criterion, which on a laptop's shared memory bandwidth is
single-digit milliseconds, and considerably more on a discrete GPU over PCIe.
It is not a FLOPs win at all; if a profile shows the time is in the matmuls,
this lever will do nothing. Measure before you bother.

Two things to watch: a GPU-resident tensor must be freed explicitly when the
prefix is replaced, or a long session leaks device memory; and the WASM backend
ignores the option entirely, so the fallback path is unchanged.

---

## 5. Shorten the per-criterion suffix

**MEASURED (shape).** Shared-prefix reuse is exact: prefilling the state once
and reusing that KV for every criterion is bit-identical to scoring each full
prompt fresh, maximum probability drift `0.000e+0` in fp32
(`bench/shared_check.mjs`, reported in `README.md` and `paper/paper.md` §3.2).
This is stronger than the server-side references, which run BF16 and see drift
large enough to flip argmaxes on near-tied decisions.

The consequence is that the *only* per-criterion cost is the suffix:
`"\nCriterion: ...\nOptions:\nA. ...\nAnswer:"`. Everything before it is paid
once per state. `bench/logit_cost.mjs` fits `time = fixed + perToken x n` by
holding the prefilled state constant and varying only the suffix length; it
takes ~45 tokens as a typical criterion. Its committed result
(`bench/results/logit-cost-Qwen3-0.6B-q4f16.json`, onnxruntime-node on CPU) is
the fit quoted in §1: 28.4 ms fixed + 3.23 ms per suffix token. That is a CPU
figure; run it yourself for your own backend.

**PROJECTED.** Suffix cost is linear in tokens, so the arithmetic is simple:
trimming each option description from 12 tokens to 5 across 4 options saves 28
of ~45 tokens, i.e. ~62% of the per-criterion work. Things that actually pay:

- Option *descriptions*, not ids. The model reads the description; the id is
  never tokenised into the prompt (`prompt.tail_text` renders
  `"A. {description}"`).
- Dropping restated context from the question. It is already in the shared head.
- Keeping the number of options at what the decision genuinely needs. Each
  option is a whole line.

What does **not** pay: shortening `INSTRUCTION` or the evidence. Those are in the
shared head, prefilled once, and amortised across every criterion for that
state. Shortening them buys you one prefill, not N suffixes.

Do not shorten the suffix past the point where accuracy moves. The right way to
find that point is to measure both, which is what `typedecide.evaluation`'s
`per_criterion` breakdown is for.

---

## 6. What does not work: batched branch evaluation

**MEASURED, and it is a negative result worth knowing before you design around
it.** The server-side form of this technique replicates the prefix KV across a
batch and evaluates every criterion in one padded forward pass, which makes
total time roughly flat in the number of criteria. ONNX Runtime Web refuses:

```
GroupQueryAttention: batch_size must be 1 when sequence_length > 1 and past context is given.
```

The fused attention kernel permits a batch wider than one only when **no**
past-key cache is supplied — which is the one case where batching would buy you
nothing, because you would be re-prefilling the shared evidence for every
criterion. wllama, the other common browser runtime, exposes no batched branch
evaluation either, for unrelated reasons.

So criteria are scored serially in the browser, and total time is linear in the
criterion count. What the shared prefill buys is a *shorter prompt* per
decision, not a single forward pass. Any claim of "one batched pass, N
decisions" belongs to a server. This is recorded in `README.md` and
`paper/paper.md` §3.3.

The practical consequence for this module: the levers that matter are the ones
that make a single forward pass cheaper — trimming, pruning, quantisation, cache
residency, suffix length. There is no batching win to chase, so do not spend the
week on one.

---

## 7. Measuring, honestly

A speed change that moves accuracy is not a speed change. Every lever here
should be checked on both axes.

```bash
# Accuracy, before and after, on the same fixture and seed.
typedecide evaluate fixtures/authored144.jsonl --model Qwen/Qwen3-0.6B --adapter runs/lora --debias cyclic
```

**MEASURED, for scale.** From `bench/results/`, `authored144` (144 rows,
Qwen3-0.6B `q4f16`), whole-run wall clock:

| Mode | Shots | Mean family balanced accuracy | Order-unstable rows | Seconds |
|---|---:|---:|---:|---:|
| readout | 0 | 0.385 | 0 (not checked) | 56.8 |
| readout | 3 | 0.419 | 0 (not checked) | 189.6 |
| generate | 3 | 0.419 | 0 (not checked) | 225.3 |
| readout, cyclic debias | 3 | 0.483 | 87 | 537.3 |

Read these carefully, because they are easy to over-claim:

- Readout and generation reach *identical* accuracy at 3 shots (0.41936868…,
  the same value to every digit), with readout 1.19x faster. The readout's case
  is that it cannot emit a malformed answer, not that it is dramatically
  quicker.
- Cyclic debiasing took about 2.8x the run time and bought +0.064 balanced accuracy —
  and exposed 87 order-unstable rows out of 144 that the single-ordering runs
  reported as 0 simply because they never checked. That `order_unstable` column
  is the most informative number in the table and it is invisible without the
  extra passes.
- These runs are the *unpruned, untrimmed* stock ONNX model. They are the
  baseline the levers in this document would be measured against. No pruned or
  trimmed run has been committed, which is why §1 and §2 are labelled DERIVED
  throughout.

If you land a pruned export, commit its result file next to these and move the
labels in this document from DERIVED to MEASURED. Until someone does that, the
arithmetic is all this document is entitled to claim.
