# Typed decisions in the browser: measuring what limits logit readout on small models

*Working draft. Every number below is produced by `bench/`; none are transcribed by
hand. Sections marked **[pending]** are waiting on a run that has not finished.*

---

## Abstract

**[pending — write last, once Results are fixed.]**

The shape it should take: small language models can answer bounded multiple-choice
decisions by reading a probability distribution off the answer-letter logits rather
than generating text. We implement this entirely client-side, report what the
technique costs and delivers in a browser, and measure what actually limits it. We
find that (i) the batched branch evaluation that makes the server-side form efficient
is unavailable in browser inference runtimes, (ii) sub-1B instruction-tuned models
carry answer-position priors severe enough that plain accuracy is uninformative, and
(iii) because the shared state is prefilled once, averaging over option orderings is
affordable here in a way it is not for generation.

## 1. Introduction

A large share of the decisions software asks a language model to make are small and
closed: route this ticket, is this claim supported, does this need approval. The
conventional implementation asks a chat model to emit JSON and parses it back into
control flow. That spends decode steps on tokens the caller discards, and it offers no
structural guarantee that the emitted value is one of the permitted ones.

An alternative is to present the options as lettered slots and read the probability
distribution over the token ids of those letters at a single position. Nothing is
sampled, so nothing can be malformed; the distribution ranges only over the options
supplied. This is the interface pattern of TypeSafe's Jev, reimplemented openly by
openjev [1] and extended to Indic models by sarvam-jev [2].

Both prior implementations run server-side on a GPU. Our question is what the
technique is worth when it runs *on the user's device*, where there is no server, no
per-token cost, and the data never leaves the tab — and, separately, what actually
limits its accuracy at the model sizes that fit there.

**Contributions.**

1. A browser-native implementation with two correctness improvements over prior work:
   answer-slot token ids are discovered at runtime from the tokenizer rather than
   hardcoded, and shared-prefix reuse is bit-exact in fp32 (§3).
2. A negative engineering result: batched branch evaluation, the property that makes
   the server-side form scale flat in the number of criteria, is refused by ONNX
   Runtime's fused attention kernel and is unavailable in browser runtimes (§3.3).
3. A measurement showing that answer-position priors are a large and *correctable*
   share of error for sub-1B instruction-tuned models on this task — worth 6.4 points
   — and that plain accuracy conceals them entirely (§5.1, §5.2). They are not the
   whole story: the gap to a 4B model remains far larger than anything debiasing
   recovers (§6).
4. Permutation-averaged readout as a debias whose cost structure is specific to this
   method: K orderings cost K short suffixes against one prefilled state, not K full
   prompts (§4.2, §5.2).
5. A like-for-like comparison against hosted frontier models on the same fixture and
   metric, with token and cost accounting (§5.3).

## 2. Background

### 2.1 Reading a decision out of the logits

Given a state $s$, a question $q$ and options $o_1 \dots o_n$, the prompt renders the
options as `A.`, `B.`, … and ends at `Answer:`. Let $t_i$ be the token id that the
tokenizer produces for letter $i$ *at that exact prompt boundary*. The decision is

$$P(o_i \mid s, q) = \mathrm{softmax}_i\big(\ell_{t_1}, \dots, \ell_{t_n}\big)$$

where $\ell$ are the logits at the final position. Two properties follow directly: the
support is exactly the supplied options, and the result is deterministic.

The letters, not the option values, are scored. Scoring option text directly fails
badly when values are multi-token or share prefixes with their neighbours — a point
sarvam-jev makes sharply for Devanagari values [2].

### 2.2 State amortisation

Every criterion asked about one state shares a prompt prefix: the instruction, any
demonstrations, and the state itself. Prefilling that prefix once into a KV cache and
evaluating each criterion as a short suffix against it is what makes asking twenty
questions about one ticket cheap. Correctness requires the prefix to be an exact
*token-level* prefix of every full prompt, which is not automatic: the character
following the state can merge into the final prefix token. We drop the last prefix
token to avoid this, following [1].

## 3. Implementation

`web/` is a static page. There is no backend, no build step, and no network call after
the weights are cached. Inference is `transformers.js` over ONNX Runtime Web, on
WebGPU where available and WebAssembly otherwise.

### 3.1 Answer slots are discovered, not assumed

Prior browser implementations ship the answer-letter token ids as configuration,
because their runtime exposes no tokenizer [2]. This is fragile: the ids are not
contiguous in any vocabulary we checked — for Gemma 3, `A,B,C,D` are `562, 603, 565,
622` — so the common `labelBase + index` shortcut is silently wrong. Whether the
natural continuation is `" A"` or `"A"` also depends on the preceding character.

We probe both separators against the real tokenizer and keep whichever appends exactly
one token per letter without disturbing the prompt's own tokenization, aborting if
neither does. The cost is a handful of encode calls per criterion.

### 3.2 Shared-prefix reuse is exact

We verified the shared path against scoring each full prompt from scratch. In fp32 the
maximum probability drift across the comparison is $0.000 \times 10^{0}$ — bit-identical.
This differs from the server-side references, which run BF16 and report drift large
enough to flip argmaxes on near-tied decisions [1, 2]. The serving numeric format, not
the method, is responsible for that instability.

### 3.3 Batched branching is unavailable in the browser **(negative result)**

The server-side form replicates the prefix KV across a batch and evaluates every
criterion in one padded forward pass, making total time roughly flat in the number of
criteria [1]. In ONNX Runtime this is refused outright:

```
GroupQueryAttention: batch_size must be 1 when sequence_length > 1 and past context is given.
```

The fused attention kernel permits a batch wider than one only when no past-key cache
is supplied. wllama, the other common browser runtime, exposes no batched branch
evaluation either. Client-side, therefore, criteria must be scored serially, and what
the prefill buys is a shorter prompt per decision rather than a single forward pass.
Claims of flat scaling in criteria count do not transfer to the browser.

## 4. Method

### 4.1 Setup

All accuracy is measured on openjev's `authored144` fixture [1]: 144 authored
decisions in three families — evidence interpretation, rule application, candidate
selection — of 48 each, three options each, chance $= 0.333$. We report **mean family
balanced accuracy** as openjev's metric contract specifies: per family, the recall of
each gold class averaged, then averaged over families. The fixture already varies
option order across rows, so a fixed-letter answerer is penalised.

Using an existing public fixture is deliberate. It has published numbers for several
models, which places our results on a scale we did not build.

### 4.2 Permutation-averaged readout

Let $\pi$ range over a set of orderings of the options. For each we score the prompt
built with options in that order and accumulate the probability assigned to each
option *identity*:

$$\bar{P}(o) = \frac{1}{|\Pi|} \sum_{\pi \in \Pi} P_\pi(o)$$

A prior that favours a *position* contributes equally to every option under a cyclic
or full permutation set and therefore cancels. We evaluate $|\Pi| = n$ (cyclic) and
$|\Pi| = n!$ (full).

The cost structure is what makes this practical here rather than merely correct. Each
ordering changes only the option block — the tail — so against a prefilled state it
costs one short suffix, not one full prompt. A generation-based system has no
equivalent: re-asking under $K$ orderings means $K$ complete requests.

### 4.3 Contextual calibration

As a cheaper alternative we estimate the position prior directly: score the same
question and ordering with the evidence replaced by a content-free string, then divide
it out and renormalise, after Zhao et al. [3]. The null distribution is cached per
(question, ordering), so the amortised cost approaches one extra forward pass.

### 4.4 Baselines

- **Generation, same model.** The identical question, with the model writing the
  answer, parsed back. This isolates the readout from the model.
- **Hosted frontier models.** The same fixture and metric through Anthropic, OpenAI
  and Google APIs, in both letter and JSON-output modes, with token counts and
  measured latency. Prices are supplied at run time rather than hardcoded, since they
  move.

## 5. Results

### 5.1 Position priors dominate at small scale

The measurement that motivated everything after it. An eight-item rotation probe,
scoring every item under every ordering of its options:

| Model | Style | Accuracy (chance ≈ 0.44) | Order-consistent | Letters chosen |
|---|---|---:|---:|---|
| Gemma 3 270M-it | completion | 0.421 | 2/8 | A: 89% |
| Gemma 3 270M-it | chat | 0.474 | 0/8 | B: 89% |
| Gemma 3 1B-it | completion | 0.421 | 2/8 | A: 79% |
| Qwen3 0.6B | completion | 0.632 | 7/8 | A: 37 B: 42 C: 21 |

Two observations. Both Gemma checkpoints sit at chance while answering with a fixed
letter, and scaling 270M → 1B did not help; model family matters more than parameter
count in this range. And plain accuracy alone would not have revealed this — only the
order-consistency column does.

### 5.2 Permutation averaging recovers accuracy; the model stays biased

Qwen3-0.6B, q4f16, three demonstrations, on all 144 rows. Chance is 0.333.

| Method | Orderings | Balanced acc | Plain acc | Order-unstable |
|---|---:|---:|---:|---:|
| readout, 0 shots | 1 | 0.385 | 0.458 | — |
| readout, 3 shots | 1 | 0.419 | 0.465 | — |
| generation, 3 shots | 1 | 0.419 | 0.465 | — |
| *Qwen3-0.6B, published by [1], native BF16* | 1 | *0.440* | — | — |
| readout + full permutation averaging | 6 | 0.442 | 0.500 | 106 / 144 |
| readout + contextual calibration | 3 | 0.459 | 0.514 | 80 / 144 |
| **readout + cyclic permutation averaging** | 3 | **0.483** | **0.535** | 87 / 144 |

Four things to take from this.

**Generation and readout score identically** (0.419 on both metrics, to five
decimal places). Greedy decoding of an answer letter selects the same argmax the
readout reads, so the readout's contribution is not accuracy. It is the closed
support, the determinism, and the per-option distribution. We state this plainly
because it is the claim a reader is most likely to assume we are making.

**The single-ordering readout reproduces the published figure.** 0.419 at q4f16
against 0.440 published for the native BF16 checkpoint [1] is consistent with
the cost of 4-bit quantisation, which
supports the harness being sound rather than the result being novel.

**Averaging over orderings is worth 6.4 points**, taking a 0.6B model to 0.483.
The cost structure is what makes this practical rather than merely correct: each
ordering changes only the option block, so against a prefilled state it costs one
short suffix. A generation-based system would need K complete requests.

**But the order-instability column does not improve.** 87 of 144 decisions still
change their argmax across rotations after averaging — averaging repairs the
*readout*, not the model. Contextual calibration, which divides the prior out of
each individual ordering rather than averaging over them, reduces instability
further (80/144) at a lower accuracy. A model that answers by position and is
corrected afterwards is not the same artefact as one that reads the evidence, and
a deployment should not be told otherwise.

**Cyclic versus full permutation averaging is unresolved, and we tested it rather
than eyeballing it.** Full averaging (6 orderings) scores below cyclic (3 orderings),
which the mechanism does not predict: both sets place each option in each position
equally often, so both should cancel a first-order position prior. We re-ran both
modes saving per-row predictions — both reproduced their aggregates exactly, since
the readout is deterministic — and applied an exact McNemar test to the discordant
pairs (`bench/compare.mjs`):

| | full right | full wrong |
|---|---:|---:|
| **cyclic right** | 69 | 8 |
| **cyclic wrong** | 3 | 64 |

Eleven discordant pairs, exact two-sided **p = 0.23**. This sample cannot distinguish
the two. We report both and do not rank them; a reader should not conclude that fewer
orderings are better.

### 5.3 Against hosted frontier models

**[pending — requires API credentials; `bench/api_baseline.mjs` is written and
guarded.]**

### 5.4 Cost of a decision

Suffix cost is linear in length. Holding the prefilled state fixed and varying only
the suffix, Qwen3-0.6B q4f16 on onnxruntime-node (CPU) fits to **28.4 ms fixed plus
3.23 ms per suffix token**, so a typical 45-token criterion costs ~174 ms on that
backend and ~91 ms on WebGPU in a browser.

The readout reads one position. The export returns logits for all of them: a
45-token criterion materialises `[1, 45, 151936]`, 26.1 MiB of fp32, and 3.94 GB
across one pass of the fixture. The unembedding is 26.1% of per-criterion FLOPs.
Trimming the export to the final position bounds between 1.27x (removing the
unembedding FLOPs alone) and 5.5x (if the whole per-token term were logits-related,
which it is not). We have not built a trimmed export, so we claim the bound and not
the speedup.

## 6. Limitations

- **The eight-item probe in §5.1 is a signal, not a benchmark.** The order-consistency
  gap is the robust part of it. The authored144 results supersede it.
- **Probabilities are conditional on the supplied options and are not calibrated
  confidence.** They should not be thresholded as operational confidence without a
  calibration step we have not performed.
- **Quantisation is uncontrolled across comparisons.** Our browser figures are q4f16;
  published comparison numbers are for other formats. Differences of a few points
  should not be read as model differences.
- **Latency figures are hardware-specific** and not comparable between machines, or
  between our WebGPU browser path and our CPU benchmarking path.
- **Debiasing does not close the capacity gap, and we do not claim it does.** Cyclic
  averaging takes Qwen3-0.6B from 0.419 to 0.483. The figure published for a 4B model
  on the same fixture is 0.813 [1]. Option-order priors are a large correctable
  *share* of small-model error; parameter count is still the dominant variable, and a
  reader choosing a model should weight it accordingly.
- **One fixture, one language, one domain.** authored144 is English and synthetic-
  authored. Nothing here establishes behaviour on real production tickets.
- **We did not run Jev itself**, and make no claim about it.

## 7. Related work

openjev [1] is the open reimplementation this work follows for the readout, the
prefix discipline and the metric. sarvam-jev [2] extended it to Indic models and
shipped the first browser demo of the technique, using wllama; our engine is a
response to that one. Contextual calibration [3] and subsequent work on option-order
sensitivity in multiple-choice evaluation motivate §4.3. Architectural background on
Jev is from [4].

## 8. Reproducibility

```bash
bench/fetch_fixtures.sh                 # clones openjev; nothing is vendored here
cd bench && npm install
./run_grid.sh                           # every local row in §5
node summarize.mjs                      # the table
```

API rows need a key in the environment and are cached on disk so a rerun is free.

## References

[1] TheoLeeCJ. *openjev*. https://github.com/TheoLeeCJ/openjev (MIT)
[2] SAGAR-TAMANG. *sarvam-jev*. https://github.com/SAGAR-TAMANG/sarvam-jev
[3] Zhao et al. *Calibrate Before Use: Improving Few-Shot Performance of Language
    Models*. ICML 2021.
[4] archerhume. *Jev's architecture unmasked*.
    https://archerhume.com/posts/jevs-architecture-unmasked/
