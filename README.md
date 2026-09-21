# Typed decisions in the browser

Read small decisions out of a model's logits instead of asking it to write them, with
the whole thing running in the visitor's own tab on Qwen3 0.6B.

Most decisions software asks a model to make are small — *route this ticket, how
urgent is it, does this need approval*. A chat model can answer, but it spends its
time generating text the caller immediately parses back into an `if`, and nothing
stops that text being wrong in shape as well as in content. This project presents the
options as lettered slots and takes the probability distribution over the token ids of
those letters. Nothing is sampled, so nothing can be malformed, and the state is
prefilled once and reused by every question asked about it.

```
                    ┌── "which queue?"      ──→ logits[A,B,C,D] ──→ softmax
state (prefilled    ├── "how urgent?"       ──→ logits[A,B,C,D] ──→ softmax
 once, reused)      ├── "is this angry?"    ──→ logits[A,B]     ──→ softmax
   KV cache         └── "needs approval?"   ──→ logits[A,B,C]   ──→ softmax
```

## What we measured before building it

All of this is reproducible from `bench/` and the notes below. Numbers are from a
MacBook, Qwen3 0.6B, q4f16 on WebGPU.

**Shared-prefix reuse is exact.** Prefilling the state once and reusing that KV cache
for every criterion gives bit-identical results to scoring each full prompt from
scratch — max probability drift `0.000e+0`. The reference implementation this design
follows runs BF16 server-side and sees drift large enough to flip argmaxes; fp32 in
the browser does not.

**Batched branch evaluation is not possible in a browser.** ONNX Runtime's fused
attention kernel rejects it outright:

```
GroupQueryAttention: batch_size must be 1 when sequence_length > 1 and past context is given.
```

wllama has the same limitation for a different reason. So criteria are scored one at a
time; what the prefill buys is a shorter prompt per decision, not a single forward pass.
Any claim of "one batched pass, N decisions" belongs to a server, not to this page.

**Model choice matters far more than the mechanism, and option order matters most of
all.** On openjev's `authored144` fixture (144 authored decisions, 3 families, 3
options each, chance 0.333), scored with their own metric:

| System | Method | Balanced accuracy | Order-unstable |
|---|---|---:|---:|
| chance | — | 0.333 | — |
| Qwen3 0.6B | readout, 0 shots | 0.385 | — |
| Qwen3 0.6B | readout, 3 shots | 0.419 | — |
| Qwen3 0.6B | generation, 3 shots | 0.419 | — |
| Qwen3 0.6B *(openjev, published, native BF16)* | readout | 0.440 | — |
| Qwen3 0.6B | readout + contextual calibration | 0.459 | 80 / 144 |
| Qwen3 0.6B | readout + full permutation averaging (6 orderings) | 0.442 | 106 / 144 |
| **Qwen3 0.6B** | **readout + cyclic permutation averaging** | **0.483** | 87 / 144 |

Three things in that table are worth stopping on.

**Generation and readout score identically.** Greedy decoding of an answer letter picks
the same argmax the readout reads. So the readout does not buy accuracy — it buys a
structural guarantee, determinism, and a probability per option. Anyone selling it as
"more accurate" is selling something else.

**87 of 144 decisions changed their answer when the options were reordered.** The
position prior is not a footnote, it is the dominant error source at this model size.
Plain accuracy conceals it completely.

**Averaging over option orderings recovers 6.4 points** and puts a 0.6B model above the
published figure for it. This is cheap *here specifically*: each ordering changes only
the option block, so against a prefilled state it costs one short suffix rather than
one full prompt. A generation-based system would need K complete requests.

Cyclic and full permutation averaging are **not** statistically distinguishable here:
an exact McNemar test on the 11 rows where they disagree gives p = 0.23
(`bench/compare.mjs`). Both are reported; neither is ranked above the other.

Earlier, on an eight-item probe that motivated all of this, both Gemma 3 checkpoints
(270M and 1B) sat at chance while answering with a fixed letter, and 270M → 1B made it
worse. Model family matters more than parameter count in this range. That probe is a
signal, not a benchmark; the table above supersedes it.

## Run the page

```bash
cd web && python3 -m http.server 8090
```

Weights download once (~450 MB) and are cached by the browser. There is no backend,
so the ticket text has nowhere to go. WebGPU is used when available and WebAssembly
otherwise; the page says which it got.

**The questions are yours to write.** Every criterion's question, its options (2 to
20), and the JSON key your code will read are editable in the page, and your edits
survive a reload. The example rubric is a starting point, not a fixture. Unlike a
one-decision-at-a-time interface, every criterion here is asked about the *same*
prefilled state, which is what makes the eleventh question cost a 40-token suffix
instead of another full prompt.

Three other controls are worth knowing at a demo:

- **Shuffle option order** re-runs with the options permuted. This is the question a
  sceptical reviewer asks, so it is better to have the button than to be asked for it.
- **Name every required key for the generation lane** is on by default, because a
  baseline that was never told what was wanted is a strawman. Turning it off shows
  what the same model does with a prompt that is merely reasonable: on the double-charge
  ticket it returns valid JSON answering 4 of 10 criteria and silently dropping the
  rest.
- **Also measure without state reuse** scores every criterion again from a cold prompt,
  so the prefill saving is visible rather than asserted.

## What the demo actually demonstrates

Be careful with the speed claim. Given a well-engineered prompt, the generation lane
answers all ten criteria correctly in about the same order of time as the readout
(1.5 s against 1.15 s on the reference machine). The honest wins are:

- **Time to first decision.** The readout returns decision one in ~310 ms; generation
  has produced nothing usable until the whole object closes.
- **Malformed output is structurally impossible.** Each criterion's softmax ranges only
  over its own answer letters, so an option id belonging to a different criterion is
  not a value the readout can emit. The generation lane has no such guarantee.
- **A probability per option**, which lets you route the uncertain decisions to a human.
  Generation gives you a token and no confidence at all.
- **Determinism.** Nothing is sampled.
- **No server and no per-token cost.** ~90 ms per decision on a laptop GPU.

## Fine-tune it on your own decisions

Stock weights know the format but not your policy. [`packages/typedecide`](packages/typedecide)
is the library that turns that around, and it takes your data in whatever shape you
already have it.

```bash
pip install -e packages/typedecide[train,export]

typedecide validate tickets.csv \
  --col-state "Ticket Body" --col-question "Decision" --col-options "Choices"
typedecide split    tickets.csv --out data/
typedecide train    --config train.yaml --train data/train.jsonl
typedecide evaluate data/eval.jsonl --model Qwen/Qwen3-0.6B --debias cyclic
typedecide evaluate data/eval.jsonl --model Qwen/Qwen3-0.6B \
  --adapter runs/lora --debias cyclic          # the after number
typedecide export   --base Qwen/Qwen3-0.6B --adapter runs/lora --out web/models/mine
```

Three things it is opinionated about, each for a measured reason: it splits on groups
rather than rows so a state cannot appear on both sides; it puts loss only on the
answer-letter token, because that is the only position the readout ever reads; and it
randomises option order during training, because without that a model learns "answer
A" — which is exactly the pathology in the results table above.

Every run writes a `manifest.json` recording the base model, the resolved config, a
content hash of the dataset, the seed, and library versions. `packages/typedecide/README.md`
has the full story.

### Making it faster

[`packages/typedecide/docs/SPEED.md`](packages/typedecide/docs/SPEED.md), with every
claim marked MEASURED or PROJECTED. The two levers:

**The export returns logits for every position.** `Qwen3-0.6B-ONNX` takes
`input_ids`, `attention_mask`, `position_ids` and 56 KV tensors, and has no
`num_logits_to_keep` — so a 45-token criterion produces `[1, 45, 151936]`, **27.35 MB**,
of which the readout reads three numbers. Trimming to the last position needs a
re-export and removes almost all of it.

**The unembedding can be sliced to the answer letters.** A readout only ever looks at
~20 token ids, so the other 151,916 rows of the LM head compute nothing anyone reads:

```
$ typedecide savings --vocab 151936 --hidden 1024 --keep 20 --tied
  head_parameters                   155,582,464
  weight_bytes_removed                        0     <- tied embeddings: no download saving
  logit_bytes_per_position_before        607,744
  logit_bytes_per_position_after              80
  matmul_flops_per_position_before   311,164,928
  matmul_flops_per_position_after         40,960
```

Note the honest row: Qwen3-0.6B, 1.7B and 4B set `tie_word_embeddings=True` (8B and
up do not), so for the sizes that fit a browser the
unembedding *is* the input embedding and pruning cannot shrink the download — the full
table is still needed for lookups. The win is the matmul and the copy off the
accelerator, not the file size. A pruned model answers decisions and cannot generate
text.

## Layout

```
web/            the static page — no build step, no backend
  prompt.js       the canonical prompt, shared with the library
  rubric.js       the editable criteria: your questions, your options, your JSON keys
  engine.js       prefill, branch, read the logits
  worker.js       keeps the model off the main thread
packages/
  typedecide/     the library: bring your own data, fine-tune, evaluate, export
bench/          every number in this README, reproducible
paper/          the write-up
train/          the original prototype scripts, superseded by packages/typedecide
```

`reference-openjev/` and `reference-sarvam-jev/` are cloned by
`bench/fetch_fixtures.sh` and gitignored — they are other people's code and data.

## Credits

The letter-slot readout, the prefix-verification discipline and the rotation metric
follow [SAGAR-TAMANG/sarvam-jev](https://github.com/SAGAR-TAMANG/sarvam-jev) and, behind
it, [TheoLeeCJ/openjev](https://github.com/TheoLeeCJ/openjev) (MIT). Architectural
background on Jev itself is from
[archerhume's teardown](https://archerhume.com/posts/jevs-architecture-unmasked/).
None of those projects are affiliated with this one.
