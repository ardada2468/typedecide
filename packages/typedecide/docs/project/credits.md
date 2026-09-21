# Third-party credits

Nothing from the projects below is vendored into `typedecide`. The two reference
repositories are fetched locally by `bench/fetch_fixtures.sh` and are gitignored,
because they are other people's code and data and should be obtained from their own
sources. The authoritative statement is
[`THIRD_PARTY.md`](https://github.com/ardada2468/typedecide/blob/main/THIRD_PARTY.md).

## openjev (MIT)

[TheoLeeCJ/openjev](https://github.com/TheoLeeCJ/openjev)

- `benchmarks/data/authored144.jsonl` is the fixture every accuracy number on this
  site was measured on. It is **not** redistributed here.
- The metric, mean family balanced accuracy, follows openjev's
  `benchmarks/manifests/metric-contract.json`.
- The letter-slot readout and the prefix-verification discipline originate there.
- The published comparison figure for Qwen3-0.6B, 0.440, is theirs: a **native BF16**
  score measured on their hardware. (Q8_0 is the quantised browser artefact openjev
  lists beside it, not the precision that score was measured at.)

## sarvam-jev

[SAGAR-TAMANG/sarvam-jev](https://github.com/SAGAR-TAMANG/sarvam-jev)

The immediate predecessor of this project: the same mechanism on Sarvam's Indic
models, with a wllama browser demo. The rotation metric that became
`order_consistency` follows it. The browser engine in `web/` is a response to that
one, and the repository README records how it differs.

## Jev

Architectural background on Jev itself is from
[archerhume's teardown](https://archerhume.com/posts/jevs-architecture-unmasked/).

!!! note "No affiliation"
    None of these projects are affiliated with this one, and none of them have
    endorsed it.

## Runtime dependencies

| Package | Used for | When |
|---|---|---|
| [NumPy](https://numpy.org/) | Seeding, ONNX graph constants | Base install |
| [PyYAML](https://pyyaml.org/) | Reading `TrainConfig` files | Base install |
| [PyTorch](https://pytorch.org/) | Training and scoring | `train` extra |
| [Transformers](https://github.com/huggingface/transformers) | Models, tokenizers, the trainer | `train` extra |
| [PEFT](https://github.com/huggingface/peft) | LoRA adapters | `train` extra |
| [Accelerate](https://github.com/huggingface/accelerate) | Used by the trainer | `train` extra |
| [Datasets](https://github.com/huggingface/datasets) | Loading a Hugging Face dataset id | `train` extra |
| [Optimum](https://github.com/huggingface/optimum) | The ONNX export | `export` extra |
| [ONNX](https://onnx.ai/) and [ONNX Runtime](https://onnxruntime.ai/) | Graph surgery and quantisation | `export` extra |
| [PyArrow](https://arrow.apache.org/docs/python/) | Reading Parquet | `parquet` extra |

The browser demo loads [transformers.js](https://github.com/huggingface/transformers.js)
from a CDN. Each dependency is distributed under its own licence.

## This site

Built with [MkDocs](https://www.mkdocs.org/),
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) and
[mkdocstrings](https://mkdocstrings.github.io/).
