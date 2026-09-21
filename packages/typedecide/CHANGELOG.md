# Changelog

All notable changes to `typedecide` are recorded here.

The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) and
the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While
the version is `0.y.z` the public API may change in a minor release; any such change is
listed under **Changed** or **Removed** with the migration.

Each release heading must read `## [X.Y.Z] - YYYY-MM-DD`. The release workflow extracts
the section whose heading matches the tag and uses it as the GitHub Release body, and
fails the release if it cannot find one.

## [Unreleased]

## [0.1.0] - 2026-09-20

First public release.

### Added

- **Core schema and prompt.** `Decision`, `Criterion` and `Option` dataclasses with
  `decision_from_dict` / `decision_to_dict` round-tripping and a content `fingerprint`.
  `prompt` renders a decision as lettered answer slots (`full_prompt`, `head_text`,
  `tail_text`, `answer_slots`) and mirrors the browser's `web/prompt.js` byte for byte.
  A `TypeDecideError` hierarchy (`SchemaError`, `DataError`, `PromptError`,
  `ConfigError`, `TrainingError`, `ExportError`) whose messages name the fix.
- **`typedecide.data`.** `load_decisions` reads JSONL, JSON, CSV, Parquet and Hugging
  Face dataset ids into one `Decision` type, with `FieldMapping` to point at existing
  column names, delimited or wide option columns, and answers given as an option id,
  letter, 0- or 1-based index, or exact option text. `write_decisions` writes them
  back. `validate` returns a `DataReport` of findings that hurt later: contradictory
  labels, single-class criteria, inconsistent option sets, classes too rare to split,
  over-long states, duplicate ids. `group_split` splits on `Decision.grouping` so a
  state never lands on both sides, and refuses a split that leaves a class out of
  eval. `randomise_option_order`, `all_rotations` and `all_permutations` permute
  options while carrying `answer_id` through untouched.
- **`typedecide.training`.** `finetune` LoRA-tunes a causal LM with the loss on the
  answer-letter token only (`AnswerTokenCollator`, labels masked with `IGNORE_INDEX`
  everywhere else) and left-truncation so the criterion, options and `Answer:` always
  survive. `TrainConfig` is validated and loadable from YAML. Every run writes a
  `manifest.json`: base model, resolved config, dataset content hash, row counts, seed,
  library versions, wall-clock window and the resolved answer-slot token ids.
- **`typedecide.evaluation`.** `evaluate` scores a model by logit readout under one or
  more option orderings and returns an `EvalResult` reporting balanced accuracy, order
  consistency and the letter distribution together. `debias` modes (`cyclic`,
  `permutation`, `calibrated`) cancel a positional prior. `evaluate_with_scorer` accepts
  any scoring function, so the runner is testable without a model. Pure-Python
  `metrics`: `accuracy`, `balanced_accuracy`, `mean_group_balanced_accuracy`,
  `order_consistency`, `letter_distribution`, `brier_score`,
  `expected_calibration_error`, `chance_accuracy`.
- **`typedecide.export`.** `export_onnx` merges an adapter, prunes the unembedding to
  the answer letters (`prune_lm_head`, with a `slot_map.json` that travels with the
  model) and traces the decoder to ONNX with a past-key cache.
  `trim_logits_to_last_position` returns one position instead of the whole sequence.
  `quantize` writes the transformers.js-named tiers (for example `model_q4f16.onnx`).
  `lm_head_savings` does the arithmetic for a given vocabulary, hidden size and tied or
  untied embeddings.
- **`typedecide` command-line interface** with `validate`, `split`, `train`,
  `evaluate`, `export` and `savings` subcommands.
- **Optional extras.** `train` (torch, transformers, peft, accelerate, datasets),
  `export` (optimum, onnx, onnxruntime), `parquet` (pyarrow), `docs`, `dev` and `all`.
  `import typedecide` imports neither torch nor transformers; the five entry points are
  re-exported lazily, and a missing dependency is reported with the extra to install.
- **Typing.** The package is fully annotated, checked with `mypy --strict`, and ships a
  PEP 561 `py.typed` marker.
- **Packaging.** Python 3.10 to 3.13. Published to PyPI from GitHub Actions by Trusted
  Publishing; release candidates go to TestPyPI first. See `RELEASING.md`.

[Unreleased]: https://github.com/ardada2468/typedecide/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ardada2468/typedecide/releases/tag/v0.1.0
