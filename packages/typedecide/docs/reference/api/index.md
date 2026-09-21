# Python API

These pages are generated from the source with
[mkdocstrings](https://mkdocstrings.github.io/) by static analysis. The signatures and
docstrings you read here are the ones in the release the site was built from.

## The top-level package

::: typedecide
    options:
      members: false
      show_root_heading: false
      show_source: false

### What `import typedecide` gives you

Imported eagerly (standard library only):

| Name | From |
|---|---|
| `Option`, `Criterion`, `Decision`, `LETTERS`, `MIN_OPTIONS`, `MAX_OPTIONS` | [`typedecide.schema`](schema.md) |
| `decision_from_dict`, `decision_to_dict`, `fingerprint` | [`typedecide.schema`](schema.md) |
| `INSTRUCTION`, `head_text`, `tail_text`, `full_prompt`, `answer_slots` | [`typedecide.prompt`](prompt.md) |
| `TypeDecideError` and its subclasses | [`typedecide.errors`](errors.md) |
| `__version__` | the package |

Resolved lazily on first attribute access, so that `import typedecide` never imports a
heavy dependency:

| Name | Resolves to |
|---|---|
| `load_decisions`, `write_decisions` | [`typedecide.data.loaders`](data/loaders.md) |
| `validate` | [`typedecide.data.validate`](data/validate.md) |
| `group_split` | [`typedecide.data.split`](data/split.md) |
| `finetune` | [`typedecide.training.trainer`](training/trainer.md) |
| `TrainConfig` | [`typedecide.training.config`](training/config.md) |
| `evaluate`, `EvalResult` | [`typedecide.evaluation.runner`](evaluation/runner.md) |
| `export_onnx`, `ExportConfig` | [`typedecide.export.onnx`](export/onnx.md) |

```python
import typedecide
from typedecide import load_decisions            # works: resolved lazily
from typedecide.data import FieldMapping          # subpackage imports always work
```

## Subpackages

| Package | Public names (`__all__`) |
|---|---|
| `typedecide.data` | `FieldMapping`, `load_decisions`, `write_decisions`, `validate`, `DataReport`, `Finding`, `estimate_tokens`, `DEFAULT_MAX_STATE_TOKENS`, `group_split`, `randomise_option_order`, `all_rotations`, `all_permutations` |
| `typedecide.training` | `TrainConfig`, `finetune`, `TrainResult`, `DecisionDataset`, `AnswerTokenCollator`, `build_example`, `answer_slots_for`, `labelled_only`, `pad_batch`, `reorder_options`, `build_manifest`, `write_manifest`, `library_versions`, `seed_everything`, `IGNORE_INDEX`, `MANIFEST_NAME`, `LORA_TARGET_MODULES`, `LR_SCHEDULER_TYPE`, `WARMUP_RATIO`, `LOGGING_STEPS` |
| `typedecide.evaluation` | `evaluate`, `evaluate_with_scorer`, `EvalResult`, `orderings`, `DEBIAS_MODES`, `NULL_STATE`, `ScoreFn`, `ScoreRequest`, `TorchReadout`, `load_readout`, `accuracy`, `balanced_accuracy`, `mean_group_balanced_accuracy`, `expected_calibration_error`, `brier_score`, `order_consistency`, `letter_distribution`, `chance_accuracy` |
| `typedecide.export` | `ExportConfig`, `export_onnx`, `trim_logits_to_last_position`, `prune_lm_head`, `answer_letter_token_ids`, `answer_letter_slots`, `SlotMap`, `build_slot_map`, `read_slot_map`, `write_slot_map`, `select_rows`, `validate_keep`, `lm_head_savings`, `SLOT_MAP_FILENAME`, `quantize`, `resolve_mode`, `QuantizationTier`, `QUANTIZATION_MODES` |

Names beginning with an underscore are private and may change in any release. See
[Versioning and deprecation](../../project/versioning.md) for what counts as public.

## Typing

The package is fully annotated and checked with `mypy --strict` in CI.
