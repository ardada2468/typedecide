"""LoRA fine-tuning where the loss lands only on the answer-letter token.

    from typedecide.training import TrainConfig, finetune

    result = finetune(train_decisions, TrainConfig(
        base_model="Qwen/Qwen3-0.6B", output_dir=Path("runs/triage-lora"),
    ), eval_set=held_out)

Importing this package pulls in nothing heavier than the standard library: `torch`,
`transformers` and `peft` are imported inside the functions that need them, and their
absence is reported as a `TrainingError` naming `pip install typedecide[train]`.
"""

from __future__ import annotations

from .config import (
    LOGGING_STEPS,
    LORA_TARGET_MODULES,
    LR_SCHEDULER_TYPE,
    WARMUP_RATIO,
    TrainConfig,
)
from .dataset import (
    IGNORE_INDEX,
    AnswerTokenCollator,
    DecisionDataset,
    answer_slots_for,
    build_example,
    labelled_only,
    pad_batch,
    reorder_options,
)
from .trainer import (
    MANIFEST_NAME,
    TrainResult,
    build_manifest,
    finetune,
    library_versions,
    seed_everything,
    write_manifest,
)

__all__ = [
    "IGNORE_INDEX",
    "LOGGING_STEPS",
    "LORA_TARGET_MODULES",
    "LR_SCHEDULER_TYPE",
    "MANIFEST_NAME",
    "WARMUP_RATIO",
    "AnswerTokenCollator",
    "DecisionDataset",
    "TrainConfig",
    "TrainResult",
    "answer_slots_for",
    "build_example",
    "build_manifest",
    "finetune",
    "labelled_only",
    "library_versions",
    "pad_batch",
    "reorder_options",
    "seed_everything",
    "write_manifest",
]
