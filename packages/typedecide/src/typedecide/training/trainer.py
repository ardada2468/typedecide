"""LoRA fine-tuning on the answer token, and the manifest that makes the run traceable.

`torch`, `transformers` and `peft` are optional extras and are imported inside the
functions that need them, so `import typedecide` stays fast and a base install can still
load, validate and split data. Anything missing raises `TrainingError` naming the extra.

Nothing here re-implements the prompt or the masking: both come from `prompt.py` and
`dataset.py`. What this module owns is the surrounding run -- seeding, the adapter, the
schedule, and `manifest.json`.
"""

from __future__ import annotations

import json
import logging
import platform
import random
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._compat import from_pretrained_with_dtype
from ..errors import TrainingError
from ..schema import Decision, fingerprint
from .config import LOGGING_STEPS, LORA_TARGET_MODULES, LR_SCHEDULER_TYPE, WARMUP_RATIO, TrainConfig
from .dataset import AnswerTokenCollator, DecisionDataset, labelled_only

log = logging.getLogger("typedecide.training.trainer")

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

_INSTALL_HINT = (
    "Install the training extra with `pip install typedecide[train]` "
    "(torch, transformers, peft, accelerate)."
)


@dataclass(frozen=True)
class TrainResult:
    """Where the adapter landed, what produced it, and how it went."""

    adapter_dir: Path
    manifest: dict[str, Any]
    train_loss: float
    eval_loss: float | None


# ------------------------------------------------------------------ optional deps


def _require_training_deps() -> dict[str, Any]:
    """Import the training extras or explain, once, exactly what is missing."""
    missing: list[str] = []
    modules: dict[str, Any] = {}
    for name in ("torch", "transformers", "peft"):
        try:
            modules[name] = __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise TrainingError(
            f"Training needs {', '.join(missing)}, which {'is' if len(missing) == 1 else 'are'} "
            f"not installed. {_INSTALL_HINT}"
        )
    return modules


def library_versions() -> dict[str, str]:
    """Versions of everything that can change a weight, for the manifest.

    Reported as "not installed" rather than omitted, because a manifest that is silent
    about a library cannot be told apart from one written before it mattered.
    """
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in ("torch", "transformers", "peft", "accelerate", "numpy"):
        try:
            module = __import__(name)
        except ImportError:
            versions[name] = "not installed"
        else:
            versions[name] = str(getattr(module, "__version__", "unknown"))
    try:
        from importlib.metadata import version

        versions["typedecide"] = version("typedecide")
    except Exception:  # noqa: BLE001 - an uninstalled source checkout is normal here
        versions["typedecide"] = "source"
    return versions


def seed_everything(seed: int) -> None:
    """Seed python, numpy, torch and transformers from the one number in the config.

    Called before the model is built, because LoRA's B matrices and the dataloader
    shuffle are both drawn from these generators.
    """
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed % (2**32))
    except ImportError:  # pragma: no cover - numpy is a base dependency
        log.warning("numpy is not installed, so its generator was not seeded.")
    try:
        import torch
        from transformers import set_seed
    except ImportError:
        log.debug("torch and transformers are absent; seeded python and numpy only.")
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# ------------------------------------------------------------------------ manifest


def build_manifest(
    config: TrainConfig,
    *,
    train: Sequence[Decision],
    eval_set: Sequence[Decision] | None,
    answer_slots: dict[str, dict[str, int]],
    started_at: float,
    finished_at: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything needed to reproduce, or to disbelieve, this run.

    A result you cannot trace to its input is not a result, so this records the base
    model, the fully resolved config, the dataset fingerprints and row counts, the
    seed, the library versions, wall-clock start and end, and the resolved answer-slot
    token ids per criterion.
    """
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "base_model": config.base_model,
        "config": config.to_manifest(),
        "seed": config.seed,
        "dataset": {
            "train": {
                "fingerprint": fingerprint(train),
                "rows": len(train),
                "criteria": sorted({d.criterion.key for d in train}),
                "groups": len({d.grouping for d in train}),
            },
            "eval": None if eval_set is None else {
                "fingerprint": fingerprint(eval_set),
                "rows": len(eval_set),
                "criteria": sorted({d.criterion.key for d in eval_set}),
                "groups": len({d.grouping for d in eval_set}),
            },
        },
        "answer_slots": answer_slots,
        "option_order_randomisation": (
            "per-epoch" if config.randomise_option_order else "off"
        ),
        "timing": {
            "started_at": _iso(started_at),
            "finished_at": _iso(finished_at),
            "wall_seconds": round(finished_at - started_at, 3),
        },
        "versions": library_versions(),
        "argv": redacted_argv(sys.argv),
    }
    if extra:
        manifest.update(extra)
    return manifest


_SECRET_FLAG = re.compile(r"(token|secret|password|passwd|api[-_]?key|credential|auth)", re.I)
_SECRET_VALUE = re.compile(
    r"^(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})$"
)
REDACTED = "[redacted]"


def redacted_argv(argv: Sequence[str]) -> list[str]:
    """`argv` with anything that looks like a credential blanked out.

    The manifest is written beside the adapter and travels with it -- to a model hub, a
    ticket, a colleague. `finetune` is a library call, so `sys.argv` belongs to whatever
    program imported us and may carry `--hf-token hf_...`. The command line is worth
    recording; the secret on it is not.
    """
    out: list[str] = []
    hide_next = False
    for argument in argv:
        text = str(argument)
        if hide_next:
            out.append(REDACTED)
            hide_next = False
            continue
        flag, equals, value = text.partition("=")
        if flag.startswith("-") and _SECRET_FLAG.search(flag):
            if equals:
                out.append(f"{flag}={REDACTED}")
            else:
                out.append(text)
                hide_next = True
            continue
        if _SECRET_VALUE.match(text) or (equals and _SECRET_VALUE.match(value)):
            out.append(REDACTED if not equals else f"{flag}={REDACTED}")
            continue
        out.append(text)
    return out


def write_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    """Write `manifest.json` beside the adapter and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    log.info("Manifest written to %s", path)
    return path


def _iso(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------- training


def finetune(
    train: Sequence[Decision],
    config: TrainConfig,
    *,
    eval_set: Sequence[Decision] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> TrainResult:
    """LoRA-tune `config.base_model` so that one token carries all the loss.

    Every example is `encode(prompt) + [answer_token]` with every position but the last
    masked to -100; see `dataset.build_example`. Option order is re-randomised at the
    start of each epoch when `config.randomise_option_order` is set, which is the single
    most valuable preprocessing step here: without it the model learns "answer A".

    `progress`, if given, receives dicts -- `{"event": "preflight" | "log" |
    "epoch" | "done", ...}` -- so a CLI or notebook can show something without this
    module owning a progress bar.
    """
    modules = _require_training_deps()
    torch = modules["torch"]

    started_at = time.time()
    labelled, unlabelled = labelled_only(train)
    if unlabelled:
        raise TrainingError(
            f"{len(unlabelled)} of {len(train)} training decisions have no answer_id, "
            f"starting with {', '.join(unlabelled[:5])}. Label them or filter them out; "
            "training silently on a smaller set than you passed is worse than stopping."
        )
    if not labelled:
        raise TrainingError("There are no labelled decisions to train on.")

    labelled_eval: list[Decision] | None = None
    if eval_set is not None:
        labelled_eval, unlabelled_eval = labelled_only(eval_set)
        if unlabelled_eval:
            log.warning(
                "Dropping %d unlabelled eval decisions (%s%s).",
                len(unlabelled_eval), ", ".join(unlabelled_eval[:5]),
                "..." if len(unlabelled_eval) > 5 else "",
            )
        if not labelled_eval:
            raise TrainingError("The eval set has no labelled decisions to score.")

    seed_everything(config.seed)
    _emit(progress, {"event": "start", "train_rows": len(labelled),
                     "eval_rows": 0 if labelled_eval is None else len(labelled_eval)})

    tokenizer = _load_tokenizer(config.base_model)
    train_data = DecisionDataset(
        labelled, tokenizer, config.max_length,
        randomise_option_order=config.randomise_option_order, seed=config.seed,
    )
    # The eval set is never re-permuted: a moving eval set is not a measurement.
    eval_data = None if labelled_eval is None else DecisionDataset(
        labelled_eval, tokenizer, config.max_length,
        randomise_option_order=False, seed=config.seed,
    )

    answer_slots = train_data.preflight()
    if eval_data is not None:
        eval_data.preflight()
    _emit(progress, {"event": "preflight", "criteria": sorted(answer_slots)})
    log.info(
        "%d training decisions, %s held out, %d distinct criteria.",
        len(train_data), "none" if eval_data is None else str(len(eval_data)), len(answer_slots),
    )

    model = _build_peft_model(config, torch, modules["transformers"], modules["peft"])
    trainer = _build_trainer(
        config, model, tokenizer, train_data, eval_data, torch,
        modules["transformers"], progress,
    )

    train_output = trainer.train()
    train_loss = float(train_output.training_loss)

    eval_loss: float | None = None
    if eval_data is not None:
        metrics = trainer.evaluate()
        raw = metrics.get("eval_loss")
        eval_loss = None if raw is None else float(raw)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))

    manifest = build_manifest(
        config,
        train=labelled,
        eval_set=labelled_eval,
        answer_slots=answer_slots,
        started_at=started_at,
        finished_at=time.time(),
        extra={
            "losses": {"train": train_loss, "eval": eval_loss},
            "steps": int(getattr(train_output, "global_step", 0) or 0),
            "trainable_parameters": _trainable_parameters(model),
        },
    )
    write_manifest(manifest, config.output_dir)
    _emit(progress, {"event": "done", "adapter_dir": str(config.output_dir),
                     "train_loss": train_loss, "eval_loss": eval_loss})
    log.info("Adapter written to %s (train loss %.4f)", config.output_dir, train_loss)

    return TrainResult(
        adapter_dir=config.output_dir,
        manifest=manifest,
        train_loss=train_loss,
        eval_loss=eval_loss,
    )


# ------------------------------------------------------------------------ pieces


def _load_tokenizer(base_model: str) -> Any:
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except Exception as error:
        raise TrainingError(
            f"Could not load a tokenizer for {base_model!r}: {error}. "
            "Check the model id, or the local path, and your network access."
        ) from error
    if tokenizer.pad_token_id is None:
        # Padding is masked out of attention and carries -100 labels, so the choice of
        # pad token cannot reach the loss; it only has to be a real id.
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise TrainingError(
            f"The tokenizer for {base_model!r} has neither a pad token nor an eos token, so "
            "a batch cannot be padded. Set `pad_token` in the tokenizer's config and retry."
        )
    return tokenizer


def _build_peft_model(config: TrainConfig, torch: Any, transformers: Any, peft: Any) -> Any:
    try:
        model = from_pretrained_with_dtype(
            transformers.AutoModelForCausalLM.from_pretrained,
            config.base_model,
            torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    except Exception as error:
        raise TrainingError(
            f"Could not load {config.base_model!r} as a causal LM: {error}."
        ) from error
    model.config.use_cache = False  # incompatible with gradient checkpointing and unused

    lora = peft.LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.effective_lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_MODULES),
    )
    try:
        return peft.get_peft_model(model, lora)
    except ValueError as error:
        raise TrainingError(
            f"LoRA could not attach to {config.base_model!r}: {error}. "
            f"The target modules are {', '.join(LORA_TARGET_MODULES)}; an architecture "
            "that names its projections differently needs its own target list."
        ) from error


def _build_trainer(
    config: TrainConfig,
    model: Any,
    tokenizer: Any,
    train_data: DecisionDataset,
    eval_data: DecisionDataset | None,
    torch: Any,
    transformers: Any,
    progress: Callable[[dict[str, Any]], None] | None,
) -> Any:
    arguments = transformers.TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch" if eval_data is not None else "no",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=torch.cuda.is_available(),
        seed=config.seed,
        data_seed=config.seed,
        report_to=[],
        remove_unused_columns=False,  # our dataset yields dicts, not model kwargs
    )
    callbacks = [_epoch_callback(transformers, train_data, progress)]
    if progress is not None:
        callbacks.append(_log_callback(transformers, progress))
    return transformers.Trainer(
        model=model,
        args=arguments,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=AnswerTokenCollator(tokenizer.pad_token_id),
        callbacks=callbacks,
    )


def _epoch_callback(
    transformers: Any,
    dataset: DecisionDataset,
    progress: Callable[[dict[str, Any]], None] | None,
) -> Any:
    """Re-permute option order at the start of every epoch (see DecisionDataset)."""

    # `transformers` is an untyped optional import, so its base class is `Any` to mypy.
    class EpochCallback(transformers.TrainerCallback):  # type: ignore[misc]
        def on_epoch_begin(self, _args: Any, state: Any, _control: Any, **_kwargs: Any) -> None:
            # `state.epoch` is a float that accumulates step fractions, so at the start
            # of epoch 1 it can read 0.9999...; int() would floor that to 0 and replay
            # epoch 0's option order, quietly halving the orderings the model sees.
            epoch = round(float(state.epoch or 0.0))
            dataset.set_epoch(epoch)
            _emit(progress, {"event": "epoch", "epoch": epoch})

    return EpochCallback()


def _log_callback(transformers: Any, progress: Callable[[dict[str, Any]], None]) -> Any:
    # `transformers` is an untyped optional import, so its base class is `Any` to mypy.
    class LogCallback(transformers.TrainerCallback):  # type: ignore[misc]
        def on_log(
            self,
            _args: Any,
            state: Any,
            _control: Any,
            logs: dict[str, Any] | None = None,
            **_kwargs: Any,
        ) -> None:
            # Through _emit, like every other event: a progress bar that raises on step
            # 4,000 must not be what ends a twelve-hour run.
            _emit(progress, {"event": "log", "step": int(state.global_step), **(logs or {})})

    return LogCallback()


def _trainable_parameters(model: Any) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": int(trainable), "total": int(total)}


def _emit(callback: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    if callback is None:
        return
    try:
        callback(event)
    except Exception:  # noqa: BLE001 - a broken progress bar must not kill a long run
        log.exception("The progress callback raised on event %r; continuing.", event.get("event"))
