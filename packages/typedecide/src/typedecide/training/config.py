"""The full description of a fine-tuning run, in one frozen object.

Everything that changes the weights lives here, so `TrainConfig.to_manifest()` plus the
base model id and the dataset fingerprint is enough to re-run a job. Values that are
really fixed policy rather than knobs (the LoRA target modules, the schedule) are module
constants, and they are still written into the manifest -- a manifest that only records
the fields someone happened to expose is a manifest you cannot reproduce from.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from ..errors import ConfigError

log = logging.getLogger("typedecide.training.config")

# Policy, not knobs. Recorded in the manifest so a run stays reproducible even if a
# later release changes them.
LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
LOGGING_STEPS = 25

# Below this the criterion, its options and "Answer:" cannot fit, so there would be
# nothing left to truncate from the state.
MIN_MAX_LENGTH = 16

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class TrainConfig:
    """One fine-tuning job.

    `output_dir` receives the LoRA adapter, the tokenizer and `manifest.json`.
    `lora_alpha` defaults to twice the rank when left unset; read the resolved value
    from `effective_lora_alpha`, never from the field.
    """

    base_model: str
    output_dir: Path
    epochs: float = 2.0
    batch_size: int = 4
    grad_accum: int = 4
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int | None = None
    lora_dropout: float = 0.05
    max_length: int = 1024
    seed: int = 0
    randomise_option_order: bool = True

    def __post_init__(self) -> None:
        # Accept a str for output_dir so hand-built configs and parsed files agree. The
        # annotation says Path, so the check goes through `object` to stay honest with
        # mypy about what callers actually pass.
        given: object = self.output_dir
        if not isinstance(given, Path):
            object.__setattr__(self, "output_dir", Path(str(given)))

        for name in ("epochs", "learning_rate", "lora_dropout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{name} must be a number, got {value!r}.")
            if not math.isfinite(value):
                # NaN compares False against everything, so it would walk straight
                # through every range check below and into the optimiser.
                raise ConfigError(f"{name} must be a finite number, got {value!r}.")

        if not str(self.base_model).strip():
            raise ConfigError("base_model is empty; name a Hugging Face model id or a local path.")
        if not str(self.output_dir).strip():
            raise ConfigError("output_dir is empty; name a directory for the adapter and manifest.")
        if self.epochs <= 0:
            raise ConfigError(f"epochs must be greater than 0, got {self.epochs}.")
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be at least 1, got {self.batch_size}.")
        if self.grad_accum < 1:
            raise ConfigError(f"grad_accum must be at least 1, got {self.grad_accum}.")
        if self.learning_rate <= 0:
            raise ConfigError(f"learning_rate must be greater than 0, got {self.learning_rate}.")
        if self.lora_rank < 1:
            raise ConfigError(f"lora_rank must be at least 1, got {self.lora_rank}.")
        if self.lora_alpha is not None and self.lora_alpha < 1:
            raise ConfigError(
                f"lora_alpha must be at least 1 when set, got {self.lora_alpha}; "
                "leave it out to use twice the rank."
            )
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ConfigError(
                f"lora_dropout must be in [0.0, 1.0), got {self.lora_dropout}."
            )
        if self.max_length < MIN_MAX_LENGTH:
            raise ConfigError(
                f"max_length must be at least {MIN_MAX_LENGTH}, got {self.max_length}; "
                "the criterion, its options and 'Answer:' have to fit before the state "
                "can be truncated at all."
            )
        if self.seed < 0:
            raise ConfigError(f"seed must not be negative, got {self.seed}.")

    # ------------------------------------------------------------------ derived

    @property
    def effective_lora_alpha(self) -> int:
        """The alpha actually handed to peft: twice the rank unless one was given."""
        return self.lora_rank * 2 if self.lora_alpha is None else self.lora_alpha

    @property
    def effective_batch_size(self) -> int:
        """Decisions per optimiser step, which is the number that shapes the run."""
        return self.batch_size * self.grad_accum

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_file(cls, path: str | Path) -> TrainConfig:
        """Read a config from YAML or JSON.

        The format is taken from the suffix; anything else is tried as JSON and then as
        YAML. A relative `output_dir` is resolved against the config file's own
        directory, so a config can be moved with its run without being rewritten.
        """
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as error:
            raise ConfigError(
                f"Could not read the training config at {path}: {error}"
            ) from error
        except UnicodeDecodeError as error:
            raise ConfigError(
                f"The training config at {path} is not UTF-8 ({error.reason} at byte "
                f"{error.start}). Save it as UTF-8 and retry."
            ) from error

        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = _load_json(text, path)
        elif suffix in {".yaml", ".yml"}:
            payload = _load_yaml(text, path)
        else:
            try:
                payload = _load_json(text, path)
            except ConfigError:
                payload = _load_yaml(text, path)

        if not isinstance(payload, Mapping):
            raise ConfigError(
                f"{path} must hold a mapping of config keys, not {type(payload).__name__}."
            )
        return cls.from_mapping(payload, base_dir=path.parent)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, base_dir: Path | None = None
    ) -> TrainConfig:
        """Build a config from an already-parsed mapping, naming every unusable key."""
        known = {f.name for f in fields(cls)}
        # str() first: YAML happily produces int, bool or None keys, and sorting or
        # joining a mixed set raises a TypeError instead of the diagnosis below.
        unknown = sorted({str(key) for key in payload} - known)
        if unknown:
            raise ConfigError(
                f"Unknown training config key(s): {', '.join(unknown)}. "
                f"Permitted keys are: {', '.join(sorted(known))}."
            )
        missing = sorted({"base_model", "output_dir"} - set(payload))
        if missing:
            raise ConfigError(
                f"The training config is missing {', '.join(missing)}; "
                "both the base model and the output directory are required."
            )

        output_dir = Path(str(payload["output_dir"])).expanduser()
        if base_dir is not None and not output_dir.is_absolute():
            output_dir = base_dir / output_dir

        return cls(
            base_model=str(payload["base_model"]),
            output_dir=output_dir,
            epochs=_as_float(payload, "epochs", 2.0),
            batch_size=_as_int(payload, "batch_size", 4),
            grad_accum=_as_int(payload, "grad_accum", 4),
            learning_rate=_as_float(payload, "learning_rate", 1e-4),
            lora_rank=_as_int(payload, "lora_rank", 16),
            lora_alpha=None if payload.get("lora_alpha") is None
            else _as_int(payload, "lora_alpha", 0),
            lora_dropout=_as_float(payload, "lora_dropout", 0.05),
            max_length=_as_int(payload, "max_length", 1024),
            seed=_as_int(payload, "seed", 0),
            randomise_option_order=_as_bool(payload, "randomise_option_order", True),
        )

    # ----------------------------------------------------------------- manifest

    def to_manifest(self) -> dict[str, Any]:
        """A JSON-safe, fully resolved record of this config.

        Resolved means no `None` standing for "work it out later": `lora_alpha` is the
        number peft receives, and the fixed policy constants are included so a manifest
        written by one release still describes the run if a later one changes them.
        """
        return {
            "base_model": self.base_model,
            "output_dir": str(self.output_dir),
            "epochs": float(self.epochs),
            "batch_size": int(self.batch_size),
            "grad_accum": int(self.grad_accum),
            "effective_batch_size": self.effective_batch_size,
            "learning_rate": float(self.learning_rate),
            "lora_rank": int(self.lora_rank),
            "lora_alpha": self.effective_lora_alpha,
            "lora_alpha_was_explicit": self.lora_alpha is not None,
            "lora_dropout": float(self.lora_dropout),
            "lora_target_modules": list(LORA_TARGET_MODULES),
            "max_length": int(self.max_length),
            "seed": int(self.seed),
            "randomise_option_order": bool(self.randomise_option_order),
            "lr_scheduler_type": LR_SCHEDULER_TYPE,
            "warmup_ratio": WARMUP_RATIO,
            "logging_steps": LOGGING_STEPS,
        }


# ---------------------------------------------------------------------- parsing


def _load_json(text: str, path: Path) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path} is not valid JSON: {error}") from error


def _load_yaml(text: str, path: Path) -> Any:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - pyyaml is a base dependency
        raise ConfigError(
            f"{path} looks like YAML but PyYAML is not installed. "
            "Install it with `pip install pyyaml`, or write the config as JSON."
        ) from error
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(f"{path} is not valid YAML: {error}") from error


def _as_int(payload: Mapping[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    try:
        if isinstance(value, bool):
            raise TypeError("a boolean is not a number here")
        if isinstance(value, float) and not value.is_integer():
            # int(2.7) is 2. A config that says batch_size: 2.7 is a mistake to report,
            # not a number to round down without a word.
            raise ValueError("it has a fractional part")
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ConfigError(f"{key} must be a whole number, got {value!r}.") from error


def _as_float(payload: Mapping[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    try:
        if isinstance(value, bool):
            raise TypeError("a boolean is not a number here")
        return float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{key} must be a number, got {value!r}.") from error


def _as_bool(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
    if isinstance(value, int):
        return bool(value)
    raise ConfigError(
        f"{key} must be true or false, got {value!r}."
    )
