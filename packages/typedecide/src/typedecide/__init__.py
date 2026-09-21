"""Fine-tune small language models to answer bounded decisions by logit readout.

A decision here is one question about one state with a closed set of options. The
model never writes the answer: the options are rendered as lettered slots and the
answer is a softmax over the token ids of those letters at a single position. That
makes a malformed answer structurally impossible and the result deterministic.

    from typedecide import load_decisions, validate, group_split, finetune, evaluate

    decisions = load_decisions("tickets.csv")
    report = validate(decisions)
    if not report.ok:
        raise SystemExit(report.render())

    train, held = group_split(decisions, eval_fraction=0.2, seed=0)
    config = TrainConfig(base_model="Qwen/Qwen3-0.6B", output_dir=Path("runs/lora"))
    result = finetune(train, config)
    print(evaluate(held, "Qwen/Qwen3-0.6B", adapter=result.adapter_dir, debias="cyclic").render())

Score with `debias="cyclic"` unless you have a reason not to. Small models carry a
strong preference for whichever option is listed first, and plain accuracy hides it
completely -- see `order_consistency` in the result, which is the number that shows it.

Heavy dependencies are optional extras. `import typedecide` pulls in neither torch
nor transformers; the module that needs one imports it, and says which extra to
install if it is missing.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from .errors import (
    ConfigError,
    DataError,
    EvaluationError,
    ExportError,
    PromptError,
    SchemaError,
    ScorerError,
    TrainingError,
    TypeDecideError,
)
from .prompt import INSTRUCTION, answer_slots, full_prompt, head_text, tail_text
from .schema import (
    LETTERS,
    MAX_OPTIONS,
    MIN_OPTIONS,
    Criterion,
    Decision,
    Option,
    decision_from_dict,
    decision_to_dict,
    fingerprint,
)

__version__ = "0.1.0"

# A library never configures logging; the application does. The NullHandler only stops
# Python's last-resort handler from printing our warnings to stderr in a program that
# has configured nothing, and is a no-op as soon as the application adds a handler.
logging.getLogger("typedecide").addHandler(logging.NullHandler())

__all__ = [
    "INSTRUCTION", "LETTERS", "MAX_OPTIONS", "MIN_OPTIONS", "__version__",
    "ConfigError", "Criterion", "DataError", "Decision", "EvaluationError", "ExportError",
    "Option", "PromptError", "SchemaError", "ScorerError", "TrainingError", "TypeDecideError",
    "answer_slots", "decision_from_dict", "decision_to_dict", "fingerprint",
    "full_prompt", "head_text", "tail_text",
    # Lazily re-exported from the submodules below.
    "evaluate", "finetune", "group_split", "load_decisions", "validate",
]


# name -> (submodule, attribute). Immutable on purpose: no global mutable state.
_LAZY: Final[Mapping[str, tuple[str, str]]] = MappingProxyType({
    "load_decisions": ("data", "load_decisions"),
    "validate": ("data", "validate"),
    "group_split": ("data", "group_split"),
    "write_decisions": ("data", "write_decisions"),
    "finetune": ("training", "finetune"),
    "TrainConfig": ("training", "TrainConfig"),
    "evaluate": ("evaluation", "evaluate"),
    "EvalResult": ("evaluation", "EvalResult"),
    "export_onnx": ("export", "export_onnx"),
    "ExportConfig": ("export", "ExportConfig"),
})


def __getattr__(name: str) -> object:
    """Re-export the five entry points without importing their dependencies at
    import time. `typedecide.finetune` should not cost a torch import until called."""
    if name in _LAZY:
        module_name, attribute = _LAZY[name]
        module = importlib.import_module(f"{__name__}.{module_name}")
        return getattr(module, attribute)
    raise AttributeError(f"module 'typedecide' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Advertise the lazy names too, so tab completion and `dir()` find them."""
    return sorted({*globals(), *_LAZY})
