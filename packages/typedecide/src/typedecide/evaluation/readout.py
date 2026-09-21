"""Turning a causal language model into a probability distribution over options.

The readout is the whole trick of this library: instead of asking the model to write
an answer and then parsing whatever it wrote, we build the prompt so that the very
next token must be an answer letter, run one forward pass, and read the softmax over
just those letters' logits. The output is a real distribution over the options that
were listed, so it can be averaged over orderings and calibrated -- neither of which
is possible with generated text.

Everything that touches torch or transformers is imported inside the function that
needs it, so `import typedecide` stays cheap and the base install stays light.

The seam that matters for testing is `ScoreFn`: anything that maps a batch of
`ScoreRequest` to a batch of probability vectors is a scorer. `runner.evaluate`
builds a `TorchReadout`; a test injects a fake and exercises the same debiasing code
with no model at all.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._compat import from_pretrained_with_dtype
from ..errors import ConfigError
from ..prompt import answer_slots, full_prompt
from ..schema import Criterion

LOGGER = logging.getLogger("typedecide.evaluation.readout")

__all__ = ["ScoreFn", "ScoreRequest", "TorchReadout", "load_readout"]

#: Most answer-slot lookups a `TorchReadout` remembers at once (oldest dropped first).
SLOT_CACHE_LIMIT = 2048


@dataclass(frozen=True, slots=True)
class ScoreRequest:
    """One prompt to score: a state, and a criterion in the exact option order to
    present. The returned probabilities are indexed by *position in this criterion*,
    so a caller that reordered the options must map them back to option ids itself.
    """

    state: str
    criterion: Criterion

    @property
    def prompt(self) -> str:
        return full_prompt(self.state, self.criterion)


# A scorer takes a batch of requests and returns one probability vector per request,
# each as long as that request's option list. Implementations may batch internally
# however they like; they must preserve order.
ScoreFn = Callable[[Sequence[ScoreRequest]], list[list[float]]]


def _require(module: str, extra: str = "train") -> Any:
    """Import an optional heavy dependency, or explain how to get it."""
    try:
        return importlib.import_module(module)
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ConfigError(
            f"Scoring a model needs {module!r}, which is not installed. "
            f"Install the extra that provides it:  pip install 'typedecide[{extra}]'"
        ) from error


class TorchReadout:
    """A `ScoreFn` backed by a local transformers model.

    Construction loads the weights, so build one and reuse it across a whole eval
    rather than per decision.
    """

    def __init__(
        self,
        model_id: str,
        *,
        adapter: Path | None = None,
        device: str | None = None,
        batch_size: int = 8,
        seed: int = 0,
        max_length: int | None = None,
    ) -> None:
        torch = _require("torch")
        transformers = _require("transformers")

        if batch_size < 1:
            raise ConfigError(f"batch_size must be at least 1, got {batch_size}.")

        # Deterministic even though a readout does not sample: a run that cannot be
        # reproduced is not evidence.
        torch.manual_seed(seed)

        self.model_id = model_id
        self.adapter = adapter
        self.batch_size = batch_size
        self.seed = seed
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        # No `trust_remote_code`: see `from_pretrained_with_dtype`.
        model = from_pretrained_with_dtype(
            transformers.AutoModelForCausalLM.from_pretrained, model_id, dtype
        )
        if adapter is not None:
            peft = _require("peft")
            model = peft.PeftModel.from_pretrained(model, str(adapter))
            model = model.merge_and_unload()
        self.model = model.to(self.device).eval()

        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ConfigError(
                f"Tokenizer for {model_id!r} has neither a pad nor an eos token, so batched "
                "scoring cannot pad. Set tokenizer.pad_token before evaluating, or use "
                "batch_size=1."
            )
        self.pad_id: int = int(pad)
        self._slot_cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ scoring

    def __call__(self, requests: Sequence[ScoreRequest]) -> list[list[float]]:
        return self.score(requests)

    def score(self, requests: Sequence[ScoreRequest]) -> list[list[float]]:
        """Probability over each request's options, in the order the request lists them."""
        rows = list(requests)
        out: list[list[float]] = []
        for start in range(0, len(rows), self.batch_size):
            out.extend(self._score_batch(rows[start : start + self.batch_size]))
        return out

    def _slots_for(self, prompt: str, count: int) -> list[int]:
        cached = self._slot_cache.get(prompt)
        if cached is None:
            cached, _separator = answer_slots(self.tokenizer, prompt, count)
            # The key is the whole prompt, state included, so nearly every entry is
            # unique. Unbounded, this held every prompt of an evaluation in memory for
            # the life of the readout; bounded, it still serves the repeats that matter
            # (content-free calibration prompts and re-scored decisions).
            if len(self._slot_cache) >= SLOT_CACHE_LIMIT:
                self._slot_cache.pop(next(iter(self._slot_cache)))
            self._slot_cache[prompt] = cached
        return cached

    def _score_batch(self, chunk: Sequence[ScoreRequest]) -> list[list[float]]:
        torch = _require("torch")

        prompts = [request.prompt for request in chunk]
        encoded = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        slot_lists = [
            self._slots_for(prompt, len(request.criterion.options))
            for prompt, request in zip(prompts, chunk, strict=True)
        ]

        width = max(len(ids) for ids in encoded)
        if self.max_length is not None and width > self.max_length:
            LOGGER.warning(
                "A prompt is %d tokens, over max_length=%d. It is scored in full anyway: "
                "truncating would move or destroy the answer boundary and the readout "
                "would score the wrong token.",
                width,
                self.max_length,
            )

        input_ids: list[list[int]] = []
        attention: list[list[int]] = []
        for ids in encoded:
            gap = width - len(ids)
            # LEFT padding, not right. Two reasons, both of which produce silently
            # wrong numbers rather than a crash if you get them wrong:
            #
            #  1. We read the distribution from the logits at the LAST position. With
            #     left padding every prompt's final real token sits at index -1, so
            #     one slice serves the whole batch. With right padding index -1 is a
            #     pad token for every row except the longest, and the readout would
            #     score the continuation of padding.
            #  2. Position ids. A decoder-only model derives positions from arange by
            #     default, so left-padded rows would be told their real tokens start
            #     at position `gap` instead of 0. Rotary/absolute embeddings then
            #     differ from what the same prompt gets when scored alone, and a row's
            #     answer changes depending on which other prompts shared its batch.
            #     The fix is the explicit position_ids below, derived from the mask.
            input_ids.append([self.pad_id] * gap + ids)
            attention.append([0] * gap + [1] * len(ids))

        ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        mask = torch.tensor(attention, dtype=torch.long, device=self.device)
        # Positions must come from the mask, not from arange: cumsum gives each real
        # token its rank among real tokens, so the first real token is 0 in every row,
        # and the clamp keeps the (masked-out) pad positions at a legal index.
        positions = (mask.cumsum(-1) - 1).clamp(min=0)

        with torch.inference_mode():
            logits = self.model(
                input_ids=ids_tensor, attention_mask=mask, position_ids=positions
            ).logits[:, -1, :].float()

        return [
            torch.softmax(logits[row, slots], dim=-1).tolist()
            for row, slots in enumerate(slot_lists)
        ]


def load_readout(
    model_id: str,
    *,
    adapter: Path | None = None,
    device: str | None = None,
    batch_size: int = 8,
    seed: int = 0,
    max_length: int | None = None,
) -> TorchReadout:
    """Load a model and return it as a `ScoreFn`."""
    return TorchReadout(
        model_id,
        adapter=adapter,
        device=device,
        batch_size=batch_size,
        seed=seed,
        max_length=max_length,
    )
