"""The canonical prompt, mirroring web/prompt.js exactly.

If these two drift, the fine-tune optimises a prompt the browser never sends and
accuracy quietly collapses. tests/test_prompt_parity.py proves they agree, and it is
wired into CI, because this is the most expensive silent failure available here.
"""

from __future__ import annotations

from typing import Any, Protocol

from .errors import PromptError
from .schema import LETTERS, Criterion


class Tokenizer(Protocol):
    """The slice of a Hugging Face tokenizer this module needs."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

INSTRUCTION = (
    "Apply the criterion to the evidence and choose exactly one listed option. "
    "Answer with only the uppercase letter of that option."
)


def head_text(state: str) -> str:
    """Everything up to and including the state. Identical across every criterion."""
    return f"{INSTRUCTION}\n\nEvidence:\n{state}\n"


def _options_of(criterion: Criterion | dict[str, Any]) -> list[tuple[str, str]]:
    if isinstance(criterion, Criterion):
        return [(o.id, o.description) for o in criterion.options]
    return [(str(o["id"]), str(o["description"])) for o in criterion["options"]]


def _question_of(criterion: Criterion | dict[str, Any]) -> str:
    return criterion.question if isinstance(criterion, Criterion) else str(criterion["question"])


def tail_text(criterion: Criterion | dict[str, Any]) -> str:
    """One criterion's own question and options. Never shared."""
    lines = [f"\nCriterion: {_question_of(criterion)}", "Options:"]
    for letter, (_, description) in zip(LETTERS, _options_of(criterion), strict=False):
        lines.append(f"{letter}. {description}")
    lines.append("Answer:")
    return "\n".join(lines)


def full_prompt(state: str, criterion: Criterion | dict[str, Any]) -> str:
    return head_text(state) + tail_text(criterion)


def state_prefix_ids(tokenizer: Tokenizer, state: str) -> list[int]:
    """Token ids of the shared head, trimmed by one.

    The last token is dropped because the character that follows the state can merge
    into it, which would stop the head being an exact token-level prefix of every
    full prompt built on it.
    """
    ids = tokenizer.encode(head_text(state), add_special_tokens=False)
    if len(ids) < 2:
        raise PromptError("The state prefix is too short to reuse.")
    return list(ids[:-1])


def answer_slots(tokenizer: Tokenizer, prompt: str, count: int) -> tuple[list[int], str]:
    """Token ids of the answer letters at this exact prompt boundary.

    These are not contiguous in any vocabulary we have checked, and whether the
    natural continuation is " A" or "A" depends on the preceding character, so both
    are probed and we keep whichever appends exactly one token without disturbing
    the prompt's own tokenization.
    """
    base = tokenizer.encode(prompt, add_special_tokens=False)
    if not base:
        raise PromptError("Prompt encoded to zero tokens.")
    for separator in (" ", ""):
        slots: list[int] = []
        for letter in LETTERS[:count]:
            encoded = tokenizer.encode(prompt + separator + letter, add_special_tokens=False)
            if len(encoded) != len(base) + 1 or encoded[: len(base)] != base:
                slots = []
                break
            slots.append(encoded[-1])
        if slots and len(slots) == len(set(slots)):
            return slots, separator
    raise PromptError(
        "No answer-letter continuation keeps the prompt boundary stable for this "
        "tokenizer, so a readout would score tokens that are not the answer. The "
        "prompt tail needs adjusting for this model."
    )
