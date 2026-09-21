"""prompt.py's tokenizer-facing half: the shared prefix and the answer slots."""

from __future__ import annotations

import re

import pytest

from typedecide.errors import PromptError
from typedecide.prompt import (
    INSTRUCTION,
    answer_slots,
    full_prompt,
    head_text,
    state_prefix_ids,
    tail_text,
)
from typedecide.schema import Criterion, Option

CRITERION = Criterion("k", "Is it?", (Option("no", "No"), Option("yes", "Yes")))


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens  # named by the Tokenizer protocol; a fake ignores it
        return [ord(c) for c in text]


class SpaceMergingTokenizer:
    """A leading space merges into the next character, as it does in most BPE vocabularies."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens  # named by the Tokenizer protocol; a fake ignores it
        pieces = re.findall(r" ?\S|\s", text)
        return [sum(ord(c) * 131**i for i, c in enumerate(piece)) for piece in pieces]


class EmptyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return []


class CaseBlindTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens  # named by the Tokenizer protocol; a fake ignores it
        return [7 if c.isupper() else ord(c) for c in text]


def test_the_prompt_text_is_pinned() -> None:
    """A change to a single character here invalidates every adapter ever trained."""
    assert full_prompt("The evidence.", CRITERION) == (
        "Apply the criterion to the evidence and choose exactly one listed option. "
        "Answer with only the uppercase letter of that option.\n\n"
        "Evidence:\nThe evidence.\n\n"
        "Criterion: Is it?\nOptions:\nA. No\nB. Yes\nAnswer:"
    )
    assert head_text("x").startswith(INSTRUCTION) and tail_text(CRITERION).endswith("Answer:")


def test_dict_and_criterion_render_identically() -> None:
    as_dict = {"question": "Is it?", "options": [{"id": "no", "description": "No"},
                                                 {"id": "yes", "description": "Yes"}]}
    assert tail_text(as_dict) == tail_text(CRITERION)


def test_state_prefix_is_a_token_prefix_of_every_full_prompt() -> None:
    for tokenizer in (CharTokenizer(), SpaceMergingTokenizer()):
        prefix = state_prefix_ids(tokenizer, "Some evidence here.")
        whole = tokenizer.encode(full_prompt("Some evidence here.", CRITERION))
        assert whole[: len(prefix)] == prefix
        assert len(prefix) == len(tokenizer.encode(head_text("Some evidence here."))) - 1
    with pytest.raises(PromptError, match="too short to reuse"):
        state_prefix_ids(EmptyTokenizer(), "x")


def test_answer_slots_choose_the_separator_the_tokenizer_needs() -> None:
    text = full_prompt("evidence", CRITERION)
    assert answer_slots(CharTokenizer(), text, 2) == ([ord("A"), ord("B")], "")
    slots, separator = answer_slots(SpaceMergingTokenizer(), text, 2)
    assert separator == " " and len(set(slots)) == 2


def test_answer_slots_refuse_rather_than_score_the_wrong_token() -> None:
    text = full_prompt("evidence", CRITERION)
    with pytest.raises(PromptError, match="zero tokens"):
        answer_slots(EmptyTokenizer(), text, 2)
    with pytest.raises(PromptError, match="No answer-letter continuation"):
        answer_slots(CaseBlindTokenizer(), text, 2)
