"""The masking invariant, verified against stub tokenizers -- no torch, no downloads.

The one thing this file exists to prove:

    input_ids = encode(full_prompt(state, criterion)) + [answer_token]
    labels    = [-100] * len(encode(full_prompt(state, criterion))) + [answer_token]
    answer_token = answer_slots(tokenizer, prompt, n_options)[0][answer_index]

Everything else here -- truncation, option permutation, padding -- is tested for the
same reason: each of them is a way that invariant could quietly stop holding.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # the package is used from a source checkout in CI
    sys.path.insert(0, str(SRC))

from typedecide.errors import TrainingError  # noqa: E402
from typedecide.prompt import answer_slots, full_prompt  # noqa: E402
from typedecide.schema import Criterion, Decision, Option  # noqa: E402
from typedecide.training import dataset as ds  # noqa: E402
from typedecide.training.dataset import (  # noqa: E402
    IGNORE_INDEX,
    AnswerTokenCollator,
    DecisionDataset,
    build_example,
    labelled_only,
    pad_batch,
    reorder_options,
)

# ------------------------------------------------------------- fake tokenizers


class CharTokenizer:
    """One token per character. Lets a test predict token counts exactly.

    `answer_slots` settles on the empty separator here, because " A" is two tokens.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens  # named by the Tokenizer protocol; a fake ignores it
        return [ord(character) for character in text]


class MergingTokenizer:
    """A space glues onto the character after it, the way a BPE vocabulary does.

    This is the case that forces `answer_slots` onto the " " separator, and the case
    where a head's final token can merge with what follows -- the reason both
    `prompt.state_prefix_ids` and our truncation drop one token before comparing.
    """

    _pattern = re.compile(r" ?[^\s]|\s")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens  # named by the Tokenizer protocol; a fake ignores it
        return [hash_token(piece) for piece in self._pattern.findall(text)]


class CollidingTokenizer:
    """Every uppercase letter shares one id, so no answer slot is distinguishable."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens  # named by the Tokenizer protocol; a fake ignores it
        return [7 if character.isupper() else ord(character) for character in text]


def hash_token(piece: str) -> int:
    # Stable across processes, unlike hash(); the exact ids never matter, only that
    # distinct pieces get distinct ids.
    value = 0
    for character in piece:
        value = value * 1103515245 + ord(character)
    return value % 2_000_000 + 1000


# ------------------------------------------------------------------- fixtures


def make_criterion(n_options: int = 3, key: str = "triage") -> Criterion:
    return Criterion(
        key=key,
        question="How urgent is this report?",
        options=tuple(
            Option(id=f"level_{i}", description=f"Priority level {i} for the queue")
            for i in range(n_options)
        ),
    )


def make_decision(
    *,
    state: str = "The pump failed at 03:10 and the backup did not start.",
    answer: str | None = "level_1",
    n_options: int = 3,
    identifier: str = "d1",
    group: str | None = None,
) -> Decision:
    return Decision(
        id=identifier,
        state=state,
        criterion=make_criterion(n_options),
        answer_id=answer,
        group_id=group,
    )


TOKENIZERS = [CharTokenizer(), MergingTokenizer()]
TOKENIZER_IDS = ["char", "merging"]


# ------------------------------------------------------------- THE INVARIANT


@pytest.mark.parametrize("tokenizer", TOKENIZERS, ids=TOKENIZER_IDS)
def test_loss_lands_only_on_the_answer_token(tokenizer: object) -> None:
    decision = make_decision()
    example = build_example(tokenizer, decision, max_length=4096)

    prompt = full_prompt(decision.state, decision.criterion)
    n_options = len(decision.criterion.options)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)  # type: ignore[attr-defined]
    slots, _ = answer_slots(tokenizer, prompt, n_options)  # type: ignore[arg-type]
    answer_token = slots[decision.answer_index]

    assert example["input_ids"] == [*prompt_ids, answer_token]
    assert example["labels"] == [IGNORE_INDEX] * len(prompt_ids) + [answer_token]
    assert example["attention_mask"] == [1] * (len(prompt_ids) + 1)
    assert len({len(v) for v in example.values()}) == 1


@pytest.mark.parametrize("tokenizer", TOKENIZERS, ids=TOKENIZER_IDS)
def test_only_one_position_is_supervised(tokenizer: object) -> None:
    example = build_example(tokenizer, make_decision(), max_length=4096)
    supervised = [label for label in example["labels"] if label != IGNORE_INDEX]
    assert len(supervised) == 1, "exactly one token may carry loss"
    assert supervised[0] == example["input_ids"][-1]


@pytest.mark.parametrize("answer_index", [0, 1, 2, 3])
def test_the_supervised_token_is_the_slot_of_the_gold_option(answer_index: int) -> None:
    tokenizer = CharTokenizer()
    decision = make_decision(answer=f"level_{answer_index}", n_options=4)
    example = build_example(tokenizer, decision, max_length=4096)

    prompt = full_prompt(decision.state, decision.criterion)
    slots, _ = answer_slots(tokenizer, prompt, 4)
    assert example["input_ids"][-1] == slots[answer_index]
    # And, for the char tokenizer, that really is the letter of that option.
    assert example["input_ids"][-1] == ord("ABCD"[answer_index])


def test_a_different_gold_moves_the_supervised_token() -> None:
    tokenizer = CharTokenizer()
    first = build_example(tokenizer, make_decision(answer="level_0"), 4096)
    second = build_example(tokenizer, make_decision(answer="level_2"), 4096)
    assert first["input_ids"][:-1] == second["input_ids"][:-1]
    assert first["input_ids"][-1] != second["input_ids"][-1]


def test_the_merging_tokenizer_really_uses_the_space_separator() -> None:
    # Guards the fixture: if this ever stops holding, the "merging" runs above would
    # silently be a second copy of the char case.
    decision = make_decision()
    prompt = full_prompt(decision.state, decision.criterion)
    _, separator = answer_slots(MergingTokenizer(), prompt, 3)
    assert separator == " "


# --------------------------------------------------------------- bad rows


def test_an_unlabelled_row_is_refused_by_name() -> None:
    decision = make_decision(answer=None, identifier="no-gold-here")
    with pytest.raises(TrainingError) as caught:
        build_example(CharTokenizer(), decision, 4096)
    assert "no-gold-here" in str(caught.value)


def test_an_unresolvable_answer_slot_is_reported_with_the_decision_id() -> None:
    decision = make_decision(identifier="row-42")
    with pytest.raises(TrainingError) as caught:
        build_example(CollidingTokenizer(), decision, 4096)
    message = str(caught.value)
    assert "row-42" in message, "the offending row must be named, not silently dropped"
    assert "triage" in message


def test_a_dataset_reports_the_offending_row_rather_than_dropping_it() -> None:
    data = DecisionDataset([make_decision(identifier="row-7")], CollidingTokenizer(), 4096)
    with pytest.raises(TrainingError) as caught:
        _ = data[0]
    assert "row-7" in str(caught.value)


# ------------------------------------------------------------- truncation


def prompt_pieces(decision: Decision) -> tuple[str, str]:
    from typedecide.prompt import head_text, tail_text

    return head_text(decision.state), tail_text(decision.criterion)


def test_a_short_prompt_is_not_truncated() -> None:
    decision = make_decision()
    example = build_example(CharTokenizer(), decision, max_length=4096)
    prompt = full_prompt(decision.state, decision.criterion)
    assert example["input_ids"][:-1] == CharTokenizer().encode(prompt)


def test_a_long_state_is_truncated_from_the_left_and_the_tail_survives() -> None:
    tokenizer = CharTokenizer()
    decision = make_decision(state="EARLIEST. " + "filler evidence. " * 400 + " LATEST.")
    max_length = 600
    example = build_example(tokenizer, decision, max_length)

    assert len(example["input_ids"]) == max_length
    assert len(example["labels"]) == max_length
    assert example["labels"] == [IGNORE_INDEX] * (max_length - 1) + [example["input_ids"][-1]]

    head, tail = prompt_pieces(decision)
    # The criterion, its lettered options and "Answer:" are byte-for-byte intact, and
    # the answer token still sits immediately after them.
    assert example["input_ids"][-(len(tail) + 1) : -1] == tokenizer.encode(tail)

    prompt = full_prompt(decision.state, decision.criterion)
    slots, _ = answer_slots(tokenizer, prompt, 3)
    assert example["input_ids"][-1] == slots[decision.answer_index]


def test_truncation_keeps_the_instruction_preamble_and_drops_the_oldest_evidence() -> None:
    tokenizer = CharTokenizer()
    decision = make_decision(state="EARLIEST. " + "filler evidence. " * 400 + " LATEST.")
    example = build_example(tokenizer, decision, max_length=600)

    kept = "".join(chr(i) for i in example["input_ids"][:-1])
    assert kept.startswith(ds.PREAMBLE[:-1]), "the instruction must survive truncation"
    assert "EARLIEST" not in kept, "the oldest evidence is what gets dropped"
    assert "LATEST" in kept, "the newest evidence is what is kept"
    assert "Answer:" in kept


def test_truncation_is_exactly_at_the_budget() -> None:
    tokenizer = CharTokenizer()
    decision = make_decision(state="x" * 5000)
    for max_length in (300, 512, 777):
        example = build_example(tokenizer, decision, max_length)
        assert len(example["input_ids"]) == max_length


def test_a_criterion_longer_than_max_length_is_refused_not_trimmed() -> None:
    long_options = Criterion(
        key="wordy",
        question="Which of these very long descriptions applies?" + " more words" * 50,
        options=tuple(
            Option(id=f"o{i}", description="a very long option description " * 20)
            for i in range(3)
        ),
    )
    decision = Decision(id="too-wide", state="short state", criterion=long_options,
                        answer_id="o0")
    with pytest.raises(TrainingError) as caught:
        build_example(CharTokenizer(), decision, max_length=64)
    message = str(caught.value)
    assert "too-wide" in message
    assert "max_length" in message


def test_when_the_boundary_cannot_be_found_the_tail_still_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = CharTokenizer()
    decision = make_decision(state="y" * 3000)
    monkeypatch.setattr(ds, "_prefix_len", lambda *_args, **_kwargs: 0)
    example = build_example(tokenizer, decision, max_length=400)

    _, tail = prompt_pieces(decision)
    assert len(example["input_ids"]) == 400
    assert example["input_ids"][-(len(tail) + 1) : -1] == tokenizer.encode(tail)


def test_max_length_must_leave_room_for_the_answer_token() -> None:
    with pytest.raises(TrainingError):
        build_example(CharTokenizer(), make_decision(), max_length=1)


# --------------------------------------------------------- option permutation


def test_reordering_preserves_the_answer_and_the_option_set() -> None:
    decisions = [make_decision(identifier=f"d{i}", n_options=5, answer="level_3")
                 for i in range(20)]
    shuffled = reorder_options(decisions, seed=3)

    assert len(shuffled) == len(decisions)
    for before, after in zip(decisions, shuffled, strict=True):
        assert after.id == before.id
        assert after.answer_id == before.answer_id
        assert set(after.criterion.option_ids) == set(before.criterion.option_ids)
        assert after.criterion.key == before.criterion.key
        assert after.state == before.state


def test_reordering_actually_moves_the_gold_around() -> None:
    decisions = [make_decision(identifier=f"d{i}", n_options=5, answer="level_3")
                 for i in range(40)]
    positions = {d.answer_index for d in reorder_options(decisions, seed=0)}
    assert len(positions) > 1, "without this the model just learns one letter"


def test_reordering_is_deterministic_in_the_seed() -> None:
    decisions = [make_decision(identifier=f"d{i}", n_options=5) for i in range(10)]
    first = [d.criterion.option_ids for d in reorder_options(decisions, seed=11)]
    again = [d.criterion.option_ids for d in reorder_options(decisions, seed=11)]
    other = [d.criterion.option_ids for d in reorder_options(decisions, seed=12)]
    assert first == again
    assert first != other


# ------------------------------------------------------------------ dataset


def test_dataset_indexes_and_counts() -> None:
    decisions = [make_decision(identifier=f"d{i}") for i in range(5)]
    data = DecisionDataset(decisions, CharTokenizer(), 4096)
    assert len(data) == 5
    assert data[3]["labels"][-1] == data[3]["input_ids"][-1]


def test_an_empty_dataset_is_refused() -> None:
    with pytest.raises(TrainingError):
        DecisionDataset([], CharTokenizer(), 4096)


def test_randomisation_off_leaves_the_option_order_alone() -> None:
    decisions = [make_decision(identifier=f"d{i}", n_options=4) for i in range(8)]
    data = DecisionDataset(decisions, CharTokenizer(), 4096, randomise_option_order=False)
    data.set_epoch(3)
    assert [d.criterion.option_ids for d in data.decisions] == [
        d.criterion.option_ids for d in decisions
    ]


def test_option_order_is_re_randomised_per_epoch() -> None:
    decisions = [make_decision(identifier=f"d{i}", n_options=5) for i in range(12)]
    data = DecisionDataset(
        decisions, CharTokenizer(), 4096, randomise_option_order=True, seed=5
    )
    seen = []
    for epoch in range(4):
        data.set_epoch(epoch)
        assert data.epoch == epoch
        seen.append([d.criterion.option_ids for d in data.decisions])
    assert len({tuple(map(tuple, orders)) for orders in seen}) == 4


def test_the_invariant_still_holds_after_a_permutation() -> None:
    decisions = [make_decision(identifier=f"d{i}", n_options=4, answer="level_2")
                 for i in range(6)]
    tokenizer = CharTokenizer()
    data = DecisionDataset(
        decisions, tokenizer, 4096, randomise_option_order=True, seed=1
    )
    data.set_epoch(2)
    for index, decision in enumerate(data.decisions):
        example = data[index]
        prompt = full_prompt(decision.state, decision.criterion)
        slots, _ = answer_slots(tokenizer, prompt, 4)
        assert example["input_ids"] == [*tokenizer.encode(prompt), slots[decision.answer_index]]
        assert example["labels"][-1] == slots[decision.answer_index]
        assert set(example["labels"][:-1]) == {IGNORE_INDEX}


def test_preflight_resolves_the_slots_once_per_criterion() -> None:
    decisions = [make_decision(identifier=f"d{i}") for i in range(4)]
    table = DecisionDataset(decisions, CharTokenizer(), 4096).preflight()
    assert table == {"triage": {"A": ord("A"), "B": ord("B"), "C": ord("C")}}


def test_preflight_fails_loudly_before_a_long_run_starts() -> None:
    data = DecisionDataset([make_decision(identifier="bad-row")], CollidingTokenizer(), 4096)
    with pytest.raises(TrainingError) as caught:
        data.preflight()
    assert "bad-row" in str(caught.value)


# ------------------------------------------------------------------ collation


def test_pad_batch_right_pads_and_masks_the_padding() -> None:
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, -100, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
    ]
    padded = pad_batch(features, pad_token_id=0)
    assert padded["input_ids"] == [[1, 2, 3], [4, 5, 0]]
    assert padded["attention_mask"] == [[1, 1, 1], [1, 1, 0]]
    assert padded["labels"] == [[-100, -100, 3], [-100, 5, -100]]


def test_padding_never_adds_a_supervised_position() -> None:
    tokenizer = CharTokenizer()
    features = [
        build_example(tokenizer, make_decision(identifier="a", state="short"), 4096),
        build_example(tokenizer, make_decision(identifier="b", state="a much longer " * 20), 4096),
    ]
    padded = pad_batch(features, pad_token_id=0)
    for labels in padded["labels"]:
        assert sum(1 for label in labels if label != IGNORE_INDEX) == 1


def test_an_empty_batch_is_refused() -> None:
    with pytest.raises(TrainingError):
        pad_batch([], pad_token_id=0)


def test_the_collator_asks_for_the_training_extra_when_torch_is_missing() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("torch is installed, so there is no missing-extra path to test")
    with pytest.raises(TrainingError) as caught:
        AnswerTokenCollator(pad_token_id=0)([
            {"input_ids": [1], "attention_mask": [1], "labels": [1]}
        ])
    assert "typedecide[train]" in str(caught.value)


# --------------------------------------------------------------- housekeeping


def test_labelled_only_names_what_it_drops() -> None:
    keep, dropped = labelled_only([
        make_decision(identifier="ok1"),
        make_decision(identifier="missing", answer=None),
        make_decision(identifier="ok2"),
    ])
    assert [d.id for d in keep] == ["ok1", "ok2"]
    assert dropped == ["missing"]


def test_importing_training_pulls_in_no_heavy_extra() -> None:
    """The base install stays light: torch and friends are imported inside functions."""
    import subprocess

    probe = (
        f"import sys; sys.path.insert(0, {str(SRC)!r}); "
        "import typedecide.training; "
        "heavy = [m for m in ('torch', 'transformers', 'peft') if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", f"training imported {result.stdout.strip()} eagerly"
