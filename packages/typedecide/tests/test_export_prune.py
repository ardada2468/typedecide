"""Pruning logic, exercised without torch.

Everything here runs on plain lists and a fake tokenizer. That is deliberate:
the bug this module exists to prevent -- slicing a *tied* unembedding and
silently corrupting the input embedding -- is a bug in bookkeeping, not in
linear algebra, so it can and should be caught in CI on a machine with no GPU,
no torch, and no model download.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:  # conftest.py does this too; repeated here so the file stands alone
    sys.path.insert(0, _SRC)

from typedecide.errors import ExportError  # noqa: E402
from typedecide.export.prune import (  # noqa: E402
    SLOT_MAP_FILENAME,
    SlotMap,
    answer_letter_slots,
    build_slot_map,
    lm_head_savings,
    prune_lm_head,
    read_slot_map,
    select_rows,
    validate_keep,
    write_slot_map,
)
from typedecide.schema import LETTERS  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"


def a_map(keep: list[int], **kwargs: Any) -> SlotMap:
    options: dict[str, Any] = {"model_id": MODEL, "separator": " "}
    options.update(kwargs)
    return build_slot_map(keep, **options)


# --------------------------------------------------------------- slot map shape


def test_rows_are_numbered_in_the_order_given_not_sorted() -> None:
    """Sorting would silently permute the answer letters: B would score A's token."""
    slot_map = a_map([9200, 33, 151000, 88])
    assert slot_map.keep == (9200, 33, 151000, 88)
    assert slot_map.token_to_row == {9200: 0, 33: 1, 151000: 2, 88: 3}
    assert [slot_map.token_of(row) for row in range(4)] == [9200, 33, 151000, 88]


def test_letters_default_to_the_first_k_and_pair_with_rows() -> None:
    slot_map = a_map([10, 20, 30])
    assert slot_map.letters == "ABC" == LETTERS[:3]
    assert [slot_map.letter_of(r) for r in range(3)] == ["A", "B", "C"]
    assert slot_map.rows == 3


def test_row_of_and_token_of_are_inverses() -> None:
    keep = [4, 400, 40000]
    slot_map = a_map(keep)
    for row, token in enumerate(keep):
        assert slot_map.row_of(token) == row
        assert slot_map.token_of(row) == token


def test_row_of_a_dropped_token_explains_that_the_row_is_gone() -> None:
    slot_map = a_map([4, 400])
    with pytest.raises(ExportError, match="no row for it"):
        slot_map.row_of(999)


def test_token_of_an_out_of_range_row_names_the_valid_range() -> None:
    slot_map = a_map([4, 400])
    with pytest.raises(ExportError, match=r"0\.\.1"):
        slot_map.token_of(2)
    with pytest.raises(ExportError):
        slot_map.token_of(-1)


# ------------------------------------------------------------------ validation


def test_duplicate_token_ids_are_refused_by_position_and_by_letter() -> None:
    with pytest.raises(ExportError) as caught:
        validate_keep([33, 44, 33])
    message = str(caught.value)
    assert "33" in message
    assert "0 and 2" in message
    assert "'A'" in message and "'C'" in message


def test_a_duplicate_is_refused_even_when_adjacent() -> None:
    with pytest.raises(ExportError, match="appears twice"):
        validate_keep([7, 7])


def test_negative_token_ids_are_refused() -> None:
    with pytest.raises(ExportError, match="negative"):
        validate_keep([33, -1, 44])


def test_token_ids_at_or_past_the_vocabulary_are_refused() -> None:
    with pytest.raises(ExportError) as caught:
        validate_keep([33, 151936], vocab_size=151936)
    assert "151936" in str(caught.value)
    assert "0..151935" in str(caught.value)
    # One below the ceiling is fine.
    assert validate_keep([33, 151935], vocab_size=151936) == [33, 151935]


def test_an_empty_keep_is_refused() -> None:
    with pytest.raises(ExportError, match="at least one row"):
        validate_keep([])


@pytest.mark.parametrize("bad", [3.0, "33", None, True, [33]])
def test_non_integer_token_ids_are_refused(bad: Any) -> None:
    with pytest.raises(ExportError, match="must be integers"):
        validate_keep([33, bad])


def test_an_integer_like_object_is_accepted_and_normalised() -> None:
    class Indexy:
        def __index__(self) -> int:
            return 77

    assert validate_keep([Indexy(), 78]) == [77, 78]
    assert all(type(t) is int for t in validate_keep([Indexy()]))


def test_letters_must_be_the_same_length_as_the_kept_rows() -> None:
    with pytest.raises(ExportError, match="same length"):
        a_map([1, 2, 3], letters="AB")


def test_a_slot_map_must_name_its_model() -> None:
    with pytest.raises(ExportError, match="name the model"):
        build_slot_map([1, 2], model_id="  ", separator=" ")


def test_keeping_more_rows_than_there_are_letters_needs_explicit_letters() -> None:
    too_many = list(range(len(LETTERS) + 1))
    with pytest.raises(ExportError, match="answer letters"):
        a_map(too_many)


# ---------------------------------------------------------------- round tripping


def test_dict_round_trip_preserves_every_field() -> None:
    original = a_map([100, 5, 60000], separator="", original_vocab_size=151936)
    restored = SlotMap.from_dict(original.to_dict())
    assert restored == original
    assert restored.separator == ""
    assert restored.original_vocab_size == 151936


def test_json_file_round_trip(tmp_path: Path) -> None:
    original = a_map([100, 5, 60000], original_vocab_size=151936)
    written = write_slot_map(original, tmp_path)
    assert written == tmp_path / SLOT_MAP_FILENAME
    assert read_slot_map(written) == original
    # A directory works too, because callers hold the export dir, not the file.
    assert read_slot_map(tmp_path) == original


def test_the_written_json_is_the_shape_the_browser_reads(tmp_path: Path) -> None:
    payload = json.loads(write_slot_map(a_map([33, 34]), tmp_path).read_text())
    assert payload["model_id"] == MODEL
    assert payload["separator"] == " "
    assert payload["letters"] == "AB"
    assert payload["rows"] == 2
    assert payload["keep"] == [33, 34]
    # JSON object keys are strings; the consumer must parse them back.
    assert payload["token_to_row"] == {"33": 0, "34": 1}
    assert payload["can_generate_text"] is False


def test_a_missing_slot_map_says_why_it_is_needed(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="rows are 0..K-1|rows are 0"):
        read_slot_map(tmp_path / "nowhere")


def test_a_corrupt_slot_map_is_refused(tmp_path: Path) -> None:
    path = tmp_path / SLOT_MAP_FILENAME
    path.write_text("{not json")
    with pytest.raises(ExportError, match="not valid JSON"):
        read_slot_map(path)


def test_a_hand_edited_slot_map_whose_two_directions_disagree_is_refused() -> None:
    payload = a_map([33, 34]).to_dict()
    payload["token_to_row"] = {"33": 1, "34": 0}  # someone "fixed" it by hand
    with pytest.raises(ExportError, match="disagrees"):
        SlotMap.from_dict(payload)


def test_a_future_version_is_refused_rather_than_guessed_at() -> None:
    payload = a_map([33, 34]).to_dict()
    payload["version"] = 99
    with pytest.raises(ExportError, match="version"):
        SlotMap.from_dict(payload)


def test_a_slot_map_missing_required_fields_names_them() -> None:
    with pytest.raises(ExportError, match="missing"):
        SlotMap.from_dict({"model_id": MODEL, "letters": "AB"})


# ----------------------------------------------------------------- row slicing


def test_select_rows_takes_the_named_rows_in_the_named_order() -> None:
    matrix = [[float(row * 10 + column) for column in range(3)] for row in range(6)]
    assert select_rows(matrix, [4, 0, 2]) == [
        [40.0, 41.0, 42.0],
        [0.0, 1.0, 2.0],
        [20.0, 21.0, 22.0],
    ]


def test_select_rows_returns_copies_not_views() -> None:
    matrix = [[1.0, 2.0], [3.0, 4.0]]
    taken = select_rows(matrix, [0])
    taken[0][0] = 99.0
    assert matrix[0][0] == 1.0


# ------------------------------------------------------- pruning, with a stub


class FakeLinear:
    """Stands in for `nn.Linear`: just a weight matrix and an output width."""

    def __init__(self, weight: list[list[float]]) -> None:
        self.weight = weight
        self.bias: list[float] | None = None
        self.out_features = len(weight)
        self.in_features = len(weight[0])


class FakeEmbedding:
    def __init__(self, weight: list[list[float]]) -> None:
        self.weight = weight


class FakeConfig:
    def __init__(self, vocab_size: int, tie_word_embeddings: bool) -> None:
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings


class FakeCausalLM:
    """A causal LM the way `prune_lm_head` sees one.

    When `tied` is true the head and the embedding hold the *same list object*,
    exactly as transformers holds the same Parameter. That is what makes this a
    real test of the untie rather than a test of list slicing.
    """

    def __init__(self, vocab_size: int = 8, hidden_size: int = 3, *, tied: bool = True) -> None:
        table = [
            [float(row * hidden_size + column) for column in range(hidden_size)]
            for row in range(vocab_size)
        ]
        self.config = FakeConfig(vocab_size, tied)
        self.embed = FakeEmbedding(table)
        self.lm_head = FakeLinear(table if tied else [list(r) for r in table])
        self._tied_weights_keys = ["lm_head.weight"] if tied else []

    def get_input_embeddings(self) -> FakeEmbedding:
        return self.embed

    def get_output_embeddings(self) -> FakeLinear:
        return self.lm_head


def test_pruning_a_tied_model_leaves_the_input_embedding_untouched() -> None:
    model = FakeCausalLM(vocab_size=8, hidden_size=3, tied=True)
    before = [list(row) for row in model.embed.weight]
    assert model.lm_head.weight is model.embed.weight  # the trap, present

    prune_lm_head(model, [5, 1, 7])

    assert model.embed.weight == before, "the input embedding was corrupted by the slice"
    assert len(model.embed.weight) == 8
    assert model.lm_head.weight is not model.embed.weight


def test_pruning_keeps_the_right_rows_in_the_right_order() -> None:
    model = FakeCausalLM(vocab_size=8, hidden_size=3, tied=True)
    prune_lm_head(model, [5, 1, 7])
    assert model.lm_head.weight == [
        [15.0, 16.0, 17.0],
        [3.0, 4.0, 5.0],
        [21.0, 22.0, 23.0],
    ]
    assert model.lm_head.out_features == 3


def test_pruning_unties_the_config_so_a_reload_cannot_re_tie() -> None:
    model = FakeCausalLM(tied=True)
    prune_lm_head(model, [0, 1])
    assert model.config.tie_word_embeddings is False


def test_pruning_clears_tied_weight_keys_so_save_pretrained_keeps_the_head() -> None:
    """Left set, save_pretrained drops lm_head.weight as a tied duplicate and the
    reloaded model has no head at all."""
    model = FakeCausalLM(tied=True)
    prune_lm_head(model, [0, 1])
    assert model._tied_weights_keys == []


def test_pruning_does_not_change_vocab_size() -> None:
    """The input side still needs a row per token id; only the output shrank."""
    model = FakeCausalLM(vocab_size=8, tied=True)
    prune_lm_head(model, [0, 3])
    assert model.config.vocab_size == 8


def test_pruning_an_untied_model_works_too() -> None:
    model = FakeCausalLM(vocab_size=8, hidden_size=3, tied=False)
    before = [list(row) for row in model.embed.weight]
    prune_lm_head(model, [2, 6])
    assert model.embed.weight == before
    assert model.lm_head.weight == [[6.0, 7.0, 8.0], [18.0, 19.0, 20.0]]


def test_pruning_refuses_a_token_id_the_head_has_no_row_for() -> None:
    model = FakeCausalLM(vocab_size=8, tied=True)
    with pytest.raises(ExportError, match="outside the model's vocabulary"):
        prune_lm_head(model, [0, 8])
    # And it refused before touching anything.
    assert len(model.lm_head.weight) == 8


def test_pruning_refuses_duplicates_before_touching_the_weights() -> None:
    model = FakeCausalLM(vocab_size=8, tied=True)
    with pytest.raises(ExportError, match="appears twice"):
        prune_lm_head(model, [3, 3])
    assert len(model.lm_head.weight) == 8


def test_pruning_a_model_with_no_head_says_so() -> None:
    class Headless:
        config = FakeConfig(8, False)

    with pytest.raises(ExportError, match="no output embedding"):
        prune_lm_head(Headless(), [0, 1])


def test_pruning_slices_a_bias_alongside_the_weight() -> None:
    model = FakeCausalLM(vocab_size=8, hidden_size=3, tied=False)
    model.lm_head.bias = [float(i) for i in range(8)]
    prune_lm_head(model, [6, 2])
    assert model.lm_head.bias == [6.0, 2.0]


def test_the_pruned_head_pairs_with_a_slot_map_built_from_the_same_keep() -> None:
    """The weights record nothing about which token a row came from. This pairing
    is the only thing that makes a pruned model interpretable."""
    keep = [5, 1, 7]
    model = FakeCausalLM(vocab_size=8, hidden_size=3, tied=True)
    prune_lm_head(model, keep)
    slot_map = a_map(keep, original_vocab_size=8)
    assert slot_map.rows == len(model.lm_head.weight)
    assert model.lm_head.weight[slot_map.row_of(1)] == [3.0, 4.0, 5.0]


# -------------------------------------------------- answer letters, fake tokenizer


class CharTokenizer:
    """One token per character. Nothing merges, so the natural continuation is
    the bare letter and `answer_slots` must land on separator ""."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens  # named by the Tokenizer protocol; a fake ignores it
        return [ord(c) for c in text]


class SpaceMergingTokenizer:
    """GPT-2 flavoured: a token is an optional leading space plus one character,
    so " A" is one token and the separator " " is viable."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens  # named by the Tokenizer protocol; a fake ignores it
        pieces = re.findall(r"\s?\S|\s", text)
        return [1000 + (hash(p) % 100000) for p in pieces]


def test_answer_letter_slots_picks_the_bare_letter_when_nothing_merges() -> None:
    slots, separator = answer_letter_slots(CharTokenizer(), 4)
    assert separator == ""
    assert slots == [ord("A"), ord("B"), ord("C"), ord("D")]


def test_answer_letter_slots_prefers_a_leading_space_when_the_tokenizer_merges() -> None:
    slots, separator = answer_letter_slots(SpaceMergingTokenizer(), 3)
    assert separator == " "
    assert len(slots) == 3
    assert len(set(slots)) == 3


def test_answer_letter_slots_refuses_an_option_count_it_has_no_letters_for() -> None:
    with pytest.raises(ExportError, match="outside the 2..20"):
        answer_letter_slots(CharTokenizer(), 21)
    with pytest.raises(ExportError, match="outside the 2..20"):
        answer_letter_slots(CharTokenizer(), 1)


def test_probed_slots_feed_straight_into_a_slot_map() -> None:
    slots, separator = answer_letter_slots(CharTokenizer(), 4)
    slot_map = build_slot_map(slots, model_id=MODEL, separator=separator)
    assert slot_map.letters == "ABCD"
    assert slot_map.row_of(ord("C")) == 2


# ------------------------------------------------------------------ arithmetic


def test_qwen3_0_6b_lm_head_arithmetic() -> None:
    """The numbers docs/SPEED.md quotes, pinned so they cannot drift."""
    savings = lm_head_savings(151936, 1024, 4, tied=True)
    assert savings["head_parameters"] == 155_582_464
    assert savings["kept_parameters"] == 4096
    assert savings["removed_parameters"] == 155_578_368
    # Tied: the table still serves the input embedding, so nothing leaves the file.
    assert savings["weight_bytes_removed"] == 0.0
    # fp32 logits for one position.
    assert savings["logit_bytes_per_position_before"] == 607_744
    assert savings["logit_bytes_per_position_after"] == 16
    assert savings["matmul_flops_per_position_before"] == 311_164_928
    assert savings["matmul_flops_per_position_after"] == 8192


def test_an_untied_head_actually_leaves_the_download() -> None:
    """Qwen3-4B: 151936 x 2560, untied, at 4.5 bits a weight."""
    savings = lm_head_savings(151936, 2560, 4, tied=False)
    assert savings["head_parameters"] == 388_956_160
    assert savings["removed_parameters"] == 388_945_920
    # 388,945,920 x 4.5 bits / 8 = 218,782,080 bytes = 208.6 MiB off the download.
    assert savings["weight_bytes_removed"] == pytest.approx(218_782_080.0, rel=1e-12)


def test_savings_arithmetic_refuses_nonsense() -> None:
    with pytest.raises(ExportError, match="must be positive"):
        lm_head_savings(0, 1024, 4)
    with pytest.raises(ExportError, match="between 1 and vocab_size"):
        lm_head_savings(100, 16, 101)
