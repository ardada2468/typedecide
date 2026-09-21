"""The parts of export/prune.py that only a tensor-shaped object or a broken model
class reaches: the torch branches (against a stand-in), and every refusal in
`_verify_untied`, which exists to stop a quietly corrupted model leaving the building."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from typedecide.errors import ExportError
from typedecide.export import prune
from typedecide.export.prune import (
    SlotMap,
    answer_letter_token_ids,
    build_slot_map,
    prune_lm_head,
    read_slot_map,
    select_rows,
    validate_keep,
    write_slot_map,
)


class FakeTensor:
    """Just enough of a torch tensor: rows, fancy indexing, detach/clone, a data pointer."""

    def __init__(self, rows: list[Any], pointer: int = 0) -> None:
        self.rows, self.pointer = rows, pointer
        self.shape = (len(rows),)

    def detach(self) -> FakeTensor:
        return self

    def index_select(self, *_args: Any) -> FakeTensor:  # presence is what is probed
        return self

    def clone(self) -> FakeTensor:
        return FakeTensor([list(r) if isinstance(r, list) else r for r in self.rows], id(self))

    def data_ptr(self) -> int:
        return self.pointer

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, list):
            return FakeTensor([self.rows[i] for i in index], self.pointer)
        return self.rows[index]


class Layer:
    def __init__(self, weight: Any, bias: Any = None) -> None:
        self.weight, self.bias = weight, bias


class TensorModel:
    def __init__(self, *, tied: bool) -> None:
        table = FakeTensor([[float(r), float(r) + 0.5] for r in range(6)], pointer=111)
        other = table if tied else FakeTensor([list(r) for r in table.rows], pointer=222)
        self.config = types.SimpleNamespace(vocab_size=6, tie_word_embeddings=tied)
        self.model = types.SimpleNamespace(embed_tokens=Layer(table))
        self.lm_head = Layer(other, bias=FakeTensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]))


def _fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = types.ModuleType("torch")

    def parameter(tensor: Any, requires_grad: bool) -> Any:
        assert requires_grad is False, "a pruned head is exported, never trained"
        return tensor

    torch.nn = types.SimpleNamespace(Parameter=parameter)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_tensor_branches_copy_rows_and_bias_and_untie(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_torch(monkeypatch)
    model = TensorModel(tied=True)
    prune_lm_head(model, [4, 1])
    assert model.lm_head.weight.rows == [[4.0, 4.5], [1.0, 1.5]]
    assert model.lm_head.bias.rows == [0.4, 0.1]
    assert len(model.model.embed_tokens.weight.rows) == 6
    assert model.config.tie_word_embeddings is False
    assert select_rows(FakeTensor([[1], [2], [3]]), [2, 0]).rows == [[3], [1]]


def test_a_stale_tied_flag_is_reported_and_cleared(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _fake_torch(monkeypatch)
    model = TensorModel(tied=False)
    model.config.tie_word_embeddings = True
    with caplog.at_level(logging.WARNING, logger="typedecide.export.prune"):
        prune_lm_head(model, [0, 5])
    assert "the config was stale" in caplog.text and model.config.tie_word_embeddings is False


class _Stubborn:
    """A model class that defeats the untie in one specific way per test."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.table = [[float(r)] for r in range(5)]
        self.embed = Layer(self.table)
        self.config = _StickyConfig() if mode == "sticky-config" else types.SimpleNamespace(
            vocab_size=5, tie_word_embeddings=True
        )
        self.head = _Head(self)

    def get_output_embeddings(self) -> Any:
        return self.head

    def get_input_embeddings(self) -> Any:
        if self.mode == "no-getter":
            raise NotImplementedError
        return self.embed


class _StickyConfig:
    vocab_size = 5

    @property
    def tie_word_embeddings(self) -> bool:
        return True

    @tie_word_embeddings.setter
    def tie_word_embeddings(self, _value: bool) -> None:
        pass


class _Head:
    def __init__(self, owner: _Stubborn) -> None:
        self.owner = owner
        self._weight: Any = owner.table

    @property
    def weight(self) -> Any:
        return self._weight

    @weight.setter
    def weight(self, value: Any) -> None:
        mode = self.owner.mode
        if mode == "ignores-assignment":
            return
        self._weight = value
        if mode == "reaches-through-rows":
            self.owner.embed.weight = value
            self._weight = [list(r) for r in value]
        elif mode == "reaches-through-values":
            self.owner.embed.weight = [[r[0] + 100.0] for r in self.owner.table]
        elif mode == "still-shared":
            self.owner.embed.weight = value


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("ignores-assignment", "The slice did not take"),
        ("still-shared", "still sharing storage with the input embedding"),
        ("reaches-through-rows", "changed the input embedding from 5 to 2 rows"),
        ("reaches-through-values", "changed the values of the input embedding"),
        ("sticky-config", "tie_word_embeddings is still True after pruning"),
    ],
)
def test_a_model_the_untie_did_not_work_on_is_never_returned(mode: str, message: str) -> None:
    with pytest.raises(ExportError, match=message):
        prune_lm_head(_Stubborn(mode), [3, 1])


def test_models_without_a_reachable_input_embedding_still_prune() -> None:
    model = _Stubborn("no-getter")
    assert prune_lm_head(model, [3, 1]).head.weight == [[3.0], [1.0]]

    class WeightlessInput(_Stubborn):
        def get_input_embeddings(self) -> Any:
            return object()

    assert prune_lm_head(WeightlessInput("plain"), [0, 2]).head.weight == [[0.0], [2.0]]


def test_slot_map_refusals_and_paths(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    slot_map = build_slot_map([65, 66], model_id="m", separator=" ")
    with pytest.raises(ExportError, match="outside the 2 letters"):
        slot_map.letter_of(2)
    assert slot_map.letter_of(1) == "B" and slot_map.token_to_row == {65: 0, 66: 1}
    with pytest.raises(ExportError, match="must be a JSON object, got list"):
        SlotMap.from_dict([])  # type: ignore[arg-type]
    with pytest.raises(ExportError, match="'keep' must be a list"):
        SlotMap.from_dict({"model_id": "m", "letters": "AB", "keep": "65,66"})

    explicit = write_slot_map(slot_map, tmp_path / "custom.json")
    assert explicit.name == "custom.json" and read_slot_map(explicit) == slot_map
    (tmp_path / "latin.json").write_bytes(b'{"model_id": "caf\xe9"}')
    with pytest.raises(ExportError, match="is not valid JSON"):
        read_slot_map(tmp_path / "latin.json")

    with caplog.at_level(logging.WARNING, logger="typedecide.export.prune"):
        many = build_slot_map(list(range(25)), model_id="m", separator="", letters="x" * 25)
        validate_keep([7])
    assert many.rows == 25 and "more than the 20 answer letters" in caplog.text
    assert "this head cannot answer one" in caplog.text
    with pytest.raises(ExportError, match="appears twice") as caught:
        validate_keep([*range(30), 5])
    assert "Letters" not in str(caught.value), "past the alphabet there is no letter to name"


def test_answer_letter_token_ids_loads_only_the_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class CharTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens  # named by the Tokenizer protocol; a fake ignores it
            return [ord(c) for c in text]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> CharTokenizer:
            calls.append((model_id, kwargs))
            if model_id == "org/missing":
                raise OSError("404")
            return CharTokenizer()

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    assert answer_letter_token_ids("org/model", 4) == [65, 66, 67, 68]
    assert calls == [("org/model", {})], "no trust_remote_code"
    with pytest.raises(ExportError, match="Could not load the tokenizer for 'org/missing': 404"):
        answer_letter_token_ids("org/missing", 4)

    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ExportError, match="'transformers' is needed here"):
        answer_letter_token_ids("org/model", 4)
    assert prune._shares_storage([1], None) is False
