"""Regression tests for the training layer found during the hardening audit."""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from typedecide.data.augment import randomise_option_order
from typedecide.errors import ConfigError, TrainingError
from typedecide.schema import Criterion, Decision, Option
from typedecide.training.config import TrainConfig
from typedecide.training.dataset import (
    IGNORE_INDEX,
    AnswerTokenCollator,
    DecisionDataset,
    build_example,
    labelled_only,
    pad_batch,
    reorder_options,
)


def _write(tmp_path: Path, name: str, text: str | bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    return path


BASE = 'base_model: org/model\noutput_dir: runs/x\n'


# ------------------------------------------------------------------------ config


@pytest.mark.parametrize("key", ["batch_size", "grad_accum", "lora_rank", "max_length", "seed"])
def test_a_fractional_whole_number_is_refused_not_floored(tmp_path: Path, key: str) -> None:
    """int(2.7) is 2: `batch_size: 2.7` trained with 2 and recorded 2 in the manifest."""
    path = _write(tmp_path, "c.yaml", BASE + f"{key}: 2.7\n")
    with pytest.raises(ConfigError, match=f"{key} must be a whole number, got 2.7"):
        TrainConfig.from_file(path)
    assert TrainConfig.from_file(_write(tmp_path, "ok.yaml", BASE + f"{key}: 32.0\n"))


@pytest.mark.parametrize("key", ["epochs", "learning_rate", "lora_dropout"])
@pytest.mark.parametrize("value", [".nan", ".inf"])
def test_non_finite_numbers_are_refused(tmp_path: Path, key: str, value: str) -> None:
    """NaN compares False with everything, so `epochs: .nan` passed `epochs <= 0` and
    every other range check on its way to the optimiser."""
    path = _write(tmp_path, "c.yaml", BASE + f"{key}: {value}\n")
    with pytest.raises(ConfigError, match=f"{key} must be a finite number"):
        TrainConfig.from_file(path)
    with pytest.raises(ConfigError, match="must be a finite number"):
        TrainConfig(base_model="m", output_dir=Path("x"), **{key: float("nan")})


def test_a_non_numeric_field_on_a_hand_built_config_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="epochs must be a number, got '2'"):
        TrainConfig(base_model="m", output_dir=Path("x"), epochs="2")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="learning_rate must be a number, got True"):
        TrainConfig(base_model="m", output_dir=Path("x"), learning_rate=True)


def test_non_string_yaml_keys_are_named_not_a_type_error(tmp_path: Path) -> None:
    """YAML yields int and bool keys; sorting a mixed set raised TypeError."""
    path = _write(tmp_path, "c.yaml", BASE + "7: seven\nyes: true\nzeta: 1\n")
    with pytest.raises(ConfigError, match=r"Unknown training config key\(s\): 7, True, zeta"):
        TrainConfig.from_file(path)


def test_config_encoding_problems_are_config_errors(tmp_path: Path) -> None:
    latin = _write(tmp_path, "c.yaml", "base_model: caf\xe9\noutput_dir: x\n".encode("cp1252"))
    with pytest.raises(ConfigError, match="is not UTF-8"):
        TrainConfig.from_file(latin)
    bom = _write(tmp_path, "bom.json", b"\xef\xbb\xbf" + b'{"base_model": "m", "output_dir": "x"}')
    assert TrainConfig.from_file(bom).base_model == "m"
    with pytest.raises(ConfigError, match="Could not read the training config"):
        TrainConfig.from_file(tmp_path / "absent.yaml")


def test_config_files_without_a_known_suffix_and_bad_shapes(tmp_path: Path) -> None:
    absolute = (tmp_path / "elsewhere" / "x").resolve()
    as_json = _write(
        tmp_path, "train.conf", json.dumps({"base_model": "m", "output_dir": str(absolute)})
    )
    assert TrainConfig.from_file(as_json).output_dir == absolute
    as_yaml = _write(tmp_path, "train.cfg", BASE + "randomise_option_order: 'no'\nlora_alpha: 8\n")
    config = TrainConfig.from_file(as_yaml)
    assert config.output_dir == tmp_path / "runs" / "x"
    assert config.randomise_option_order is False and config.lora_alpha == 8
    assert TrainConfig.from_mapping({"base_model": "m", "output_dir": "x",
                                     "randomise_option_order": 1}).randomise_option_order is True

    for name, text, message in [
        ("list.yaml", "- a\n- b\n", "must hold a mapping"),
        ("bad.yaml", "base_model: [unclosed\n", "is not valid YAML"),
        ("bad.json", "{not json", "is not valid JSON"),
        ("missing.yaml", "base_model: m\n", "is missing output_dir"),
        ("bool.yaml", BASE + "randomise_option_order: maybe\n", "must be true or false"),
        ("num.yaml", BASE + "epochs: many\n", "epochs must be a number"),
        ("flag.yaml", BASE + "batch_size: true\n", "batch_size must be a whole number"),
        ("rate.yaml", BASE + "learning_rate: false\n", "learning_rate must be a number"),
        ("rank.yaml", BASE + "lora_rank: 0\n", "lora_rank must be at least 1"),
        ("accum.yaml", BASE + "grad_accum: 0\n", "grad_accum must be at least 1"),
        ("empty.yaml", "base_model: ' '\noutput_dir: x\n", "base_model is empty"),
    ]:
        with pytest.raises(ConfigError, match=message):
            TrainConfig.from_file(_write(tmp_path, name, text))


def test_yaml_cannot_construct_python_objects(tmp_path: Path) -> None:
    """safe_load only: a config file is data and must never be able to run code."""
    hostile = BASE + "seed: !!python/object/apply:os.system ['echo pwned']\n"
    with pytest.raises(ConfigError, match="is not valid YAML"):
        TrainConfig.from_file(_write(tmp_path, "c.yaml", hostile))


# ----------------------------------------------------------------------- dataset


def _decisions(n: int = 3) -> list[Decision]:
    options = tuple(Option(f"o{i}", f"choice {i}") for i in range(4))
    criterion = Criterion("pick", "Which?", options)
    return [Decision(id=f"d{i}", state=f"state {i}", criterion=criterion, answer_id=f"o{i % 4}")
            for i in range(n)]


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens  # named by the Tokenizer protocol; a fake ignores it
        return [ord(c) for c in text]


class UnstablePrefixTokenizer(CharTokenizer):
    """Encodes a text differently depending on how it ends, so the head is never a
    token-level prefix of the whole prompt."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = super().encode(text, add_special_tokens)
        return ids if text.rstrip(" ABCDEFGH").endswith("Answer:") else [i + 1 for i in ids]


def test_reorder_options_is_the_one_permutation_in_the_package() -> None:
    """A second, positional seeding scheme used to sit behind an unreachable ImportError
    fallback: the same seed would have produced a different order had it ever run."""
    rows = _decisions(6)
    assert reorder_options(rows, seed=4) == randomise_option_order(rows, seed=4)
    assert reorder_options(list(reversed(rows)), seed=4) == list(
        reversed(reorder_options(rows, seed=4))
    ), "a row's order must not depend on where it sits in the dataset"


def test_dataset_epochs_differ_and_replay(caplog: pytest.LogCaptureFixture) -> None:
    rows = _decisions(8)
    data = DecisionDataset(rows, CharTokenizer(), 512, randomise_option_order=True, seed=1)
    first = [d.criterion.option_ids for d in data.decisions]
    data.set_epoch(1)
    second = [d.criterion.option_ids for d in data.decisions]
    data.set_epoch(0)
    assert first != second and [d.criterion.option_ids for d in data.decisions] == first
    assert data.epoch == 0 and len(data) == 8
    assert [d.answer_id for d in data.decisions] == [d.answer_id for d in rows]

    fixed = DecisionDataset(rows, CharTokenizer(), 512)
    fixed.set_epoch(3)
    assert fixed.decisions == tuple(rows) and fixed.epoch == 3
    with pytest.raises(TrainingError, match="the dataset is empty"):
        DecisionDataset([], CharTokenizer(), 512)

    with caplog.at_level(logging.DEBUG, logger="typedecide.training.dataset"):
        assert data.preflight() == {"pick": dict(zip("ABCD", map(ord, "ABCD"), strict=True))}
    assert "state 0" not in caplog.text, "customer text must not reach the log, even at DEBUG"


def test_preflight_warns_when_one_key_has_two_option_sets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class CountSensitive(CharTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens  # named by the Tokenizer protocol; a fake ignores it
            shift = text.count("\n")
            return [ord(c) + shift for c in text]

    two = Criterion("pick", "Which?", (Option("a", "A"), Option("b", "B")))
    three = Criterion("pick", "Which one?", (Option("a", "A"), Option("b", "B"), Option("c", "C")))
    rows = [Decision(id="x", state="s", criterion=two, answer_id="a"),
            Decision(id="y", state="s", criterion=three, answer_id="a")]
    with caplog.at_level(logging.WARNING, logger="typedecide.training.dataset"):
        DecisionDataset(rows, CountSensitive(), 512).preflight()
    assert "different answer-slot ids" in caplog.text


def test_truncation_falls_back_when_the_boundary_cannot_be_found(
    caplog: pytest.LogCaptureFixture,
) -> None:
    decision = Decision(id="long", state="evidence " * 60, criterion=_decisions(1)[0].criterion,
                        answer_id="o1")
    with caplog.at_level(logging.WARNING, logger="typedecide.training.dataset"):
        example = build_example(UnstablePrefixTokenizer(), decision, 128)
    assert "could not locate the state/criterion token boundary" in caplog.text
    assert len(example["input_ids"]) == 128
    assert example["labels"][-1] == ord("B") and set(example["labels"][:-1]) == {IGNORE_INDEX}
    tail = "".join(chr(i) for i in example["input_ids"][:-1])
    assert tail.endswith("Answer:"), "the answer boundary survives any truncation"


def test_truncation_with_no_room_for_the_preamble_keeps_the_nearest_evidence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    decision = Decision(id="long", state="0123456789" * 40, criterion=_decisions(1)[0].criterion,
                        answer_id="o0")
    with caplog.at_level(logging.WARNING, logger="typedecide.training.dataset"):
        example = build_example(CharTokenizer(), decision, 100)
    assert "no room for the instruction preamble" in caplog.text
    text = "".join(chr(i) for i in example["input_ids"][:-1])
    assert len(example["input_ids"]) == 100 and text.endswith("Answer:")
    assert text[0].isdigit(), "what is kept from the head is the end of the state"


def test_build_example_refusals() -> None:
    labelled = _decisions(1)[0]
    unlabelled = Decision(id="u", state="s", criterion=labelled.criterion)
    with pytest.raises(TrainingError, match="has no answer_id"):
        build_example(CharTokenizer(), unlabelled, 64)
    with pytest.raises(TrainingError, match="room for at least the answer token"):
        build_example(CharTokenizer(), labelled, 1)
    assert labelled_only([labelled, unlabelled]) == ([labelled], ["u"])


def test_collator_needs_torch_and_right_pads(monkeypatch: pytest.MonkeyPatch) -> None:
    features = [
        {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [-100, -100, 7]},
        {"input_ids": [5, 7], "attention_mask": [1, 1], "labels": [-100, 7]},
    ]
    assert pad_batch(features, 0)["input_ids"] == [[5, 6, 7], [5, 7, 0]]
    assert pad_batch(features, 0)["labels"][1] == [-100, 7, IGNORE_INDEX]
    with pytest.raises(TrainingError, match="empty batch"):
        pad_batch([], 0)

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(TrainingError, match=r"typedecide\[train\]"):
        AnswerTokenCollator(0)(features)

    torch = types.ModuleType("torch")
    torch.long = "long"  # type: ignore[attr-defined]
    torch.tensor = lambda data, dtype: (dtype, data)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    batch: dict[str, Any] = AnswerTokenCollator(9)(features)
    assert batch["input_ids"] == ("long", [[5, 6, 7], [5, 7, 9]])
    assert batch["attention_mask"] == ("long", [[1, 1, 1], [1, 1, 0]])
