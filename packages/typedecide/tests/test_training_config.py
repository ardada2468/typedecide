"""TrainConfig: parsing, validation and the manifest record. Pure stdlib."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # the package is used from a source checkout in CI
    sys.path.insert(0, str(SRC))

from typedecide.errors import ConfigError, TypeDecideError  # noqa: E402
from typedecide.training.config import (  # noqa: E402
    LORA_TARGET_MODULES,
    MIN_MAX_LENGTH,
    TrainConfig,
)


def make(**overrides: object) -> TrainConfig:
    base: dict[str, object] = {"base_model": "Qwen/Qwen3-0.6B", "output_dir": Path("runs/x")}
    base.update(overrides)
    return TrainConfig(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------- defaults


def test_defaults_match_the_contract() -> None:
    config = make()
    assert config.epochs == 2.0
    assert config.batch_size == 4
    assert config.grad_accum == 4
    assert config.learning_rate == 1e-4
    assert config.lora_rank == 16
    assert config.lora_alpha is None
    assert config.lora_dropout == 0.05
    assert config.max_length == 1024
    assert config.seed == 0
    assert config.randomise_option_order is True


def test_config_is_frozen() -> None:
    config = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 7  # type: ignore[misc]


def test_lora_alpha_defaults_to_twice_the_rank() -> None:
    assert make(lora_rank=8).effective_lora_alpha == 16
    assert make(lora_rank=8, lora_alpha=64).effective_lora_alpha == 64


def test_effective_batch_size_is_the_product() -> None:
    assert make(batch_size=4, grad_accum=8).effective_batch_size == 32


def test_output_dir_accepts_a_string() -> None:
    config = make(output_dir="runs/from-a-string")
    assert isinstance(config.output_dir, Path)
    assert config.output_dir == Path("runs/from-a-string")


# ----------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_model": "   "},
        {"epochs": 0},
        {"epochs": -1.0},
        {"batch_size": 0},
        {"grad_accum": 0},
        {"learning_rate": 0.0},
        {"learning_rate": -1e-4},
        {"lora_rank": 0},
        {"lora_alpha": 0},
        {"lora_dropout": -0.1},
        {"lora_dropout": 1.0},
        {"max_length": MIN_MAX_LENGTH - 1},
        {"seed": -1},
    ],
)
def test_unrunnable_configs_are_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        make(**overrides)


def test_config_errors_are_typedecide_errors() -> None:
    with pytest.raises(TypeDecideError):
        make(epochs=0)


def test_the_max_length_message_explains_the_answer_slot() -> None:
    with pytest.raises(ConfigError) as caught:
        make(max_length=4)
    assert "Answer:" in str(caught.value)


# -------------------------------------------------------------------- loading


def test_from_file_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps(
            {
                "base_model": "Qwen/Qwen3-0.6B",
                "output_dir": "runs/triage",
                "epochs": 3,
                "lora_rank": 32,
                "randomise_option_order": False,
            }
        ),
        encoding="utf-8",
    )
    config = TrainConfig.from_file(path)
    assert config.base_model == "Qwen/Qwen3-0.6B"
    assert config.epochs == 3.0
    assert config.lora_rank == 32
    assert config.randomise_option_order is False


def test_relative_output_dir_resolves_against_the_config_file(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps({"base_model": "m", "output_dir": "runs/here"}), encoding="utf-8"
    )
    assert TrainConfig.from_file(path).output_dir == tmp_path / "runs" / "here"


def test_absolute_output_dir_is_left_alone(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps({"base_model": "m", "output_dir": str(target)}), encoding="utf-8"
    )
    assert TrainConfig.from_file(path).output_dir == target


def test_unknown_keys_are_named(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps({"base_model": "m", "output_dir": "o", "learnign_rate": 1e-4}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as caught:
        TrainConfig.from_file(path)
    assert "learnign_rate" in str(caught.value)


def test_missing_required_keys_are_named(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    path.write_text(json.dumps({"epochs": 1}), encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        TrainConfig.from_file(path)
    message = str(caught.value)
    assert "base_model" in message
    assert "output_dir" in message


def test_a_missing_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        TrainConfig.from_file(tmp_path / "nope.json")
    assert "nope.json" in str(caught.value)


def test_malformed_json_says_so(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        TrainConfig.from_file(path)


def test_a_suffixless_file_is_tried_as_json(tmp_path: Path) -> None:
    path = tmp_path / "trainconfig"
    path.write_text(json.dumps({"base_model": "m", "output_dir": "o"}), encoding="utf-8")
    assert TrainConfig.from_file(path).base_model == "m"


def test_a_top_level_list_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    path.write_text(json.dumps([{"base_model": "m"}]), encoding="utf-8")
    with pytest.raises(ConfigError):
        TrainConfig.from_file(path)


@pytest.mark.parametrize(
    ("given", "expected"), [("yes", True), ("no", False), ("TRUE", True), (0, False), (1, True)]
)
def test_booleans_are_read_forgivingly(given: object, expected: bool) -> None:
    config = TrainConfig.from_mapping(
        {"base_model": "m", "output_dir": "o", "randomise_option_order": given}
    )
    assert config.randomise_option_order is expected


def test_a_nonsense_boolean_is_refused() -> None:
    with pytest.raises(ConfigError):
        TrainConfig.from_mapping(
            {"base_model": "m", "output_dir": "o", "randomise_option_order": "maybe"}
        )


def test_a_non_numeric_number_is_refused() -> None:
    with pytest.raises(ConfigError) as caught:
        TrainConfig.from_mapping({"base_model": "m", "output_dir": "o", "lora_rank": "sixteen"})
    assert "lora_rank" in str(caught.value)


def test_a_boolean_is_not_accepted_as_a_number() -> None:
    with pytest.raises(ConfigError):
        TrainConfig.from_mapping({"base_model": "m", "output_dir": "o", "batch_size": True})


def test_from_file_reads_yaml_when_pyyaml_is_available(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "train.yaml"
    path.write_text(
        "base_model: Qwen/Qwen3-0.6B\noutput_dir: runs/triage\nseed: 11\n", encoding="utf-8"
    )
    config = TrainConfig.from_file(path)
    assert config.seed == 11
    assert config.output_dir == tmp_path / "runs" / "triage"


# ------------------------------------------------------------------- manifest


def test_manifest_is_json_serialisable_and_fully_resolved() -> None:
    manifest = make(lora_rank=8, max_length=512, seed=3).to_manifest()
    round_tripped = json.loads(json.dumps(manifest))

    assert round_tripped["lora_alpha"] == 16, "alpha must be resolved, not left as null"
    assert round_tripped["lora_alpha_was_explicit"] is False
    assert round_tripped["output_dir"] == str(Path("runs/x"))
    assert round_tripped["max_length"] == 512
    assert round_tripped["seed"] == 3
    assert round_tripped["lora_target_modules"] == list(LORA_TARGET_MODULES)
    assert "lr_scheduler_type" in round_tripped
    assert "warmup_ratio" in round_tripped


def test_manifest_covers_every_field_of_the_config() -> None:
    manifest = make().to_manifest()
    for field in dataclasses.fields(TrainConfig):
        assert field.name in manifest, f"{field.name} is not recorded in the manifest"


def test_manifest_records_an_explicit_alpha_as_explicit() -> None:
    manifest = make(lora_rank=8, lora_alpha=99).to_manifest()
    assert manifest["lora_alpha"] == 99
    assert manifest["lora_alpha_was_explicit"] is True
