"""trainer.py without torch: the manifest, the callbacks, and the run's orchestration.

`finetune` is driven end to end against stand-ins for torch, transformers and peft. That
proves the wiring -- what is refused, what is seeded, what reaches the manifest, that the
option order is re-permuted per epoch -- and nothing about the arithmetic, which only a
real run on real weights can show.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from typedecide.errors import TrainingError
from typedecide.schema import Criterion, Decision, Option
from typedecide.training import trainer
from typedecide.training.config import TrainConfig
from typedecide.training.trainer import (
    MANIFEST_NAME,
    REDACTED,
    build_manifest,
    finetune,
    library_versions,
    redacted_argv,
    seed_everything,
    write_manifest,
)


def _decisions(n: int = 4, *, labelled: bool = True) -> list[Decision]:
    criterion = Criterion("urgent", "Is it urgent?", (Option("no", "No"), Option("yes", "Yes")))
    return [
        Decision(id=f"d{i}", state=f"ticket number {i}", criterion=criterion,
                 answer_id=("yes" if i % 2 else "no") if labelled else None, group_id=f"g{i}")
        for i in range(n)
    ]


class CharTokenizer:
    pad_token_id: int | None = 0
    pad_token: str | None = "<pad>"
    eos_token: str | None = "<eos>"

    def __init__(self) -> None:
        self.saved_to: str | None = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens  # named by the Tokenizer protocol; a fake ignores it
        return [ord(c) for c in text]

    def save_pretrained(self, path: str) -> None:
        self.saved_to = path


# ----------------------------------------------------------------------- manifest


def test_redacted_argv_blanks_credentials_and_keeps_the_rest() -> None:
    """sys.argv went into manifest.json verbatim, and the manifest ships with the
    adapter. `finetune` is a library call: argv belongs to the host program."""
    argv = ["app", "--hf-token", "hf_abcdefghijklmnopqrstuvwx", "--epochs", "2",
            "--api-key=sk-live-0123456789abcdefghij", "--out", "runs/x",
            "ghp_abcdefghijklmnopqrstuvwxyz0123", "HF_TOKEN=hf_abcdefghijklmnopqrstuvwx"]
    assert redacted_argv(argv) == [
        "app", "--hf-token", REDACTED, "--epochs", "2", f"--api-key={REDACTED}",
        "--out", "runs/x", REDACTED, f"HF_TOKEN={REDACTED}",
    ]
    assert redacted_argv(["typedecide", "train", "--config", "c.yaml"]) == [
        "typedecide", "train", "--config", "c.yaml"
    ]


def test_manifest_records_inputs_and_never_a_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["host", "--token", "hf_abcdefghijklmnopqrstuvwx"])
    config = TrainConfig(base_model="org/model", output_dir=tmp_path, seed=3)
    train, held = _decisions(4), _decisions(2)
    manifest = build_manifest(
        config, train=train, eval_set=held, answer_slots={"urgent": {"A": 65, "B": 66}},
        started_at=1_700_000_000.0, finished_at=1_700_000_012.5, extra={"steps": 9},
    )
    assert manifest["dataset"]["train"]["rows"] == 4
    assert manifest["dataset"]["eval"]["groups"] == 2
    assert manifest["dataset"]["train"]["fingerprint"] != manifest["dataset"]["eval"]["fingerprint"]
    assert manifest["timing"] == {
        "started_at": "2023-11-14T22:13:20+00:00",
        "finished_at": "2023-11-14T22:13:32+00:00",
        "wall_seconds": 12.5,
    }
    assert manifest["seed"] == 3 and manifest["steps"] == 9
    assert manifest["option_order_randomisation"] == "per-epoch"
    assert "hf_" not in json.dumps(manifest)

    path = write_manifest(manifest, tmp_path / "deep" / "er")
    assert path.name == MANIFEST_NAME
    assert json.loads(path.read_text(encoding="utf-8")) == manifest

    bare = build_manifest(
        TrainConfig(base_model="m", output_dir=tmp_path, randomise_option_order=False),
        train=train, eval_set=None, answer_slots={}, started_at=0.0, finished_at=1.0,
    )
    assert bare["dataset"]["eval"] is None and bare["option_order_randomisation"] == "off"


def test_library_versions_names_absent_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "peft", None)
    versions = library_versions()
    assert versions["peft"] == "not installed"
    assert versions["numpy"] not in {"not installed", "unknown"}
    assert versions["typedecide"]


def test_seed_everything_without_torch_seeds_python_and_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import random

    import numpy

    monkeypatch.setitem(sys.modules, "torch", None)
    seed_everything(11)
    first = (random.random(), float(numpy.random.rand()))
    seed_everything(11)
    assert (random.random(), float(numpy.random.rand())) == first


def test_missing_training_extras_are_named(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("torch", "transformers", "peft"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(TrainingError, match=r"torch, transformers, peft.*typedecide\[train\]"):
        finetune(_decisions(), TrainConfig(base_model="m", output_dir=Path("x")))


# ---------------------------------------------------------------------- callbacks


class _Dataset:
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


FAKE_TRANSFORMERS = types.SimpleNamespace(TrainerCallback=object)


def test_epoch_callback_rounds_the_accumulated_float() -> None:
    """state.epoch accumulates step fractions and reads 0.9999... at the start of epoch
    1; int() floored it, so epoch 1 replayed epoch 0's option order."""
    dataset, events = _Dataset(), []
    callback = trainer._epoch_callback(FAKE_TRANSFORMERS, dataset, events.append)  # type: ignore[arg-type]
    for epoch in (None, 0.0, 0.9999999999, 2.0000000001):
        callback.on_epoch_begin(None, types.SimpleNamespace(epoch=epoch), None, model=None)
    assert dataset.epochs == [0, 0, 1, 2]
    assert [e["epoch"] for e in events] == [0, 0, 1, 2]


def test_a_raising_progress_callback_cannot_kill_the_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """on_log called `progress` directly, bypassing the guard every other event had."""

    def broken(_event: dict[str, Any]) -> None:
        raise RuntimeError("progress bar fell over")

    callback = trainer._log_callback(FAKE_TRANSFORMERS, broken)
    with caplog.at_level(logging.ERROR, logger="typedecide.training.trainer"):
        callback.on_log(None, types.SimpleNamespace(global_step=25), None, logs={"loss": 0.5})
    assert "progress callback raised" in caplog.text

    seen: list[dict[str, Any]] = []
    trainer._log_callback(FAKE_TRANSFORMERS, seen.append).on_log(
        None, types.SimpleNamespace(global_step=50), None, logs=None
    )
    assert seen == [{"event": "log", "step": 50}]


# ---------------------------------------------------------------------- tokenizer


def _fake_transformers(tokenizer: Any = None, error: Exception | None = None) -> types.ModuleType:
    module = types.ModuleType("transformers")

    class AutoTokenizer:
        calls: list[dict[str, Any]] = []

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> Any:
            cls.calls.append({"model_id": model_id, **kwargs})
            if error is not None:
                raise error
            return tokenizer

    module.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    return module


def test_tokenizer_without_pad_borrows_eos(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoPad(CharTokenizer):
        pad_token_id = None

        @property  # type: ignore[override]
        def pad_token(self) -> str | None:
            return None

        @pad_token.setter
        def pad_token(self, value: str | None) -> None:
            type(self).pad_token_id = 2 if value else None

    fake = _fake_transformers(NoPad())
    monkeypatch.setitem(sys.modules, "transformers", fake)
    assert trainer._load_tokenizer("org/model").pad_token_id == 2
    assert fake.AutoTokenizer.calls == [{"model_id": "org/model"}], "no trust_remote_code"


def test_tokenizer_with_neither_pad_nor_eos_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """pad_token = eos_token = None went on to torch.tensor([... None ...]) mid-run."""

    class Nothing(CharTokenizer):
        pad_token_id = None
        eos_token = None

    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(Nothing()))
    with pytest.raises(TrainingError, match="neither a pad token nor an eos token"):
        trainer._load_tokenizer("org/model")


def test_tokenizer_load_failure_is_a_training_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_transformers(error=OSError("404 not found"))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    with pytest.raises(TrainingError, match="Could not load a tokenizer.*404"):
        trainer._load_tokenizer("org/missing")


# ------------------------------------------------------------ finetune, end to end


class _Param:
    def __init__(self, n: int, requires_grad: bool) -> None:
        self.n, self.requires_grad = n, requires_grad

    def numel(self) -> int:
        return self.n


class _Model:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(use_cache=True)
        self.saved_to: str | None = None

    def parameters(self) -> list[_Param]:
        return [_Param(1000, False), _Param(24, True)]

    def save_pretrained(self, path: str) -> None:
        self.saved_to = path


def _install_fakes(monkeypatch: pytest.MonkeyPatch, *, epochs: int = 2) -> dict[str, Any]:
    seen: dict[str, Any] = {"model_kwargs": [], "orders": [], "seeds": []}
    tokenizer, model = CharTokenizer(), _Model()

    torch = types.ModuleType("torch")
    torch.float32, torch.bfloat16 = "float32", "bfloat16"  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    torch.manual_seed = seen["seeds"].append  # type: ignore[attr-defined]

    transformers = _fake_transformers(tokenizer)
    transformers.set_seed = lambda _seed: None  # type: ignore[attr-defined]
    transformers.TrainerCallback = object  # type: ignore[attr-defined]

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _Model:
            seen["model_kwargs"].append({"model_id": model_id, **kwargs})
            return model

    class TrainingArguments:
        def __init__(self, **kwargs: Any) -> None:
            seen["arguments"] = kwargs

    class Trainer:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            seen["trainer"] = kwargs

        def train(self) -> Any:
            dataset = self.kwargs["train_dataset"]
            for epoch in range(epochs):
                for callback in self.kwargs["callbacks"]:
                    if hasattr(callback, "on_epoch_begin"):
                        state = types.SimpleNamespace(epoch=float(epoch))
                        callback.on_epoch_begin(None, state, None)
                seen["orders"].append([d.criterion.option_ids for d in dataset.decisions])
                _ = [dataset[i] for i in range(len(dataset))]
                for callback in self.kwargs["callbacks"]:
                    if hasattr(callback, "on_log"):
                        callback.on_log(None, types.SimpleNamespace(global_step=epoch + 1), None,
                                        logs={"loss": 0.7 - 0.1 * epoch})
            return types.SimpleNamespace(training_loss=0.42, global_step=epochs)

        def evaluate(self) -> dict[str, float]:
            return {"eval_loss": 0.5}

    transformers.AutoModelForCausalLM = AutoModelForCausalLM  # type: ignore[attr-defined]
    transformers.TrainingArguments = TrainingArguments  # type: ignore[attr-defined]
    transformers.Trainer = Trainer  # type: ignore[attr-defined]

    peft = types.ModuleType("peft")
    peft.LoraConfig = lambda **kwargs: kwargs  # type: ignore[attr-defined]

    def get_peft_model(base: _Model, lora: dict[str, Any]) -> _Model:
        seen["lora"] = lora
        return base

    peft.get_peft_model = get_peft_model  # type: ignore[attr-defined]

    for name, module in (("torch", torch), ("transformers", transformers), ("peft", peft)):
        monkeypatch.setitem(sys.modules, name, module)
    seen["tokenizer"], seen["model"] = tokenizer, model
    return seen


def test_finetune_wires_the_run_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_fakes(monkeypatch)
    out = tmp_path / "adapter"
    config = TrainConfig(base_model="org/model", output_dir=out, seed=5, max_length=512)
    events: list[dict[str, Any]] = []
    held = [*_decisions(2), *_decisions(1, labelled=False)]

    result = finetune(_decisions(6), config, eval_set=held, progress=events.append)

    assert result.adapter_dir == out and result.train_loss == 0.42 and result.eval_loss == 0.5
    assert seen["model"].saved_to == str(out) and seen["tokenizer"].saved_to == str(out)
    assert seen["model"].config.use_cache is False
    # dtype in the current spelling, and never trust_remote_code
    assert seen["model_kwargs"] == [{"model_id": "org/model", "dtype": "float32"}]
    assert seen["lora"]["r"] == 16 and seen["lora"]["lora_alpha"] == 32
    assert seen["arguments"]["seed"] == seen["arguments"]["data_seed"] == 5
    assert seen["arguments"]["eval_strategy"] == "epoch"
    assert 5 in seen["seeds"]

    on_disk = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert on_disk == result.manifest
    assert on_disk["dataset"]["train"]["rows"] == 6 and on_disk["dataset"]["eval"]["rows"] == 2
    assert on_disk["answer_slots"] == {"urgent": {"A": ord("A"), "B": ord("B")}}
    assert on_disk["trainable_parameters"] == {"trainable": 24, "total": 1024}
    assert on_disk["losses"] == {"train": 0.42, "eval": 0.5} and on_disk["steps"] == 2

    assert [e["event"] for e in events] == [
        "start", "preflight", "epoch", "log", "epoch", "log", "done"
    ]
    # The option order really is re-drawn each epoch, and the eval set is left alone.
    assert seen["orders"][0] != seen["orders"][1]
    held_out = seen["trainer"]["eval_dataset"].decisions
    assert all(d.criterion.option_ids == ("no", "yes") for d in held_out)


def test_finetune_without_eval_or_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_fakes(monkeypatch, epochs=1)
    result = finetune(_decisions(4), TrainConfig(base_model="m", output_dir=tmp_path))
    assert result.eval_loss is None and seen["arguments"]["eval_strategy"] == "no"
    assert len(seen["trainer"]["callbacks"]) == 1


def test_finetune_refuses_unlabelled_training_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)
    config = TrainConfig(base_model="m", output_dir=Path("unused"))
    with pytest.raises(TrainingError, match="2 of 6 training decisions have no answer_id"):
        finetune([*_decisions(4), *_decisions(2, labelled=False)], config)
    with pytest.raises(TrainingError, match="no labelled decisions to score"):
        finetune(_decisions(4), config, eval_set=_decisions(2, labelled=False))
    with pytest.raises(TrainingError, match="no labelled decisions to train on"):
        finetune([], config)


def test_model_and_lora_failures_are_training_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fakes(monkeypatch)
    config = TrainConfig(base_model="org/model", output_dir=tmp_path)

    def no_attach(_model: Any, _lora: Any) -> Any:
        raise ValueError("Target modules not found")

    monkeypatch.setattr(sys.modules["peft"], "get_peft_model", no_attach)
    with pytest.raises(TrainingError, match="LoRA could not attach.*q_proj"):
        finetune(_decisions(4), config)

    def no_model(_model_id: str, **_kwargs: Any) -> Any:
        raise OSError("weights missing")

    monkeypatch.setattr(
        sys.modules["transformers"].AutoModelForCausalLM, "from_pretrained", staticmethod(no_model)
    )
    with pytest.raises(TrainingError, match="as a causal LM: weights missing"):
        finetune(_decisions(4), config)
