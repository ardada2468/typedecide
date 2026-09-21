"""TorchReadout and the scorer contract, without torch.

The readout's batching rules -- left-pad, read position -1, derive position ids from the
mask -- are the part of this library that fails silently, so they are exercised here
against a numpy stand-in for torch. The stand-in implements only what `_score_batch`
calls; the model it drives is built to give a *different* answer if a row is padded on
the wrong side or handed arange positions, which is what makes the test worth having.
It cannot show that a real transformer agrees; only a run on real weights can.
"""

from __future__ import annotations

import contextlib
import logging
import math
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from typedecide._compat import from_pretrained_with_dtype
from typedecide.errors import ConfigError, EvaluationError, ScorerError
from typedecide.evaluation import readout as readout_module
from typedecide.evaluation.readout import ScoreRequest, TorchReadout, load_readout
from typedecide.evaluation.runner import evaluate, evaluate_with_scorer
from typedecide.schema import Criterion, Decision, Option

VOCAB = 1200


def _criterion(n: int = 3) -> Criterion:
    options = tuple(Option(f"o{i}", f"choice number {i}") for i in range(n))
    return Criterion("pick", "Which one applies?", options)


def _decisions(n: int = 5) -> list[Decision]:
    criterion = _criterion()
    return [
        Decision(id=f"d{i}", state="evidence " * (i + 1), criterion=criterion,
                 answer_id=f"o{i % 3}")
        for i in range(n)
    ]


# ------------------------------------------------------------- a numpy "torch"


class Tensor(np.ndarray):  # type: ignore[type-arg]
    def clamp(self, min: int) -> Tensor:  # noqa: A002 - torch's own keyword
        return np.maximum(self, min).view(Tensor)

    def float(self) -> Tensor:
        return self.astype(np.float64).view(Tensor)


def _fake_torch(*, cuda: bool = False) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.long, torch.float32, torch.bfloat16 = "long", "float32", "bfloat16"  # type: ignore[attr-defined]
    torch.seeds = []  # type: ignore[attr-defined]
    torch.manual_seed = torch.seeds.append  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    torch.inference_mode = contextlib.nullcontext  # type: ignore[attr-defined]

    def tensor(data: Any, dtype: Any = None, device: Any = None) -> Tensor:
        del dtype, device
        return np.asarray(data, dtype=np.int64).view(Tensor)

    def softmax(values: np.ndarray, dim: int) -> np.ndarray:  # type: ignore[type-arg]
        shifted = np.exp(values - values.max(axis=dim, keepdims=True))
        return shifted / shifted.sum(axis=dim, keepdims=True)

    torch.tensor, torch.softmax = tensor, softmax  # type: ignore[attr-defined]
    return torch


class PositionSensitiveModel:
    """Logits depend on every real token *and the position it is given*.

    Right-padding puts a pad token at index -1; arange positions shift a left-padded
    row's tokens. Either mistake changes this model's output for the shorter rows of a
    batch, so batched and one-at-a-time scores stop agreeing.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.device: str | None = None
        self.in_eval = False

    def to(self, device: str) -> PositionSensitiveModel:
        self.device = device
        return self

    def eval(self) -> PositionSensitiveModel:
        self.in_eval = True
        return self

    def __call__(self, *, input_ids: Tensor, attention_mask: Tensor, position_ids: Tensor) -> Any:
        self.calls.append({"input_ids": input_ids, "mask": attention_mask, "pos": position_ids})
        batch, width = input_ids.shape
        logits = np.zeros((batch, width, VOCAB))
        for row in range(batch):
            real = attention_mask[row] == 1
            signal = float(np.sum(input_ids[row][real] * (position_ids[row][real] + 1)))
            last = int(input_ids[row, -1])
            logits[row, -1, :] = np.sin(np.arange(VOCAB) * 0.37 + signal * 1e-3 + last)
        return types.SimpleNamespace(logits=logits.view(Tensor))


class CharTokenizer:
    def __init__(self, pad: int | None = 0, eos: int | None = 1) -> None:
        self.pad_token_id, self.eos_token_id = pad, eos
        self.encodes = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens  # named by the Tokenizer protocol; a fake ignores it
        self.encodes += 1
        return [ord(c) % VOCAB for c in text]


def _install(
    monkeypatch: pytest.MonkeyPatch, *, tokenizer: CharTokenizer | None = None,
    cuda: bool = False, legacy_dtype: bool = False,
) -> dict[str, Any]:
    seen: dict[str, Any] = {"model": PositionSensitiveModel(), "loads": [], "adapters": []}
    tokenizer = tokenizer or CharTokenizer()
    transformers = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> CharTokenizer:
            seen["loads"].append(("tokenizer", model_id, kwargs))
            return tokenizer

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> PositionSensitiveModel:
            if legacy_dtype and "dtype" in kwargs:
                raise TypeError("__init__() got an unexpected keyword argument 'dtype'")
            seen["loads"].append(("model", model_id, kwargs))
            return seen["model"]

    transformers.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = AutoModelForCausalLM  # type: ignore[attr-defined]

    peft = types.ModuleType("peft")

    class PeftModel:
        @staticmethod
        def from_pretrained(model: Any, adapter: str) -> Any:
            seen["adapters"].append(adapter)
            return types.SimpleNamespace(merge_and_unload=lambda: model)

    peft.PeftModel = PeftModel  # type: ignore[attr-defined]
    torch = _fake_torch(cuda=cuda)
    for name, module in (("torch", torch), ("transformers", transformers), ("peft", peft)):
        monkeypatch.setitem(sys.modules, name, module)
    seen["torch"], seen["tokenizer"] = torch, tokenizer
    return seen


# ----------------------------------------------------------------- the readout


def test_batched_scores_equal_one_at_a_time_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch)
    requests = [ScoreRequest(d.state, d.criterion) for d in _decisions(5)]
    batched = TorchReadout("org/model", batch_size=4)(requests)
    single = TorchReadout("org/model", batch_size=1).score(requests)
    assert np.allclose(batched, single, atol=1e-12)
    assert all(math.isclose(sum(v), 1.0) and len(v) == 3 for v in batched)

    first = seen["model"].calls[0]
    assert first["input_ids"].shape[0] == 4
    for row in range(4):
        mask = first["mask"][row]
        real = int(mask.sum())
        assert list(mask) == [0] * (len(mask) - real) + [1] * real, "padding must be on the left"
        assert list(first["pos"][row][mask == 1]) == list(range(real)), "positions restart at 0"
        assert int(first["pos"][row].min()) == 0
    assert seen["model"].in_eval and seen["model"].device == "cpu"


def test_loading_never_trusts_remote_code_and_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch)
    readout = load_readout("org/model", adapter=Path("runs/lora"), seed=9, batch_size=2)
    assert seen["loads"] == [
        ("tokenizer", "org/model", {}),
        ("model", "org/model", {"dtype": "float32"}),
    ]
    assert seen["adapters"] == [str(Path("runs/lora"))]
    assert seen["torch"].seeds == [9] and readout.pad_id == 0


def test_cuda_picks_bfloat16_and_old_transformers_gets_torch_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`dtype=` is unknown to transformers 4.51-4.55, which pyproject permits; the
    readout had no fallback and died with a TypeError on those releases."""
    seen = _install(monkeypatch, cuda=True, legacy_dtype=True)
    readout = TorchReadout("org/model")
    assert readout.device == "cuda"
    assert seen["loads"][-1] == ("model", "org/model", {"torch_dtype": "bfloat16"})


def test_dtype_fallback_does_not_swallow_an_unrelated_type_error() -> None:
    def loader(_model_id: str, **_kwargs: Any) -> Any:
        raise TypeError("config is not subscriptable")

    with pytest.raises(TypeError, match="not subscriptable"):
        from_pretrained_with_dtype(loader, "m", "float32")


def test_pad_falls_back_to_eos_then_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, tokenizer=CharTokenizer(pad=None, eos=7))
    assert TorchReadout("m").pad_id == 7
    _install(monkeypatch, tokenizer=CharTokenizer(pad=None, eos=None))
    with pytest.raises(ConfigError, match="neither a pad nor an eos token"):
        TorchReadout("m")
    with pytest.raises(ConfigError, match="batch_size must be at least 1"):
        TorchReadout("m", batch_size=0)


def test_slot_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keyed on the full prompt, so nearly every entry is unique: it grew by one prompt
    per request for the life of the readout."""
    _install(monkeypatch)
    monkeypatch.setattr(readout_module, "SLOT_CACHE_LIMIT", 3)
    readout = TorchReadout("m", batch_size=8)
    requests = [ScoreRequest(d.state, d.criterion) for d in _decisions(8)]
    readout(requests)
    assert len(readout._slot_cache) == 3
    assert list(readout._slot_cache) == [r.prompt for r in requests[-3:]]

    before = readout.tokenizer.encodes
    readout(requests[-1:])
    assert readout.tokenizer.encodes == before + 1, "a cached prompt is probed only once"


def test_over_length_prompt_is_scored_whole_and_says_so(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install(monkeypatch)
    readout = TorchReadout("m", max_length=8)
    with caplog.at_level(logging.WARNING, logger="typedecide.evaluation.readout"):
        (vector,) = readout([ScoreRequest("some evidence", _criterion())])
    assert "scored in full anyway" in caplog.text and len(vector) == 3
    assert "some evidence" not in caplog.text, "customer text must not reach the log"


def test_missing_torch_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ConfigError, match=r"typedecide\[train\]"):
        TorchReadout("m")


def test_evaluate_attaches_the_model_to_the_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)
    result = evaluate(_decisions(6), "org/model", adapter=Path("runs/lora"), debias="cyclic",
                      batch_size=4, seed=2)
    assert result.n == 6 and result.order_consistency is not None
    assert result.manifest["model"] == "org/model"
    assert result.manifest["adapter"] == str(Path("runs/lora"))
    assert result.manifest["device"] == "cpu" and result.manifest["orderings_scored"] == 3
    assert "org/model + " in result.render()
    with pytest.raises(ConfigError, match="is not a mode"):
        evaluate(_decisions(2), "org/model", debias="sideways")


# ----------------------------------------------------------- the scorer contract


def _scorer(output: Any) -> Any:
    def score(_requests: Sequence[ScoreRequest]) -> Any:
        return output

    return score


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_probabilities_are_refused(bad: float) -> None:
    """NaN is not < 0, so it passed the only value check, lost every comparison in the
    argmax, and the row was scored as 'first option' with nothing to show for it."""
    rows = _decisions(1)
    with pytest.raises(EvaluationError, match="non-finite probability"):
        evaluate_with_scorer(rows, _scorer([[0.2, bad, 0.1]]))


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ([[0.5, 0.5]], "2 probabilities for request 0, which lists 3"),
        ([], "0 probability vectors for 1 requests"),
        ([[0.5, -0.1, 0.6]], "negative probability"),
        (None, "did not return a sequence"),
        ([["a", "b", "c"]], "did not return a sequence"),
        ([0.2, 0.3, 0.5], "did not return a sequence"),
    ],
)
def test_contract_violations_are_evaluation_errors_and_still_config_errors(
    output: Any, message: str
) -> None:
    """New code catches EvaluationError; code written against 0.1.0 caught ConfigError.
    ScorerError is both, so neither breaks."""
    for expected in (EvaluationError, ConfigError, ScorerError):
        with pytest.raises(expected, match=message):
            evaluate_with_scorer(_decisions(1), _scorer(output))


def test_a_bad_mode_is_a_config_error_but_not_an_evaluation_error() -> None:
    with pytest.raises(ConfigError) as caught:
        evaluate_with_scorer(_decisions(1), _scorer([[1, 0, 0]]), debias="sideways")
    assert not isinstance(caught.value, EvaluationError)


def test_orderings_and_an_empty_request_list() -> None:
    from typedecide.evaluation.runner import _score_all, orderings

    with pytest.raises(ConfigError, match="is not a mode"):
        orderings(3, "sideways")
    assert orderings(3, "calibrated") == [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    assert _score_all(_scorer(None), []) == []


def test_zero_mass_vectors_fall_back_to_uniform() -> None:
    result = evaluate_with_scorer(_decisions(2), _scorer([[0, 0, 0], [0, 0, 0]]))
    assert result.brier == pytest.approx(2 / 3)
    calibrated = evaluate_with_scorer(
        _decisions(1), lambda requests: [[0.0, 0.0, 0.0] for _ in requests], debias="calibrated"
    )
    assert calibrated.n == 1 and calibrated.manifest["forward_passes"] == 6
