"""export/onnx.py and export/quantize.py.

`export_onnx` is driven against stand-ins for torch, transformers, peft and optimum, so
what is tested is the order of operations, the files written and the errors raised --
not that Optimum produces a loadable graph, which needs the real thing. The graph surgery
in `trim_logits_to_last_position` *is* tested for real, on a tiny hand-built ONNX model,
whenever `onnx` is installed.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from typedecide.errors import ExportError
from typedecide.export import onnx as onnx_export
from typedecide.export.onnx import (
    MANIFEST_FILENAME,
    STAGING_DIRNAME,
    ExportConfig,
    export_onnx,
    trim_logits_to_last_position,
)
from typedecide.export.prune import SLOT_MAP_FILENAME, SlotMap, read_slot_map
from typedecide.export.quantize import QUANTIZATION_MODES, quantize, resolve_mode

VOCAB, HIDDEN = 128, 3


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens  # named by the Tokenizer protocol; a fake ignores it
        return [ord(c) for c in text]

    def save_pretrained(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")


class Layer:
    def __init__(self, weight: list[list[float]]) -> None:
        self.weight = weight
        self.bias = None
        self.out_features = len(weight)


class Model:
    def __init__(self) -> None:
        table = [[float(r * HIDDEN + c) for c in range(HIDDEN)] for r in range(VOCAB)]
        self.config = types.SimpleNamespace(vocab_size=VOCAB, tie_word_embeddings=True)
        self.embed, self.lm_head = Layer(table), Layer(table)
        self._tied_weights_keys = ["lm_head.weight"]

    def get_input_embeddings(self) -> Layer:
        return self.embed

    def get_output_embeddings(self) -> Layer:
        return self.lm_head

    def save_pretrained(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "rows.txt").write_text(str(len(self.lm_head.weight)), encoding="utf-8")


def _install(
    monkeypatch: pytest.MonkeyPatch, *, export_error: Exception | None = None,
    legacy_dtype: bool = False, load_error: Exception | None = None,
    merge_error: Exception | None = None, main_export: bool = True,
) -> dict[str, Any]:
    seen: dict[str, Any] = {"model": Model(), "loads": [], "exports": [], "merged": []}

    torch = types.ModuleType("torch")
    torch.float32 = "float32"  # type: ignore[attr-defined]
    transformers = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> CharTokenizer:
            seen["loads"].append(("tokenizer", model_id, kwargs))
            return CharTokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> Model:
            if legacy_dtype and "dtype" in kwargs:
                raise TypeError("got an unexpected keyword argument 'dtype'")
            if load_error is not None:
                raise load_error
            seen["loads"].append(("model", model_id, kwargs))
            return seen["model"]

    transformers.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = AutoModelForCausalLM  # type: ignore[attr-defined]

    peft = types.ModuleType("peft")

    class PeftModel:
        @staticmethod
        def from_pretrained(model: Model, adapter: str) -> Any:
            if merge_error is not None:
                raise merge_error
            seen["merged"].append(adapter)
            return types.SimpleNamespace(merge_and_unload=lambda: model)

    peft.PeftModel = PeftModel  # type: ignore[attr-defined]

    exporters = types.ModuleType("optimum.exporters.onnx")

    def run_export(**kwargs: Any) -> None:
        staged = Path(kwargs["model_name_or_path"])
        seen["exports"].append({**kwargs, "staged_rows": (staged / "rows.txt").read_text()})
        if export_error is not None:
            raise export_error
        (Path(kwargs["output"]) / "model.onnx").write_bytes(b"onnx")

    if main_export:
        exporters.main_export = run_export  # type: ignore[attr-defined]
    for name, module in (
        ("torch", torch), ("transformers", transformers), ("peft", peft),
        ("optimum", types.ModuleType("optimum")),
        ("optimum.exporters", types.ModuleType("optimum.exporters")),
        ("optimum.exporters.onnx", exporters),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return seen


# ------------------------------------------------------------------ ExportConfig


def test_export_config_validates_and_normalises(tmp_path: Path) -> None:
    config = ExportConfig("org/model", "runs/lora", str(tmp_path), prune_head_to=[65, 66])  # type: ignore[arg-type]
    assert config.adapter == Path("runs/lora") and config.output_dir == tmp_path
    assert config.prune_head_to == (65, 66)
    assert config.to_manifest()["prune_head_to"] == [65, 66]
    assert ExportConfig("m", None, tmp_path).to_manifest()["adapter"] is None
    with pytest.raises(ExportError, match="`base_model` is empty"):
        ExportConfig("  ", None, tmp_path)
    with pytest.raises(ExportError, match="opset 11 is below 14"):
        ExportConfig("m", None, tmp_path, opset=11)
    with pytest.raises(ExportError, match="appears twice"):
        ExportConfig("m", None, tmp_path, prune_head_to=[65, 65])


# -------------------------------------------------------------------- export_onnx


def test_export_merges_then_prunes_then_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _install(monkeypatch)
    adapter = tmp_path / "lora"
    adapter.mkdir()
    out = tmp_path / "onnx"
    keep = [ord(c) for c in "ABC"]

    result = export_onnx(ExportConfig("org/model", adapter, out, prune_head_to=keep))

    assert result == out and (out / "model.onnx").exists()
    assert seen["merged"] == [str(adapter)]
    # The head was already 3 rows when Optimum saw the staged checkpoint.
    assert seen["exports"][0]["staged_rows"] == "3"
    assert seen["exports"][0]["task"] == "text-generation-with-past"
    assert seen["exports"][0]["opset"] == 17
    assert len(seen["model"].embed.weight) == VOCAB, "the input embedding keeps every row"
    assert seen["loads"] == [
        ("tokenizer", "org/model", {}),
        ("model", "org/model", {"dtype": "float32"}),
    ], "no trust_remote_code, anywhere"

    slot_map = read_slot_map(out)
    assert slot_map.keep == tuple(keep) and slot_map.letters == "ABC"
    assert slot_map.separator == "" and slot_map.original_vocab_size == VOCAB
    manifest = json.loads((out / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["pruned"] is True and manifest["can_generate_text"] is False
    assert manifest["slot_map"]["keep"] == keep
    assert not (out / STAGING_DIRNAME).exists(), "the merged checkpoint is cleaned up"


def test_unpruned_export_keeps_the_head_and_can_keep_the_staging_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _install(monkeypatch)
    out = tmp_path / "onnx"
    export_onnx(ExportConfig("org/model", None, out, keep_merged=True))
    assert seen["exports"][0]["staged_rows"] == str(VOCAB)
    assert (out / STAGING_DIRNAME / "tokenizer.json").exists()
    assert not (out / SLOT_MAP_FILENAME).exists()
    manifest = json.loads((out / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["pruned"] is False and manifest["can_generate_text"] is True


def test_kept_ids_the_tokenizer_would_not_produce_record_a_null_separator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="typedecide.export.onnx"):
        export_onnx(ExportConfig("m", None, tmp_path, prune_head_to=[10, 11, 12]))
    assert read_slot_map(tmp_path).separator is None
    assert "may be scoring different tokens" in caplog.text
    assert onnx_export._probe_separator(CharTokenizer(), [65]) is None


def test_export_failures_are_export_errors_and_clean_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, export_error=RuntimeError("unsupported op"))
    with pytest.raises(ExportError, match="The ONNX export failed: unsupported op"):
        export_onnx(ExportConfig("m", None, tmp_path))
    assert not (tmp_path / STAGING_DIRNAME).exists()
    assert not (tmp_path / MANIFEST_FILENAME).exists()

    _install(monkeypatch)
    with pytest.raises(ExportError, match="No adapter at"):
        export_onnx(ExportConfig("m", tmp_path / "absent", tmp_path))

    (tmp_path / "lora").mkdir()
    _install(monkeypatch, merge_error=ValueError("size mismatch"))
    with pytest.raises(ExportError, match="Merging .* failed: size mismatch"):
        export_onnx(ExportConfig("m", tmp_path / "lora", tmp_path))

    _install(monkeypatch, main_export=False)
    with pytest.raises(ExportError, match="has no `main_export`"):
        export_onnx(ExportConfig("m", None, tmp_path))

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ExportError, match=r"typedecide\[export\]"):
        export_onnx(ExportConfig("m", None, tmp_path))


def test_a_failed_dtype_retry_is_still_an_export_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The torch_dtype retry ran inside `except TypeError:`, and an exception raised in
    one except arm is invisible to its siblings -- so on transformers < 4.56 a missing
    model escaped as a raw OSError instead of an ExportError."""
    _install(monkeypatch, legacy_dtype=True, load_error=OSError("no such repo"))
    with pytest.raises(ExportError, match="Could not load 'org/missing': no such repo"):
        export_onnx(ExportConfig("org/missing", None, tmp_path))

    seen = _install(monkeypatch, legacy_dtype=True)
    export_onnx(ExportConfig("org/model", None, tmp_path))
    assert seen["loads"][-1] == ("model", "org/model", {"torch_dtype": "float32"})


def test_tokenizer_failure_is_an_export_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)

    def boom(_model_id: str) -> Any:
        raise OSError("offline")

    monkeypatch.setattr(sys.modules["transformers"].AutoTokenizer, "from_pretrained",
                        staticmethod(boom))
    with pytest.raises(ExportError, match="Could not load the tokenizer.*offline"):
        export_onnx(ExportConfig("m", None, tmp_path))


# ------------------------------------------------ trim_logits_to_last_position


def _tiny_decoder(path: Path, *, output_name: str = "logits", rank: int = 3) -> Path:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    shape = ["batch", "sequence", 4][:rank] if rank == 3 else ["batch", 4]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["hidden"], [output_name])],
        "tiny",
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(path))
    return path


def test_trim_makes_the_sequence_axis_one(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    source = _tiny_decoder(tmp_path / "export" / "onnx" / "model.onnx")
    written = trim_logits_to_last_position(tmp_path / "export", output_path=tmp_path / "t.onnx")

    proto = onnx.load(str(written))
    onnx.checker.check_model(proto)
    (output,) = proto.graph.output
    assert output.name == "logits"
    assert output.type.tensor_type.shape.dim[1].dim_value == 1
    assert [n.op_type for n in proto.graph.node] == ["Identity", "Slice"]
    assert proto.graph.node[0].output[0] == "logits_all_positions"
    # the source is untouched when an output path is given, and overwritten when not
    assert len(onnx.load(str(source)).graph.node) == 1
    assert trim_logits_to_last_position(source) == source
    assert len(onnx.load(str(source)).graph.node) == 2


def test_trim_evaluates_to_the_last_position(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    reference = pytest.importorskip("onnx.reference")
    import numpy as np

    written = trim_logits_to_last_position(_tiny_decoder(tmp_path / "model.onnx"))
    hidden = np.arange(2 * 5 * 4, dtype=np.float32).reshape(2, 5, 4)
    (logits,) = reference.ReferenceEvaluator(str(written)).run(None, {"hidden": hidden})
    assert logits.shape == (2, 1, 4)
    assert np.array_equal(logits[:, 0, :], hidden[:, -1, :])


def test_trim_refuses_graphs_that_are_not_decoders(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("onnx")
    with pytest.raises(ExportError, match="3-D"):
        trim_logits_to_last_position(_tiny_decoder(tmp_path / "flat.onnx", rank=2))
    with pytest.raises(ExportError, match="No ONNX model at"):
        trim_logits_to_last_position(tmp_path / "nowhere")
    with caplog.at_level(logging.WARNING, logger="typedecide.export.onnx"):
        trim_logits_to_last_position(_tiny_decoder(tmp_path / "odd.onnx", output_name="scores"))
    assert "no output named 'logits'" in caplog.text


def test_trim_without_onnx_names_the_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "onnx", None)
    with pytest.raises(ExportError, match="'onnx' is needed to export ONNX"):
        trim_logits_to_last_position(tmp_path)


def test_versions_lists_only_what_is_installed() -> None:
    assert "torch" not in onnx_export._versions() or onnx_export._versions()["torch"]


# ----------------------------------------------------------------------- quantize


def test_tiers_and_size_arithmetic() -> None:
    assert set(QUANTIZATION_MODES) == {"fp32", "fp16", "q8", "q4", "q4f16"}
    tier = resolve_mode("q4f16")
    assert tier.bits_per_weight == 4.5 and tier.bytes_per_weight == 0.5625
    assert tier.bytes_for(596_049_920) == pytest.approx(335_278_080.0)
    with pytest.raises(ExportError, match="cannot be negative"):
        tier.bytes_for(-1)
    with pytest.raises(ExportError, match="Unknown quantisation mode 'q2'.*fp16"):
        resolve_mode("q2")


def test_quantize_argument_checks(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")
    assert quantize(source, mode="fp32") == source
    assert quantize(tmp_path, mode="fp32") == source
    with pytest.raises(ExportError, match="No ONNX model at"):
        quantize(tmp_path / "absent")
    with pytest.raises(ExportError, match="positive multiple of 16"):
        quantize(source, mode="q4", block_size=20)
    with pytest.raises(ExportError, match="onto itself"):
        quantize(source, mode="q8", output_path=source)


def _fake_quant_stack(monkeypatch: pytest.MonkeyPatch, *, size: int = 10) -> dict[str, Any]:
    seen: dict[str, Any] = {"saved": [], "int8": [], "fp16": 0}

    class Proto:
        def ByteSize(self) -> int:  # noqa: N802 - protobuf's own spelling
            return size

    onnx = types.ModuleType("onnx")
    onnx.load = lambda _path: Proto()  # type: ignore[attr-defined]
    onnx.save_model = lambda _proto, path, **kw: seen["saved"].append((path, kw))  # type: ignore[attr-defined]

    class MatMul4BitsQuantizer:
        def __init__(self, proto: Any, *, block_size: int, is_symmetric: bool) -> None:
            seen["block_size"], seen["symmetric"] = block_size, is_symmetric
            self.model = types.SimpleNamespace(model=proto)

        def process(self) -> None:
            if seen.get("explode"):
                raise RuntimeError("bad graph")

    four_bit = types.ModuleType("onnxruntime.quantization.matmul_4bits_quantizer")
    four_bit.MatMul4BitsQuantizer = MatMul4BitsQuantizer  # type: ignore[attr-defined]
    quantization = types.ModuleType("onnxruntime.quantization")
    quantization.QuantType = types.SimpleNamespace(QUInt8="quint8")  # type: ignore[attr-defined]

    def quantize_dynamic(source: str, destination: str, **kwargs: Any) -> None:
        if seen.get("explode"):
            raise RuntimeError("bad graph")
        seen["int8"].append((source, destination, kwargs))

    quantization.quantize_dynamic = quantize_dynamic  # type: ignore[attr-defined]
    float16 = types.ModuleType("onnxconverter_common.float16")

    def convert(proto: Any, keep_io_types: bool) -> Any:
        assert keep_io_types is True
        seen["fp16"] += 1
        return proto

    float16.convert_float_to_float16 = convert  # type: ignore[attr-defined]
    for name, module in (
        ("onnx", onnx), ("onnxruntime", types.ModuleType("onnxruntime")),
        ("onnxruntime.quantization", quantization),
        ("onnxruntime.quantization.matmul_4bits_quantizer", four_bit),
        ("onnxconverter_common", types.ModuleType("onnxconverter_common")),
        ("onnxconverter_common.float16", float16),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return seen


def test_each_tier_calls_its_backend_and_names_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _fake_quant_stack(monkeypatch)
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")

    assert quantize(source, mode="q4f16", block_size=64).name == "model_q4f16.onnx"
    assert seen["block_size"] == 64 and seen["fp16"] == 1
    assert quantize(source, mode="q4").name == "model_q4.onnx" and seen["fp16"] == 1
    assert quantize(source, mode="fp16").name == "model_fp16.onnx" and seen["fp16"] == 2
    assert quantize(source, mode="q8", output_path=tmp_path / "deep" / "m.onnx").parent.is_dir()
    assert seen["int8"][0][2]["op_types_to_quantize"] == ["MatMul"]
    assert [Path(p).name for p, _ in seen["saved"]] == [
        "model_q4f16.onnx", "model_q4.onnx", "model_fp16.onnx"
    ]
    assert all(kw == {} for _, kw in seen["saved"])


def test_backend_failures_and_big_graphs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")
    seen = _fake_quant_stack(monkeypatch, size=2_000_000_001)
    quantize(source, mode="fp16")
    assert seen["saved"][0][1]["save_as_external_data"] is True
    assert seen["saved"][0][1]["location"] == "model_fp16.onnx.data"

    seen["explode"] = True
    with pytest.raises(ExportError, match="4-bit quantisation of .* failed: bad graph"):
        quantize(source, mode="q4")
    with pytest.raises(ExportError, match="int8 quantisation of .* failed: bad graph"):
        quantize(source, mode="q8")

    monkeypatch.delattr(
        sys.modules["onnxruntime.quantization.matmul_4bits_quantizer"], "MatMul4BitsQuantizer"
    )
    with pytest.raises(ExportError, match="onnxruntime >= 1.20"):
        quantize(source, mode="q4")

    monkeypatch.setitem(sys.modules, "onnxconverter_common.float16", None)
    monkeypatch.setitem(sys.modules, "onnxruntime.transformers.float16", None)
    with pytest.raises(ExportError, match="No float16 converter available"):
        quantize(source, mode="fp16")

    monkeypatch.setitem(sys.modules, "onnx", None)
    with pytest.raises(ExportError, match="is needed to quantise"):
        quantize(source, mode="fp16")


def test_slot_map_with_junk_rows_is_an_export_error() -> None:
    """int('x') on a hand-edited token_to_row raised a bare ValueError."""
    payload = SlotMap("m", "", "AB", (65, 66)).to_dict()
    payload["token_to_row"] = {"65": "zero", "66": 1}
    with pytest.raises(ExportError, match="disagrees with its 'keep' order"):
        SlotMap.from_dict(payload)
    payload["token_to_row"] = {"65": "0", "66": 1.0}
    assert SlotMap.from_dict(payload).keep == (65, 66)
