"""Package-level guarantees: a light import, quiet logging, and no dangerous calls."""

from __future__ import annotations

import ast
import logging
import subprocess
import sys
from pathlib import Path

import pytest

import typedecide
from typedecide import errors

SRC = Path(typedecide.__file__).resolve().parent
LIBRARY_FILES = sorted(p for p in SRC.rglob("*.py") if p.name != "cli.py")


def test_importing_the_package_pulls_in_nothing_heavy() -> None:
    probe = (
        f"import sys; sys.path.insert(0, {str(SRC.parent)!r}); "
        "import typedecide, typedecide.data, typedecide.training, typedecide.evaluation, "
        "typedecide.export, typedecide.cli; "
        "heavy = ('torch', 'transformers', 'peft', 'datasets', 'optimum', 'onnx', "
        "'onnxruntime', 'pyarrow'); "
        "print(','.join(m for m in heavy if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""


def test_package_logger_has_a_null_handler_and_nothing_else() -> None:
    handlers = logging.getLogger("typedecide").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)
    assert all(isinstance(h, logging.NullHandler) for h in handlers)


def test_every_error_is_a_typedecide_error_and_is_exported() -> None:
    names = [n for n in dir(errors) if n.endswith("Error")]
    assert "EvaluationError" in names and "ScorerError" in names
    for name in names:
        cls = getattr(errors, name)
        assert issubclass(cls, errors.TypeDecideError)
        assert getattr(typedecide, name) is cls
        assert name in typedecide.__all__


def test_scorer_error_is_both_new_and_backward_compatible() -> None:
    assert issubclass(errors.ScorerError, errors.EvaluationError)
    assert issubclass(errors.ScorerError, errors.ConfigError)
    assert not issubclass(errors.EvaluationError, errors.ConfigError)


def test_lazy_exports_resolve_and_unknown_names_fail() -> None:
    from typedecide.data import load_decisions
    from typedecide.export import ExportConfig
    from typedecide.training import TrainConfig

    assert typedecide.load_decisions is load_decisions
    assert typedecide.TrainConfig is TrainConfig
    assert typedecide.ExportConfig is ExportConfig
    for name in typedecide.__all__:
        assert getattr(typedecide, name) is not None
    assert {"finetune", "write_decisions", "EvalResult"} <= set(dir(typedecide))
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = typedecide.nope


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _dotted(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return node.id if isinstance(node, ast.Name) else ""


@pytest.mark.parametrize("path", LIBRARY_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_library_code_never_prints_or_configures_logging(path: Path) -> None:
    for call in _calls(ast.parse(path.read_text(encoding="utf-8"))):
        name = _dotted(call.func)
        assert name != "print", f"{path.name}:{call.lineno} prints; use a logger"
        assert not name.endswith("basicConfig"), f"{path.name}:{call.lineno} configures logging"


@pytest.mark.parametrize(
    "path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC))
)
def test_no_code_execution_primitives_anywhere(path: Path) -> None:
    """eval/exec/pickle/shell=True/yaml.load, and `trust_remote_code` as an argument."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
            banned = {"pickle", "marshal", "shelve", "subprocess", "dill", "cloudpickle"}
            assert not banned & {m.split(".")[0] for m in modules}, f"{path.name}:{node.lineno}"
    for call in _calls(tree):
        name = _dotted(call.func)
        where = f"{path.name}:{call.lineno}"
        assert name not in {"eval", "exec", "compile", "os.system", "os.popen"}, where
        assert name not in {"yaml.load", "yaml.unsafe_load", "yaml.full_load"}, where
        for keyword in call.keywords:
            assert keyword.arg != "trust_remote_code", f"{where} passes trust_remote_code"
            if keyword.arg == "shell":
                assert not (isinstance(keyword.value, ast.Constant) and keyword.value.value), where
