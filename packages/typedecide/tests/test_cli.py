"""The command line, driven in-process through `typedecide.cli.main([...])`.

Exit codes are the contract a shell script or a CI job depends on:
  0  the command did what it was asked
  1  `validate` ran and the dataset is not usable as given
  2  a TypeDecideError (or an unwritable output path): `error: ...` on stderr, no traceback
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from typedecide import cli
from typedecide.cli import build_parser, main
from typedecide.data import load_decisions

HEADER = "id,state,question,options,option_ids,answer_id,group_id\n"


def _good_csv(tmp_path: Path, rows: int = 20) -> Path:
    lines = [
        f"t{n},Ticket {n} body text,How urgent?,Can wait|Today,low|high,"
        f"{'high' if n % 2 else 'low'},case-{n}\n"
        for n in range(rows)
    ]
    path = tmp_path / "tickets.csv"
    path.write_text(HEADER + "".join(lines), encoding="utf-8")
    return path


def _customer_csv(tmp_path: Path) -> Path:
    header = "Ticket ID,Ticket Body,Decision,Choices,Chosen,Case\n"
    lines = [
        f"T-{n},Body of ticket {n},Which queue?,Billing|Shipping,"
        f"{'Billing' if n % 2 else 'Shipping'},C-{n}\n"
        for n in range(12)
    ]
    path = tmp_path / "export.csv"
    path.write_text(header + "".join(lines), encoding="utf-8")
    return path


CUSTOMER_FLAGS = [
    "--col-id", "Ticket ID", "--col-state", "Ticket Body", "--col-question", "Decision",
    "--col-options", "Choices", "--col-answer", "Chosen", "--col-group", "Case",
]


# ---------------------------------------------------------------------- validate


def test_validate_ok_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(_good_csv(tmp_path))]) == 0
    out, err = capsys.readouterr()
    assert "Dataset report: OK" in out and "decisions=20" in out
    assert err == ""


def test_validate_unusable_dataset_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "tickets.csv"
    path.write_text(
        HEADER
        + "t1,Same evidence,How urgent?,Can wait|Today,low|high,low,g1\n"
        + "t2,Same evidence,How urgent?,Can wait|Today,low|high,high,g2\n",
        encoding="utf-8",
    )
    assert main(["validate", str(path)]) == 1
    out, err = capsys.readouterr()
    assert "NOT USABLE AS GIVEN" in out and "contradictory_labels" in out
    assert "Traceback" not in out + err


def test_validate_json_out_matches_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "reports" / "nested" / "report.json"
    code = main(["validate", str(_good_csv(tmp_path, rows=4)), "--json-out", str(target),
                 "--min-per-class", "3"])
    assert code == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["ok"] is True and payload["counts"]["decisions"] == 4
    assert payload["label_balance"] == {"how_urgent": {"low": 2, "high": 2}}
    assert [f["code"] for f in payload["findings"]] == ["rare_class"]
    assert set(payload["findings"][0]) == {"severity", "code", "message", "decisions"}
    assert f"wrote {target}" in capsys.readouterr().out


def test_validate_json_out_is_written_even_when_the_dataset_fails(tmp_path: Path) -> None:
    path = tmp_path / "tickets.csv"
    path.write_text(HEADER + "t1,s1,q?,A|B,a|b,,\nt2,s2,q?,A|B,a|b,,\n", encoding="utf-8")
    target = tmp_path / "report.json"
    assert main(["validate", str(path), "--json-out", str(target)]) == 1
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert "no_labels" in [f["code"] for f in payload["findings"]]


def test_customer_column_mapping_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _customer_csv(tmp_path)
    assert main(["validate", str(source), *CUSTOMER_FLAGS]) == 0
    assert "decisions=12" in capsys.readouterr().out

    # Without the flags the same file is a diagnosed failure, not a traceback.
    assert main(["validate", str(source)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "no 'state' column" in err and "Ticket Body" in err


def test_mapping_is_none_when_no_flag_is_given() -> None:
    args = build_parser().parse_args(["validate", "x.csv"])
    assert cli._mapping_from(args) is None
    args = build_parser().parse_args(["validate", "x.csv", "--col-group", "Case"])
    mapping = cli._mapping_from(args)
    assert mapping.group == "Case" and mapping.state == "state"


def test_explicit_format_flag(tmp_path: Path) -> None:
    source = tmp_path / "tickets.dat"
    source.write_bytes(_good_csv(tmp_path).read_bytes())
    assert main(["validate", str(source)]) == 2
    assert main(["validate", str(source), "--format", "csv"]) == 0


# ------------------------------------------------------------------------- split


def test_split_writes_both_sides_without_leaking_a_group(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "data" / "split"
    code = main(["split", str(_customer_csv(tmp_path)), "--out", str(out_dir),
                 "--eval-fraction", "0.25", "--seed", "3", *CUSTOMER_FLAGS])
    assert code == 0
    train = load_decisions(out_dir / "train.jsonl")
    held = load_decisions(out_dir / "eval.jsonl")
    assert len(train) + len(held) == 12 and len(held) == 3
    assert not {d.grouping for d in train} & {d.grouping for d in held}
    assert train[0].id.startswith("T-") and train[0].group_id.startswith("C-")
    out = capsys.readouterr().out
    assert "12 decisions in 12 groups" in out and "no state appears on both sides" in out


def test_split_is_reproducible_from_its_seed(tmp_path: Path) -> None:
    source = _good_csv(tmp_path)
    for name in ("a", "b"):
        assert main(["split", str(source), "--out", str(tmp_path / name), "--seed", "7"]) == 0
    assert main(["split", str(source), "--out", str(tmp_path / "c"), "--seed", "8"]) == 0
    a, b, c = ((tmp_path / name / "eval.jsonl").read_bytes() for name in "abc")
    assert a == b and a != c


def test_split_refusal_is_exit_two_with_a_diagnosis(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["split", str(_good_csv(tmp_path)), "--out", str(tmp_path / "o"),
                 "--eval-fraction", "1.5"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: eval_fraction=1.5 must be strictly between 0 and 1")
    assert "Traceback" not in err and not (tmp_path / "o").exists()


# ----------------------------------------------------------------------- savings


def test_savings_prints_the_arithmetic(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["savings", "--vocab", "151936", "--hidden", "1024", "--keep", "4"]) == 0
    out = capsys.readouterr().out
    assert "head_parameters" in out and "155,582,464.00" in out
    assert "arithmetic, not a measurement" in out
    untied = next(line for line in out.splitlines() if "weight_bytes_removed" in line)
    assert not untied.rstrip().endswith(" 0.00")

    assert main(["savings", "--vocab", "151936", "--hidden", "1024", "--tied"]) == 0
    lines = capsys.readouterr().out.splitlines()
    tied = next(line for line in lines if "weight_bytes_removed" in line)
    assert tied.rstrip().endswith(" 0.00")


def test_savings_nonsense_is_exit_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["savings", "--vocab", "10", "--hidden", "4", "--keep", "11"]) == 2
    assert capsys.readouterr().err.startswith("error: kept_rows must be between 1 and")


# ------------------------------------------------------------------- error paths


def test_a_typedecide_error_is_one_line_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["validate", str(tmp_path / "absent.csv")]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("error: No such dataset file") and err.count("\n") == 1
    assert "Traceback" not in err


def test_an_unwritable_output_is_exit_two_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory", encoding="utf-8")
    code = main(["validate", str(_good_csv(tmp_path)), "--json-out", str(blocker / "r.json")])
    assert code == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_unexpected_exceptions_are_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only deliberate errors are turned into `error:`; a bug must still look like one."""

    def boom(_args: Any) -> int:
        raise RuntimeError("a genuine bug")

    monkeypatch.setattr(cli, "cmd_validate", boom)
    with pytest.raises(RuntimeError, match="a genuine bug"):
        main(["validate", str(tmp_path / "x.csv")])


def test_keyboard_interrupt_is_exit_130(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupt(_args: Any) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "cmd_savings", interrupt)
    assert main(["savings", "--vocab", "10", "--hidden", "4"]) == 130
    assert "interrupted" in capsys.readouterr().err


def test_usage_errors_are_argparse_exit_two(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in ([], ["nonsense"], ["split", "x.csv"], ["evaluate", "x", "--model", "m",
                                                          "--debias", "sideways"]):
        with pytest.raises(SystemExit) as caught:
            main(argv)
        assert caught.value.code == 2
    assert "usage: typedecide" in capsys.readouterr().err


def test_every_documented_subcommand_and_flag_still_parses() -> None:
    """The docs site is written against these spellings; this is the tripwire."""
    parser = build_parser()
    input_flags = ["--format", "csv", *CUSTOMER_FLAGS]
    for argv in (
        ["-v", "validate", "d.csv", "--min-per-class", "3", "--json-out", "r.json", *input_flags],
        ["split", "d.csv", "--out", "o", "--eval-fraction", "0.3", "--seed", "1", *input_flags],
        ["train", "--config", "c.yaml", "--train", "t.jsonl", "--eval", "e.jsonl", *input_flags],
        ["evaluate", "e.jsonl", "--model", "m", "--adapter", "a", "--debias", "calibrated",
         "--batch-size", "2", "--device", "cpu", "--seed", "1", "--json-out", "r.json",
         *input_flags],
        ["export", "--base", "m", "--adapter", "a", "--out", "o", "--opset", "17",
         "--prune-to", "4", "--keep-merged"],
        ["savings", "--vocab", "10", "--hidden", "4", "--keep", "2", "--tied",
         "--logit-bytes", "2"],
    ):
        assert callable(parser.parse_args(argv).func)


# --------------------------------- train / evaluate / export, with the library stubbed


def test_train_command_reports_and_points_at_evaluate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import typedecide.training as training

    config = tmp_path / "train.json"
    config.write_text(json.dumps({"base_model": "org/model", "output_dir": "runs/x"}), "utf-8")
    seen: dict[str, Any] = {}

    def fake_finetune(train: Any, cfg: Any, *, eval_set: Any, progress: Any) -> Any:
        seen.update(train=len(train), eval=None if eval_set is None else len(eval_set))
        progress({"event": "log", "step": 25, "loss": 0.6875})
        progress({"event": "epoch", "epoch": 1})
        return types.SimpleNamespace(adapter_dir=cfg.output_dir, train_loss=0.5, eval_loss=0.25)

    monkeypatch.setattr(training, "finetune", fake_finetune)
    source = _good_csv(tmp_path)
    assert main(["train", "--config", str(config), "--train", str(source),
                 "--eval", str(source)]) == 0
    out = capsys.readouterr().out
    assert seen == {"train": 20, "eval": 20}
    assert "step 25: loss 0.6875" in out and "eval loss 0.2500" in out
    assert f"--adapter {tmp_path / 'runs' / 'x'} --debias cyclic" in out

    assert main(["train", "--config", str(config), "--train", str(source)]) == 0
    assert "<eval.jsonl>" in capsys.readouterr().out
    assert main(["train", "--config", str(tmp_path / "absent.yaml"), "--train", str(source)]) == 2


def test_evaluate_command_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import typedecide.evaluation as evaluation
    from typedecide.evaluation import evaluate_with_scorer

    def fake_evaluate(decisions: Any, model_id: str, **kwargs: Any) -> Any:
        assert kwargs["adapter"] == Path("runs/lora") and kwargs["debias"] == "cyclic"
        return evaluate_with_scorer(
            decisions, lambda reqs: [[0.9, 0.1] for _ in reqs], debias=kwargs["debias"],
            manifest={"model": model_id},
        )

    monkeypatch.setattr(evaluation, "evaluate", fake_evaluate)
    target = tmp_path / "eval.json"
    assert main(["evaluate", str(_good_csv(tmp_path)), "--model", "org/model", "--adapter",
                 "runs/lora", "--debias", "cyclic", "--json-out", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["n"] == 20 and payload["order_consistency"] == 0.0
    assert payload["letter_distribution"] == {"A": 1.0, "B": 0.0}
    assert "order consistency  0.0%" in capsys.readouterr().out


def test_export_command_prunes_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import typedecide.export as export

    seen: list[Any] = []

    def fake_export(config: Any) -> Path:
        seen.append(config)
        return Path(config.output_dir)

    monkeypatch.setattr(
        export, "answer_letter_token_ids", lambda _base, n: [65 + i for i in range(n)]
    )
    monkeypatch.setattr(export, "export_onnx", fake_export)

    assert main(["export", "--base", "org/model", "--adapter", "runs/lora", "--out",
                 str(tmp_path), "--prune-to", "3", "--keep-merged"]) == 0
    assert seen[0].prune_head_to == (65, 66, 67) and seen[0].keep_merged is True
    out = capsys.readouterr().out
    assert "cannot generate text" in out and f"exported to {tmp_path}" in out

    assert main(["export", "--base", "org/model", "--out", str(tmp_path)]) == 0
    assert seen[1].prune_head_to is None and seen[1].adapter is None


def test_verbose_flag_sets_debug_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    import logging

    levels: list[int] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: levels.append(kw["level"]))
    main(["savings", "--vocab", "10", "--hidden", "4", "--keep", "2"])
    main(["-v", "savings", "--vocab", "10", "--hidden", "4", "--keep", "2"])
    assert levels == [logging.INFO, logging.DEBUG]


def test_the_module_is_runnable(tmp_path: Path) -> None:
    import subprocess

    src = Path(cli.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "typedecide.cli", "validate", str(tmp_path / "absent.csv")],
        capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(src)},
    )
    assert result.returncode == 2
    assert result.stderr.startswith("error: No such dataset file")
    assert "Traceback" not in result.stderr


def test_tied_help_names_the_sizes_that_really_tie() -> None:
    """'as every Qwen3 size does' was false: 8B and 14B do not tie their embeddings."""
    parser = build_parser()
    subparsers = next(a for a in parser._actions if a.dest == "command")
    text = " ".join(subparsers.choices["savings"].format_help().split())  # type: ignore[union-attr]
    assert "Qwen3-0.6B, 1.7B and 4B do" in text and "8B and 14B do not" in text
    assert "every Qwen3 size" not in text
    assert "--tied" in text
