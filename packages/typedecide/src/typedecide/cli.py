"""The command line.

The library is the product; this is a thin, honest wrapper over it so a first
evaluation does not require writing Python. Every subcommand maps to one public
function and prints what that function returned.

    typedecide validate  tickets.csv
    typedecide split     tickets.csv --out data/
    typedecide train     --config train.yaml --train data/train.jsonl
    typedecide evaluate  data/eval.jsonl --model Qwen/Qwen3-0.6B --debias cyclic
    typedecide export    --base Qwen/Qwen3-0.6B --adapter runs/lora --out onnx/
    typedecide savings   --vocab 151936 --hidden 1024
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .errors import TypeDecideError

LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def _mapping_from(args: argparse.Namespace) -> Any:
    from .data import FieldMapping

    overrides = {
        k: v
        for k, v in (
            ("state", args.col_state), ("question", args.col_question),
            ("options", args.col_options), ("answer", args.col_answer),
            ("group", args.col_group), ("id", args.col_id),
        )
        if v is not None
    }
    return FieldMapping(**overrides) if overrides else None


def _load(args: argparse.Namespace, source: str | Path) -> list[Any]:
    from .data import load_decisions

    return load_decisions(source, format=args.format, mapping=_mapping_from(args))


def _write_json(destination: str, payload: dict[str, Any]) -> None:
    """Write a `--json-out` file, creating its directory, always as UTF-8."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------- commands


def cmd_validate(args: argparse.Namespace) -> int:
    from .data import validate

    decisions = _load(args, args.source)
    report = validate(decisions, min_per_class=args.min_per_class)
    print(report.render())
    if args.json_out:
        _write_json(args.json_out, {
            "counts": report.counts,
            "label_balance": report.label_balance,
            "findings": [
                {"severity": f.severity, "code": f.code, "message": f.message,
                 "decisions": list(f.decisions)}
                for f in report.findings
            ],
            "ok": report.ok,
        })
        print(f"\nwrote {args.json_out}")
    return 0 if report.ok else 1


def cmd_split(args: argparse.Namespace) -> int:
    from .data import group_split, write_decisions

    decisions = _load(args, args.source)
    train, held = group_split(decisions, eval_fraction=args.eval_fraction, seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_decisions(train, out / "train.jsonl")
    write_decisions(held, out / "eval.jsonl")

    groups = len({d.grouping for d in decisions})
    print(f"{len(decisions)} decisions in {groups} groups")
    print(f"  {out / 'train.jsonl'}  {len(train)} decisions")
    print(f"  {out / 'eval.jsonl'}   {len(held)} decisions")
    print("\nSplit on groups, not rows, so no state appears on both sides.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .training import TrainConfig, finetune

    config = TrainConfig.from_file(Path(args.config))
    train = _load(args, args.train)
    held = _load(args, args.eval) if args.eval else None

    def progress(event: dict[str, Any]) -> None:
        if "loss" in event:
            step = event.get("step", "?")
            print(f"  step {step}: loss {event['loss']:.4f}", flush=True)

    result = finetune(train, config, eval_set=held, progress=progress)
    print(f"\nadapter   {result.adapter_dir}")
    print(f"manifest  {result.adapter_dir / 'manifest.json'}")
    print(f"train loss {result.train_loss:.4f}"
          + (f"   eval loss {result.eval_loss:.4f}" if result.eval_loss is not None else ""))
    print("\nNext, and do not skip it: score the base model and the tuned one the same way.")
    print(f"  typedecide evaluate {args.eval or '<eval.jsonl>'} "
          f"--model {config.base_model} --debias cyclic")
    print(f"  typedecide evaluate {args.eval or '<eval.jsonl>'} "
          f"--model {config.base_model} --adapter {result.adapter_dir} --debias cyclic")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluation import evaluate

    decisions = _load(args, args.source)
    result = evaluate(
        decisions, args.model,
        adapter=Path(args.adapter) if args.adapter else None,
        debias=args.debias, batch_size=args.batch_size, device=args.device, seed=args.seed,
    )
    print(result.render())
    if args.json_out:
        _write_json(args.json_out, {
            "accuracy": result.accuracy,
            "balanced_accuracy": result.balanced_accuracy,
            "order_consistency": result.order_consistency,
            "ece": result.ece, "brier": result.brier,
            "per_criterion": result.per_criterion,
            "letter_distribution": result.letter_distribution,
            "n": result.n, "manifest": result.manifest,
        })
        print(f"\nwrote {args.json_out}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .export import ExportConfig, answer_letter_token_ids, export_onnx

    keep = None
    if args.prune_to:
        keep = answer_letter_token_ids(args.base, args.prune_to)
        print(f"pruning the unembedding to {len(keep)} answer-letter rows: {keep}")
        print("A pruned model cannot generate text. It answers decisions and nothing else.\n")

    path = export_onnx(ExportConfig(
        base_model=args.base,
        adapter=Path(args.adapter) if args.adapter else None,
        output_dir=Path(args.out),
        opset=args.opset,
        prune_head_to=keep,
        keep_merged=args.keep_merged,
    ))
    print(f"\nexported to {path}")
    return 0


def cmd_savings(args: argparse.Namespace) -> int:
    from .export import lm_head_savings

    numbers = lm_head_savings(
        args.vocab, args.hidden, args.keep, tied=args.tied, logit_bytes=args.logit_bytes
    )
    width = max(len(k) for k in numbers)
    for key, value in numbers.items():
        rendered = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
        print(f"  {key.ljust(width)}  {rendered}")
    print("\nThese are arithmetic, not a measurement. Export a pruned model and time it.")
    return 0


# --------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="typedecide",
        description="Fine-tune small models to answer bounded decisions by logit readout.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_input_options(p: argparse.ArgumentParser) -> None:
        """Bring-your-own-data: point at columns rather than renaming them first."""
        p.add_argument("--format", choices=["jsonl", "json", "csv", "parquet", "hf"],
                       help="inferred from the file suffix when omitted")
        for role in ("state", "question", "options", "answer", "group", "id"):
            p.add_argument(f"--col-{role}", metavar="COLUMN",
                           help=f"name of the column holding the {role}")

    p = sub.add_parser("validate", help="load a dataset and report what is wrong with it")
    p.add_argument("source")
    p.add_argument("--min-per-class", type=int, default=5)
    p.add_argument("--json-out")
    add_input_options(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("split", help="split into train and eval without leaking a state")
    p.add_argument("source")
    p.add_argument("--out", required=True)
    p.add_argument("--eval-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    add_input_options(p)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("train", help="LoRA-tune on the answer token")
    p.add_argument("--config", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--eval")
    add_input_options(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="score a model the way a sceptic would")
    p.add_argument("source")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--debias", default="none",
                   choices=["none", "cyclic", "permutation", "calibrated"],
                   help="cyclic averages the option rotations, which cancels a position prior")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out")
    add_input_options(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("export", help="merge an adapter and export ONNX for the browser")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter")
    p.add_argument("--out", required=True)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--prune-to", type=int, metavar="N",
                   help="prune the unembedding to N answer-letter rows; the model can then "
                        "answer decisions but not generate text")
    p.add_argument("--keep-merged", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("savings", help="what pruning the unembedding would buy, as arithmetic")
    p.add_argument("--vocab", type=int, required=True)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--keep", type=int, default=20)
    p.add_argument("--tied", action="store_true",
                   help="the model ties its input embedding to its unembedding, as "
                        "Qwen3-0.6B, 1.7B and 4B do (8B and 14B do not); pruning then saves "
                        "compute but not download")
    p.add_argument("--logit-bytes", type=int, default=4)
    p.set_defaults(func=cmd_savings)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format=LOG_FORMAT
    )
    try:
        return int(args.func(args))
    except TypeDecideError as error:
        # Deliberate errors carry a diagnosis; a traceback would only bury it.
        print(f"error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        # An unwritable --out or --json-out: the operating system's message is the
        # diagnosis, and a traceback through pathlib adds nothing to it.
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
