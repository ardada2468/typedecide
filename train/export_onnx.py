"""Merge the adapter and export ONNX the browser can load.

The browser build needs the same graph shape transformers.js expects: a merged
decoder with past-key inputs. We export with Optimum, then quantise to q4f16,
which is what web/models.json asks for.
"""

import argparse
import pathlib
import shutil
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", type=pathlib.Path, default=HERE / "runs/triage-lora")
    parser.add_argument("--out", type=pathlib.Path,
                        default=HERE.parent / "web/models/qwen3-0.6b-triage-onnx")
    parser.add_argument("--keep-merged", action="store_true")
    args = parser.parse_args()

    from peft import PeftModel

    merged_dir = HERE / "runs/merged"
    print(f"merging {args.adapter} into {args.model} ...")
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(merged_dir)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"exporting ONNX to {args.out} ...")
    export = subprocess.run([
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", str(merged_dir),
        "--task", "text-generation-with-past",
        "--opset", "17",
        str(args.out),
    ])
    if export.returncode:
        sys.exit("ONNX export failed. Check that `optimum[exporters]` is installed.")

    print("\nquantise for the browser with:")
    print(f"  python -m onnxruntime.quantization.matmul_4bits_quantizer \\")
    print(f"      --input_model {args.out}/model.onnx \\")
    print(f"      --output_model {args.out}/onnx/model_q4f16.onnx --block_size 32")
    print("\nthen set \"local\": true on the triage entry in web/models.json and serve web/.")

    if not args.keep_merged:
        shutil.rmtree(merged_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
