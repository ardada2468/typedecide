"""LoRA-tune the model on the one token the browser actually reads.

The browser never samples. It reads the logits at the position right after
"Answer:" and softmaxes over that criterion's answer letters. So that is the only
position this fine-tune puts loss on: every other token is masked out. Training the
model to write a sentence it will never be asked to write would spend capacity on
the wrong thing.

Refuses to start if web/prompt.js and train/prompt.py disagree, because training
against a prompt the browser does not send is a failure that looks like bad data.
"""

import argparse
import json
import pathlib
import subprocess
import sys

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)

from prompt import answer_slots, full_prompt

HERE = pathlib.Path(__file__).resolve().parent


class Decisions(Dataset):
    """Each example is: the whole prompt, masked out, plus one supervised letter."""

    def __init__(self, path, tokenizer, max_length):
        self.rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.skipped = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        criterion = {"question": row["question"], "options": row["options"]}
        prompt = full_prompt(row["state"], criterion)
        slots, _ = answer_slots(self.tokenizer, prompt, len(row["options"]))
        target = slots[row["answer_index"]]

        ids = self.tokenizer.encode(prompt, add_special_tokens=False)[-(self.max_length - 1):]
        return {
            "input_ids": ids + [target],
            "attention_mask": [1] * (len(ids) + 1),
            "labels": [-100] * len(ids) + [target],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data", type=pathlib.Path, default=HERE / "data")
    parser.add_argument("--out", type=pathlib.Path, default=HERE / "runs/triage-lora")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=768)
    args = parser.parse_args()

    if subprocess.run([sys.executable, str(HERE / "check_parity.py")]).returncode:
        sys.exit("Prompt parity failed. Fix web/prompt.js or train/prompt.py before training.")

    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.config.use_cache = False

    peft_model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    peft_model.print_trainable_parameters()

    train = Decisions(args.data / "train.jsonl", tokenizer, args.max_length)
    held = Decisions(args.data / "eval.jsonl", tokenizer, args.max_length)
    print(f"{len(train)} training decisions, {len(held)} held out")

    trainer = Trainer(
        model=peft_model,
        args=TrainingArguments(
            output_dir=str(args.out),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=25,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            bf16=torch.cuda.is_available(),
            report_to=[],
        ),
        train_dataset=train,
        eval_dataset=held,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()

    peft_model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nadapter written to {args.out}")
    print("next:")
    print(f"  python evaluate.py --model {args.model} --out runs/base.json")
    print(f"  python evaluate.py --model {args.model} --adapter {args.out} --out runs/tuned.json")
    print(f"  python export_onnx.py --adapter {args.out}")


if __name__ == "__main__":
    main()
