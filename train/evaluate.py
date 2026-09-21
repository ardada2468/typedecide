"""Score a checkpoint the way a sceptic would.

Plain accuracy hides the failure that actually matters in this design: a model that
ignores the evidence and always answers the same letter. So every decision is
scored under every rotation of its option list, and we report three things --
accuracy, how often the answer survives reordering, and which letters got picked.
A model with a letter prior shows up immediately in the last two.
"""

import argparse
import collections
import json
import pathlib

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from prompt import LETTERS, answer_slots, full_prompt


def rotations(options):
    return [options[k:] + options[:k] for k in range(len(options))]


def load(model_id, adapter, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32 if device == "cpu" else torch.bfloat16
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    return tokenizer, model.to(device).eval()


@torch.inference_mode()
def score_batch(model, tokenizer, device, prompts, slot_lists):
    """One left-padded forward pass; every prompt's answer sits at the last position."""
    encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    width = max(len(e) for e in encoded)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids, mask = [], []
    for e in encoded:
        gap = width - len(e)
        input_ids.append([pad] * gap + e)
        mask.append([0] * gap + [1] * len(e))

    ids = torch.tensor(input_ids, device=device)
    att = torch.tensor(mask, device=device)
    # Left padding means positions must come from the mask, not from arange.
    pos = (att.cumsum(-1) - 1).clamp(min=0)

    logits = model(input_ids=ids, attention_mask=att, position_ids=pos).logits[:, -1, :].float()
    return [torch.softmax(logits[i, slots], dim=-1).tolist() for i, slots in enumerate(slot_lists)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", default=None, help="LoRA directory from finetune.py")
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path(__file__).parent / "data/eval.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 means every row")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]

    tokenizer, model = load(args.model, args.adapter, args.device)

    jobs = []
    for index, row in enumerate(rows):
        for options in rotations(row["options"]):
            criterion = {"question": row["question"], "options": options}
            prompt = full_prompt(row["state"], criterion)
            slots, _ = answer_slots(tokenizer, prompt, len(options))
            jobs.append((index, prompt, slots, [o["id"] for o in options]))

    picks = collections.defaultdict(list)
    letters = collections.Counter()
    for start in range(0, len(jobs), args.batch_size):
        chunk = jobs[start : start + args.batch_size]
        probs = score_batch(model, tokenizer, args.device,
                            [c[1] for c in chunk], [c[2] for c in chunk])
        for (index, _, _, ids), p in zip(chunk, probs):
            choice = max(range(len(p)), key=p.__getitem__)
            picks[index].append(ids[choice])
            letters[LETTERS[choice]] += 1
        print(f"\r  {min(start + args.batch_size, len(jobs))}/{len(jobs)}", end="", flush=True)
    print()

    total = correct = consistent = 0
    per_key = collections.defaultdict(lambda: [0, 0])
    for index, row in enumerate(rows):
        chosen = picks[index]
        for c in chosen:
            total += 1
            hit = c == row["answer_id"]
            correct += hit
            per_key[row["key"]][0] += hit
            per_key[row["key"]][1] += 1
        consistent += len(set(chosen)) == 1

    chance = sum(1 / len(r["options"]) for r in rows) / len(rows)
    report = {
        "model": args.model, "adapter": args.adapter,
        "decisions": len(rows), "scored": total,
        "accuracy": correct / total,
        "chance": chance,
        "order_consistency": consistent / len(rows),
        "letters": dict(sorted(letters.items())),
        "per_criterion": {k: v[0] / v[1] for k, v in sorted(per_key.items())},
    }

    print(f"\nmodel              {args.model}{' + ' + args.adapter if args.adapter else ''}")
    print(f"accuracy           {report['accuracy']:.1%}   (chance {chance:.1%})")
    print(f"order consistency  {report['order_consistency']:.1%}   same answer under every rotation")
    print(f"letters picked     " + "  ".join(f"{k}:{v / total:.0%}" for k, v in report["letters"].items()))
    print("\nper criterion:")
    for key, value in report["per_criterion"].items():
        print(f"  {key:<15} {value:.1%}")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
