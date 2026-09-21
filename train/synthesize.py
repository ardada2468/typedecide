"""Build a labelled triage set whose answers are correct by construction.

Every ticket is assembled from a set of slots, and each label is read back off the
slot that produced the text, so no model or human ever labels anything. That makes
the ground truth exact. It also makes it synthetic: the tickets are more regular
than real ones, and a model can learn a phrasing shortcut that will not survive
contact with production. Replace this with real labelled tickets when you have them
-- the JSONL shape is the contract, nothing downstream changes.

Option order is randomised per example. Without that a model learns "answer A",
which is exactly the failure we measured in the stock weights.
"""

import argparse
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUBRIC = json.loads((ROOT / "web" / "presets.json").read_text())["rubric"]

OPENER = {
    "billing": [
        "I was charged {amount} twice for order {order} on {date}.",
        "My card shows two payments of {amount} against order {order}.",
        "Order {order} was billed {amount} more than the price I agreed.",
        "I cancelled order {order} on {date} but the {amount} charge is still there.",
    ],
    "auth": [
        "I can't sign in to my account and the reset email never arrives.",
        "Two-factor codes for my account stopped being accepted on {date}.",
        "My login has been locked since {date} and I can't get back in.",
        "I've requested a password reset four times today and nothing comes through.",
    ],
    "shipping": [
        "Order {order} was marked delivered on {date} but nothing arrived.",
        "The tracking for order {order} has not moved since {date}.",
        "Order {order} arrived with two of the three items missing.",
        "My parcel for order {order} was left at the wrong address on {date}.",
    ],
    "sales": [
        "Before I buy, does the team plan cover more than ten seats?",
        "I'm comparing your product with two others for a {date} rollout.",
        "Can you tell me whether order {order} can be upgraded to the annual plan?",
        "We're evaluating this for a team of thirty. What does pricing look like?",
    ],
}

TONE = {
    "angry": ["This is completely unacceptable.", "I have had enough of this.",
              "Frankly this is the worst support I have dealt with.", "I want a manager, now."],
    "frustrated": ["This has been going on for a while now.", "It's getting difficult to keep waiting.",
                   "I'd really appreciate someone actually looking at this.", "This is the second time I've had to chase."],
    "neutral": ["Please let me know how to proceed.", "Could you look into this?",
                "Happy to provide anything else you need.", "Let me know what the next step is."],
    "appreciative": ["Thanks for the help so far, I know you're busy.", "No rush at all, whenever you get a chance.",
                     "I appreciate you taking a look.", "Thanks in advance, you've always been quick before."],
}

PRIOR = {
    "stated": ["I already wrote in about this on {date} and heard nothing back.",
               "I've contacted support twice about this before.",
               "This is a follow-up to the ticket I raised last week."],
    "denied": ["This is the first time I'm writing in about it.",
               "I haven't contacted you about this before now.",
               "I wanted to raise it here first rather than anywhere else."],
    "silent": [],
}

ESCALATION = [
    "If I don't hear back I'm raising a dispute with my bank.",
    "I'll be taking this up with my card issuer otherwise.",
    "At this point I'm considering cancelling our contract entirely.",
    "I'll be posting about this publicly if it isn't sorted.",
]

DEADLINE = [
    "I need this resolved by {date}.",
    "This has to be fixed before {date}.",
    "There's a hard deadline of {date} on my side.",
]

WANT = {
    "money": ["I want the charge refunded.", "Please refund me.", "I'd like my money back."],
    "fix": ["I'd rather it was just fixed.", "I just want this working again.", "Please sort the account out."],
    "both": ["I want it fixed and the charge refunded.",
             "Please correct this and refund the difference.",
             "I need both the problem solved and my money back."],
}

INCOMPLETE = [
    "I don't have the order number to hand.",
    "I can't find the reference right now.",
    "I'm not sure which of my cards it went on.",
]

ORDERS = ["A-9923", "B-1180", "C-2010", "D-3471", "E-5562", "F-6634", "G-7720", "H-8815"]
DATES = ["3 March", "14 March", "2 April", "19 April", "7 May", "21 May", "8 June", "30 June"]


def make_ticket(rng: random.Random, index: int):
    category = rng.choice(["billing", "auth", "shipping", "sales"])
    refund = rng.choice([None, "low", "high"]) if category != "sales" else None
    amount = None if refund is None else (rng.randrange(200, 5000) if refund == "low" else rng.randrange(5200, 40000))
    tone = rng.choice(["angry", "frustrated", "neutral", "appreciative"])
    deadline = rng.random() < 0.35
    prior = rng.choice(["stated", "denied", "silent"])
    escalation_odds = {"angry": 0.65, "frustrated": 0.6, "neutral": 0.2, "appreciative": 0.02}[tone]
    escalating = rng.random() < escalation_odds
    want = rng.choice(["money", "money", "fix", "both"]) if refund else rng.choice(["fix", "both"])
    complete = rng.random() > 0.4

    fill = {"order": rng.choice(ORDERS), "date": rng.choice(DATES),
            "amount": f"{amount:,}" if amount else "the amount"}

    parts = [rng.choice(OPENER[category]).format(**fill)]
    if prior != "silent":
        parts.append(rng.choice(PRIOR[prior]).format(**fill))
    if not complete:
        parts.append(rng.choice(INCOMPLETE))
    parts.append(rng.choice(WANT[want]))
    if deadline:
        parts.append(rng.choice(DEADLINE).format(**fill))
    if escalating:
        parts.append(rng.choice(ESCALATION))
    parts.append(rng.choice(TONE[tone]))
    rng.shuffle(parts[1:-1])

    text = f"Ticket #{4000 + index}\n\n" + " ".join(parts)

    if escalating and deadline:
        urgency = "p0"
    elif escalating or deadline:
        urgency = "p1"
    elif tone in ("angry", "frustrated"):
        urgency = "p2"
    else:
        urgency = "p3"

    labels = {
        "queue": category,
        "urgency": urgency,
        "tone": tone,
        "approval": "no_refund" if refund is None else ("needed" if refund == "high" else "not_needed"),
        "escalation": "yes" if escalating else "no",
        "prior_contact": {"stated": "supported", "denied": "contradicted", "silent": "undetermined"}[prior],
        "actionable": "actionable" if complete else "needs_info",
        "routing": "person" if (tone == "angry" or escalating or deadline) else "auto",
        "deadline": "yes" if deadline else "no",
        "compensation": want,
    }
    return text, labels


def rows_for(ticket_id: str, text: str, labels: dict, rng: random.Random):
    """One row per criterion, with the options shuffled so no letter prior can form."""
    for criterion in RUBRIC:
        options = criterion["options"][:]
        rng.shuffle(options)
        answer = next(i for i, o in enumerate(options) if o["id"] == labels[criterion["key"]])
        yield {
            "id": f"{ticket_id}:{criterion['key']}",
            "state": text,
            "key": criterion["key"],
            "question": criterion["question"],
            "options": options,
            "answer_index": answer,
            "answer_id": options[answer]["id"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickets", type=int, default=400)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path(__file__).parent / "data")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tickets = [make_ticket(rng, i) for i in range(args.tickets)]

    # Split by ticket, never by row: the same state must not appear on both sides.
    cut = int(len(tickets) * (1 - args.eval_fraction))
    args.out.mkdir(parents=True, exist_ok=True)

    for name, chunk, offset in (("train", tickets[:cut], 0), ("eval", tickets[cut:], cut)):
        path = args.out / f"{name}.jsonl"
        count = 0
        with path.open("w") as handle:
            for i, (text, labels) in enumerate(chunk):
                for row in rows_for(f"t{offset + i}", text, labels, rng):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
        print(f"{path}  {len(chunk)} tickets  {count} decisions")

    spread = {}
    for text, labels in tickets:
        for key, value in labels.items():
            spread.setdefault(key, {}).setdefault(value, 0)
            spread[key][value] += 1
    print("\nlabel balance (watch for a criterion that is nearly always one answer):")
    for key, counts in spread.items():
        top = max(counts.values()) / sum(counts.values())
        flag = "  <-- skewed" if top > 0.7 else ""
        print(f"  {key:<14} {dict(sorted(counts.items()))}{flag}")


if __name__ == "__main__":
    main()
