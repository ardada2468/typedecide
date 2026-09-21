# Your first dataset

## The shape of a decision

Everything in the library speaks one type, [`Decision`](../reference/api/schema.md):

| Field | Type | Meaning |
|---|---|---|
| `id` | `str` | Unique per row. If you omit it, the loader derives a content hash, so re-importing the same rows is idempotent. |
| `state` | `str` | The evidence the model reads: a ticket body, a claim and its source, a log excerpt. |
| `criterion` | `Criterion` | A `key`, the `question`, and 2 to 20 `options`. |
| `answer_id` | `str` or `None` | The **id** of the correct option. Never a position. `None` means unlabelled. |
| `group_id` | `str` or `None` | Rows sharing a `group_id` are never split across train and eval. Unset means the row is its own group. |
| `meta` | `dict` | Any input field whose name starts with `meta_` is carried through untouched. |

Each `Option` has an `id` (what your code receives) and a `description` (what the
model reads). The id is never put in the prompt.

!!! note "Identifier rules"
    Option ids and criterion keys must match `^[a-z0-9][a-z0-9_.-]{0,63}$`: lowercase
    alphanumerics plus `_`, `.` and `-`, at most 64 characters. When the CSV loader
    has to invent ids from option text, it slugs them to fit
    (`"Billing and payments"` becomes `billing_and_payments`). A criterion needs
    between 2 and 20 options, because 20 answer letters (`A` to `T`) are rendered.

## The canonical file format: JSONL

One JSON object per line. This is what `typedecide split` writes, and what
`write_decisions` produces for any path that does not end in `.json`.

```json
{"id": "T-0-queue", "group_id": "T-0", "state": "I was charged twice for order A-1000. Please refund me.", "key": "queue", "question": "Which queue should handle this ticket?", "options": [{"id": "billing", "description": "Billing and payments"}, {"id": "shipping", "description": "Shipping and delivery"}, {"id": "access", "description": "Account access and authentication"}], "answer_id": "billing"}
```

Required: `state`, `question`, `options`. Everything else is optional. If `key` is
absent, it is derived by slugging the question.

You do not have to use this format. CSV in two shapes, JSON arrays, Parquet and
Hugging Face datasets are covered in
[Bring your own data](../guides/bring-your-own-data.md).

## Generate a practice dataset

This script writes 60 synthetic tickets with two criteria asked about each, so 120
decisions in 60 groups. It uses only the standard library.

```python title="make_tickets.py"
"""Write a small synthetic dataset: 60 tickets, two criteria asked about each."""
import json
import random

rng = random.Random(0)

QUEUE = {
    "key": "queue",
    "question": "Which queue should handle this ticket?",
    "options": [
        {"id": "billing", "description": "Billing and payments"},
        {"id": "shipping", "description": "Shipping and delivery"},
        {"id": "access", "description": "Account access and authentication"},
    ],
}
REFUND = {
    "key": "refund_requested",
    "question": "Does the customer explicitly ask for money back?",
    "options": [
        {"id": "yes", "description": "Yes"},
        {"id": "no", "description": "No"},
    ],
}
BODIES = {
    "billing": ["I was charged twice for order {n}.",
                "The invoice for order {n} shows the wrong amount."],
    "shipping": ["Order {n} has said out for delivery for six days.",
                 "The courier left order {n} at the wrong address."],
    "access": ["I cannot log in since the password reset, order {n} is stuck.",
               "Two-factor codes never arrive, so I cannot see order {n}."],
}
ASKS = {"yes": " Please refund me.", "no": " Please look into it."}

with open("tickets.jsonl", "w", encoding="utf-8") as out:
    for i in range(60):
        queue = ("billing", "shipping", "access")[i % 3]
        refund = ("yes", "no")[(i // 3) % 2]
        state = rng.choice(BODIES[queue]).format(n=f"A-{1000 + i}") + ASKS[refund]
        for criterion, answer in ((QUEUE, queue), (REFUND, refund)):
            out.write(json.dumps({
                "id": f"T-{i}-{criterion['key']}",
                "group_id": f"T-{i}",          # both criteria share the ticket
                "state": state,
                **criterion,
                "answer_id": answer,
            }) + "\n")
```

```bash
python make_tickets.py
```

!!! warning "This dataset is for learning the tool, not for learning anything else"
    It is templated, so a model can solve it from surface keywords. It exists so every
    command in the next page runs and produces output you can compare with. Do not
    read anything into accuracy numbers obtained on it.

## Why `group_id` matters already

Both criteria about ticket `T-0` share that ticket's text. If a row-level split put
`T-0-queue` in train and `T-0-refund_requested` in eval, the model would have read
the eval evidence during training. Setting `group_id` to the ticket id is what stops
that. [Group-aware splitting](../concepts/group-splitting.md) has the full argument.

## Build decisions in code

You can skip files entirely:

```python
from typedecide import Criterion, Decision, Option

queue = Criterion(
    key="queue",
    question="Which queue should handle this ticket?",
    options=(
        Option(id="billing", description="Billing and payments"),
        Option(id="shipping", description="Shipping and delivery"),
    ),
)
decision = Decision(
    id="T-1",
    state="I was charged twice for order A-9923.",
    criterion=queue,
    answer_id="billing",
    group_id="case-1",
)

print(decision.answer_index)   # 0
print(decision.grouping)       # case-1
```

All three types are frozen dataclasses. Construction validates: an `answer_id` that is
not one of the criterion's options raises
[`SchemaError`](../reference/api/errors.md) immediately.

Next: [run the whole pipeline](end-to-end.md).
