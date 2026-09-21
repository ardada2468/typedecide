# Group-aware splitting

## Split on groups, never on rows

Every criterion asked about one ticket shares that ticket's text. A row-level random
split puts "which queue?" for ticket 17 in train and "how urgent?" for ticket 17 in
eval. The model has then read the eval evidence during training, and the eval score
becomes partly a memorisation score that production will not reproduce.

[`group_split`](../reference/api/data/split.md) splits on `Decision.grouping`, which
is `group_id` when set and the decision's own `id` otherwise.

```python
from typedecide import group_split, load_decisions

decisions = load_decisions("tickets.jsonl")
train, held = group_split(decisions, eval_fraction=0.2, seed=0)

assert not {d.grouping for d in train} & {d.grouping for d in held}
```

!!! warning "The default is only safe if your rows do not share states"
    With `group_id` unset, each row is its own group, and the split is an ordinary
    row split. That is correct when every row has distinct evidence. If several rows
    share a state (several criteria per ticket, or re-labelled duplicates), set
    `group_id` to the thing that must not leak: usually the ticket, case, document or
    customer id. For CSV input, point at the column with `--col-group` or
    `FieldMapping(group=...)`.

## How the split is made

1. Group keys are **sorted**, then shuffled with `random.Random(seed)`. Sorting first
   makes the result independent of load order.
2. Groups are taken into eval, in shuffled order, until eval holds at least
   `eval_fraction` of the **rows**. At least one group always stays in train and at
   least one goes to eval.
3. Rows keep their input order within each side.

Because whole groups move, the realised eval share is approximate, and it can
overshoot when groups are large.

## The refusal

Before returning, the split checks that every `(criterion key, answer id)` pair that
occurs in the labelled data also occurs at least once in eval. If any pair would be
missing, it raises [`DataError`](../reference/api/errors.md) and lists them:

```text
A split at eval_fraction=0.2 with seed=0 would leave 1 class(es) with zero examples in eval, so the score would say nothing about them:
  queue/billing: 2 row(s) across 2 group(s)
Fix: raise eval_fraction, try another seed, collect more examples of those classes, or -- if a class genuinely lives in one group -- merge or drop it before splitting.
```

A class that lives entirely in one group is flagged in the message, because no group
split can ever put it on both sides.

The reasoning: a class absent from eval is a class the score says nothing about, and
`balanced_accuracy` silently averages over only the classes present in the gold. A
quietly narrower metric is worse than an error.

`group_split` also raises `DataError` for an empty dataset, for an `eval_fraction` not
strictly between 0 and 1, and when every row shares one group.

!!! note "What the check does not guarantee"
    It guarantees presence in **eval**. It does not guarantee every class is present
    in train, and it does not stratify. `validate` reports `rare_class` for classes
    with fewer than `min_per_class` examples, which is the earlier warning for the
    same underlying problem.

The step-by-step fix is in the
[split-refuses runbook](../runbooks/split-refuses.md).
