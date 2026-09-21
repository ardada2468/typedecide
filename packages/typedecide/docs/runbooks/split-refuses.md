# Split refuses, class missing from eval

## Symptoms

`typedecide split` (or `group_split`) exits with status 2 and:

```text
error: A split at eval_fraction=0.2 with seed=0 would leave 2 class(es) with zero examples in eval, so the score would say nothing about them:
  queue/legal: 3 row(s) across 3 group(s)
  severity/critical: 4 row(s) across 1 group(s) -- one group, so no group split can ever put it on both sides
Fix: raise eval_fraction, try another seed, collect more examples of those classes, or -- if a class genuinely lives in one group -- merge or drop it before splitting.
```

Each line is `<criterion key>/<answer id>`, its labelled row count, and how many
groups those rows span.

Related refusals from the same function:

- `All N rows share the group '...'`: there is only one group.
- `eval_fraction=... must be strictly between 0 and 1`.
- `Cannot split an empty dataset`.

## Diagnosis

The refusal is deliberate: a class with no eval examples is a class the score says
nothing about, and `balanced_accuracy` would silently average over fewer classes.

Read the **group count** on each line. It decides the fix.

| Line says | Meaning |
|---|---|
| `across 1 group(s) -- one group...` | Structural. No seed or fraction can help |
| A few rows across a few groups | The class is rare. A different seed may work; more data is the real answer |
| Many rows across many groups, and the dataset is small | `eval_fraction` is too small for this many classes |

`validate` gives the same information earlier, as `rare_class` and `unused_option`
warnings:

```bash
typedecide validate tickets.csv --min-per-class 10
```

## Fix

In order of preference:

1. **Collect more examples of the rare classes.** A class with three examples cannot be
   both trained on and measured, whatever the split does.

2. **Merge or drop a class that cannot be supported.** If `legal` has three rows, fold
   it into `other` at the source, or remove those rows and the option. Remember that
   every row sharing a criterion key must list the same option set.

3. **Fix the grouping if it is too coarse.** A class living in one group often means
   `group_id` is a customer or batch, not a ticket. Group by the thing that must not
   leak, which is usually the state.

4. **Raise `eval_fraction`.**

    ```bash
    typedecide split tickets.csv --out data/ --eval-fraction 0.3
    ```

5. **Try other seeds**, as a last resort, and record the one you used:

    ```python
    from typedecide import group_split, load_decisions
    from typedecide.errors import DataError

    decisions = load_decisions("tickets.csv")
    for seed in range(50):
        try:
            train, held = group_split(decisions, eval_fraction=0.2, seed=seed)
        except DataError:
            continue
        print("seed", seed, len(train), len(held))
        break
    else:
        raise SystemExit("no seed in 0..49 covers every class; the data needs fixing")
    ```

!!! warning "Seed-hunting selects a split"
    Picking the first seed that passes is mild. Picking the seed that gives the best
    score is not. Choose the seed before you look at any model output, and keep it.

!!! danger "Do not work around the refusal with a row-level split"
    Splitting rows yourself puts the same state on both sides whenever rows share
    evidence, and the resulting eval score measures memorisation.

## Verify

```bash
typedecide split tickets.csv --out data/ --eval-fraction 0.3 --seed 0
typedecide validate data/eval.jsonl
```

- The split prints row counts for both files and exits `0`.
- In the eval report's `label balance`, every class you care about has a non-zero
  count.
- No group appears on both sides:

    ```python
    from typedecide import load_decisions

    train = {d.grouping for d in load_decisions("data/train.jsonl")}
    held = {d.grouping for d in load_decisions("data/eval.jsonl")}
    assert not train & held
    ```
