# Prompt parity

## The contract

The browser builds its prompt with `web/prompt.js`. The library builds its prompt with
`typedecide/prompt.py`. The two must produce **byte-identical** text for the same state
and criterion:

| Python (`typedecide.prompt`) | JavaScript (`web/prompt.js`) |
|---|---|
| `INSTRUCTION` | `INSTRUCTION` |
| `LETTERS` (`"ABCDEFGHIJKLMNOPQRST"`) | `LETTERS` |
| `head_text(state)` | `headText(state)` |
| `tail_text(criterion)` | `tailText(criterion)` |
| `full_prompt(state, criterion)` | `fullPrompt(state, criterion)` |
| `answer_slots(tokenizer, prompt, count)` | `answerSlots(tok, prompt, count)` in `web/engine.js` |

## Why this is the most expensive mistake available

If the two drift, the fine-tune optimises a prompt the deployment never sends.
Nothing crashes. Offline evaluation, which uses the Python prompt, still looks fine.
Accuracy in the browser degrades quietly, and the symptom is indistinguishable from
bad training data. A single changed character is enough, because it can change the
tokenization at the `Answer:` boundary and with it the answer slots.

## What follows for you

- **Do not edit the prompt in one place.** A change to the instruction, the headers,
  the option rendering or the whitespace must be made in both files in the same
  commit, and a model fine-tuned on the old prompt must be retrained.
- **Put task-specific wording in the `question` and option `description`s.** Those are
  data, not template, and they flow through both implementations identically.
- **Treat the template as part of the model.** The training manifest records the
  resolved answer-slot ids, which change if the template changes the boundary.

## How parity is checked

Both checks execute the real `web/prompt.js` through Node.js and compare its output
with Python's, string for string, on cases that include non-ASCII text, odd
whitespace, quotes, template-literal syntax and a 20-option criterion.

| Check | Guards | Runs as |
|---|---|---|
| `packages/typedecide/tests/test_prompt_parity.py` | The library's own `typedecide/prompt.py`, plus `LETTERS` and `INSTRUCTION` | Part of the test suite: `python -m pytest packages/typedecide` |
| `train/check_parity.py` | The prototype `train/prompt.py` | Its own CI step |

```bash
python -m pytest packages/typedecide/tests/test_prompt_parity.py -q -rs
python train/check_parity.py
```

!!! warning "A skipped parity test is not a passing one"
    `test_prompt_parity.py` **skips** when `node` is not installed, and when `web/` is
    not present, which is the case when the tests run from a source distribution. The
    `-rs` flag prints skip reasons. CI installs Node.js, so the test runs there. On a
    machine without Node.js you have not checked parity.

The [parity runbook](../runbooks/prompt-parity-ci.md) covers a failing check.
