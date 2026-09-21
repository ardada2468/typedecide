# Prompt parity check fails in CI

## Symptoms

The `checks` workflow fails in one of two places.

In *Library tests*, with a pytest failure in
`tests/test_prompt_parity.py::test_prompts_are_byte_identical` or
`::test_constants_match`. This is the check on the module `typedecide train` really
uses.

Or at the step *Prompt parity, web/prompt.js against train/prompt.py*:

```text
case 0 DIFFERS
  js: 'Apply the criterion to the evidence ...\nAnswer:'
  py: 'Apply the criterion to the evidence ...\nAnswer: '
prompt parity: FAILED
```

or the step cannot start:

```text
Could not run web/prompt.js through node:
```

## Diagnosis

The check renders the same (state, criterion) cases, one of them non-ASCII, through
the JavaScript prompt and a Python prompt, and compares the strings.

**1. Reproduce locally.** Both checks need Node.js 18 or newer.

```bash
python -m pytest packages/typedecide/tests/test_prompt_parity.py -q -rs
python train/check_parity.py
```

If pytest reports the tests as **skipped**, read the reason printed by `-rs`: `node`
is missing, or `web/prompt.js` was not found. A skip is not a pass.

**2. Find the first differing character.** The `repr` output shows escapes. Typical
differences:

| Difference | Usual cause |
|---|---|
| Trailing space or newline after `Answer:` | An editor or formatter "tidied" a template string |
| `\n\n` against `\n` between sections | A joined list gained or lost an empty element |
| `A.` against `A)` or `A:` | The option rendering was changed in one file |
| Different instruction wording | `INSTRUCTION` edited in one file |
| Differs only on the non-ASCII case | One side normalises or escapes Unicode |

**3. Find which side moved.**

```bash
git log --oneline -5 -- web/prompt.js train/prompt.py packages/typedecide/src/typedecide/prompt.py
```

**4. Know which file each check covers.** The pytest test compares `web/prompt.js`
with `packages/typedecide/src/typedecide/prompt.py`, the module `typedecide train`
uses. The separate CI step compares `web/prompt.js` with the prototype
`train/prompt.py`. All three files must agree; see
[Prompt parity](../concepts/prompt-parity.md#how-parity-is-checked).

**5. If node could not run the script**, check that `node --version` is 18 or newer
and that `web/prompt.js` still parses: `node --check web/prompt.js`.

## Fix

Decide which prompt is the intended one. Then make every implementation match it.

- **The change was accidental** (a formatter, a stray edit). Revert it. Nothing else
  is needed.

- **The change was intended.** Apply it to `web/prompt.js`,
  `packages/typedecide/src/typedecide/prompt.py` and `train/prompt.py` in one commit.
  Then, because the template is part of the model:

    1. Retrain every adapter that will be served with the new prompt. An adapter tuned
       on the old text is now mismatched.
    2. Re-run evaluation for the base model as well. The base numbers change too.
    3. Re-check [answer-slot discovery](../concepts/answer-slots.md) on your
       tokenizer. A change near `Answer:` can change the boundary tokenization.
    4. Record the change in `CHANGELOG.md` under **Changed**. It alters the behaviour
       of every model users have already trained.

!!! danger "Never make the check pass by editing the test cases"
    If the two prompts differ, a model fine-tuned in Python is being served a prompt it
    was not trained on. Nothing crashes and offline evaluation still looks fine. The
    check is the only thing standing between you and that.

## Verify

```bash
python -m pytest packages/typedecide/tests/test_prompt_parity.py -q -rs   # passed, not skipped
python train/check_parity.py                                              # prints: prompt parity: OK
node --check web/prompt.js
```

- All three commands exit 0, and pytest reports the parity tests as **passed**, with
  none skipped.
- CI is green on the pull request.
- If the template changed: `typedecide train` passes preflight on your tokenizer, and
  the new `manifest.json` lists `answer_slots` for every criterion.
