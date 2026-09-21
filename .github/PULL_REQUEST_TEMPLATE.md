## What this changes

<!-- One or two sentences. What is different for a user after this merges? -->

## Why

<!-- The problem, with a link to the issue if there is one: Fixes #123 -->

## How it was tested

<!-- The commands you ran and what they showed. For a bug fix, name the regression test. -->

## Checklist

- [ ] `python -m pytest` passes (from `packages/typedecide`)
- [ ] `python -m ruff check src tests` is clean
- [ ] `python -m mypy --strict src/typedecide` is clean (on Python 3.10)
- [ ] New behaviour has tests; a bug fix has a regression test whose docstring says what used to go wrong
- [ ] Coverage did not drop, or the description explains why
- [ ] No `# type: ignore` or `# noqa` without a specific code and a reason
- [ ] `packages/typedecide/CHANGELOG.md` is updated under "Unreleased" for anything user-visible
- [ ] Docs and docstrings match the new behaviour

### Project invariants

- [ ] The text produced by `prompt.py` is unchanged — or `web/prompt.js`, `train/prompt.py` and `typedecide/prompt.py` change together, both parity checks pass, and the description explains the break
- [ ] Nothing imports torch, transformers, peft, datasets, optimum, onnx or pyarrow at module level
- [ ] Public API is additive: nothing in an `__all__`, no CLI flag and no dataclass field was renamed, removed or re-typed
- [ ] New failures raise a `TypeDecideError` subclass with a message that names the problem and the fix
- [ ] Anything random takes an explicit `seed`; anything feeding a fingerprint, split or manifest iterates in a defined order
- [ ] No `trust_remote_code`, `eval`, `pickle`, `subprocess` or unsafe YAML loading; no state text in log lines
- [ ] No credentials, model weights or real customer data in the diff
