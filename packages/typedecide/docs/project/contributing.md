# Contributing

Issues and pull requests are welcome at
[github.com/ardada2468/typedecide](https://github.com/ardada2468/typedecide). If the
repository has a
[`CONTRIBUTING.md`](https://github.com/ardada2468/typedecide/blob/main/CONTRIBUTING.md),
it takes precedence over this page. The module interfaces are specified in
[`CONTRACTS.md`](https://github.com/ardada2468/typedecide/blob/main/packages/typedecide/CONTRACTS.md).

## Set up

```bash
git clone https://github.com/ardada2468/typedecide
cd typedecide
python -m venv .venv && . .venv/bin/activate
pip install -e "packages/typedecide[dev]"
```

The `dev` extra installs pytest, pytest-cov, mypy and ruff. You do not need torch to
work on most of the library or to run its tests.

## The checks CI runs

```bash
python -m pytest packages/typedecide -q
python -m ruff check packages/typedecide/src packages/typedecide/tests
python -m mypy --strict packages/typedecide/src/typedecide
python train/check_parity.py          # needs Node.js
```

A pull request must pass all of them. The tests download no model weights: they use
fake tokenizers and stub models, so they run in seconds without a GPU.

## House rules

These come from `CONTRACTS.md` and are enforced in review:

- **Python 3.10 or newer, full type hints, `from __future__ import annotations`.**
  `mypy --strict` must stay clean.
- **Frozen dataclasses for configuration. No global mutable state.**
- **Heavy imports go inside the function that needs them.** `torch`, `transformers`,
  `peft`, `optimum`, `onnx` and `pyarrow` are never imported at module level, so
  `import typedecide` stays fast and the base install stays light. When a dependency
  is missing, raise the matching error class and name the extra to install.
- **Raise the library's own exceptions**, never a bare `ValueError`. Messages name the
  problem and the fix.
- **`logging.getLogger("typedecide.<module>")`. No `print` outside `cli.py`.**
- **Every function that consumes randomness takes an explicit `seed: int`.**
- **Everything speaks the types in `schema.py` and builds prompts only through
  `prompt.py`.** Nothing re-implements either.
- **Line length 100.** Ruff rules: `E`, `F`, `I`, `N`, `UP`, `B`, `SIM`, `RET`, `ARG`,
  `PTH`.
- **Tests go in `packages/typedecide/tests/test_<module>.py`**, pure standard library
  where possible. Use a fake tokenizer; do not download one.

!!! danger "Do not change the prompt's output casually"
    `prompt.py` mirrors `web/prompt.js` byte for byte. A change to either must be made
    to both in the same commit, and it invalidates every adapter users have trained.
    Read [Prompt parity](../concepts/prompt-parity.md) first.

## Honesty rules for numbers

This project's documentation labels every performance claim MEASURED, DERIVED or
PROJECTED, and quotes accuracy only from committed result files.

- A new benchmark number needs a committed result file under `bench/results/` that it
  can be traced to.
- Do not move a DERIVED label to MEASURED without the measurement.
- Report `order_consistency` beside any accuracy you add.

## Working on this documentation

```bash
cd packages/typedecide
pip install -r docs/requirements.txt
mkdocs serve                 # live preview at http://127.0.0.1:8000
mkdocs build --strict        # what CI runs; any warning fails the build
```

The API reference is generated from docstrings by static analysis, so improving a
docstring improves the site, and the build needs neither torch nor the package
installed. Every code sample in the guides is expected to run against the real API.
If you change a signature, search `docs/` for it.

The site is built and deployed by
[`.github/workflows/docs.yml`](https://github.com/ardada2468/typedecide/blob/main/.github/workflows/docs.yml)
on every push to `main` that touches the package or `web/`. Pull requests build the
site without deploying it.

## Pull requests

- Keep each change focused. Add a test that fails without it.
- Add a line to `CHANGELOG.md` under `[Unreleased]` for anything a user could notice.
  [What must be recorded](changelog.md#what-must-be-recorded) has the list.
- Do not commit model weights, datasets containing customer text, or credentials.
