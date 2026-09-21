# Contributing

Thanks for looking. This repository holds a Python library (`packages/typedecide`, the
part published to PyPI) and the research code around it (`bench/`, `train/`, `web/`,
`paper/`). Most contributions are to the library, and this guide is about that.

By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Security problems
go through [SECURITY.md](SECURITY.md), not the issue tracker.

## Development setup

You need Python 3.10 or newer, [uv](https://docs.astral.sh/uv/), and Node 22 (only for
the prompt-parity test). You do **not** need a GPU, torch, or any model weights: the
whole suite runs against stand-ins.

```bash
git clone https://github.com/ardada2468/typedecide
cd typedecide/packages/typedecide

uv venv --python 3.10 .venv          # 3.10 is the floor the library promises
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# optional, so the Parquet and ONNX tests run instead of skipping
uv pip install -e ".[parquet]" onnx
```

Plain `python -m venv` and `pip install -e ".[dev]"` work just as well.

## The checks CI runs

Run all three before you push. CI runs them on Python 3.10 to 3.13 on Linux, and on
macOS and Windows.

```bash
python -m pytest                                  # tests
python -m ruff check src tests                    # lint
python -m mypy --strict src/typedecide            # types; run this on Python 3.10
```

With coverage, as CI measures it:

```bash
python -m pytest --cov=typedecide --cov-branch --cov-report=term-missing
```

CI fails under 95% (line and branch) with the `dev` extra alone, and under 97% with
`pyarrow` and `onnx` installed. New code comes with tests; a pull request that lowers
coverage needs to say why.

Two notes that save time:

- **mypy targets Python 3.10** (`python_version = "3.10"` in `pyproject.toml`). On Python
  3.12+ pip installs a numpy whose type stubs use 3.12-only syntax, and mypy stops with
  `errors prevented further checking`. Use a 3.10 environment for the type check.
- **No blanket suppressions.** A `# type: ignore` or `# noqa` needs a specific code and
  a reason on the same line, for example
  `# type: ignore[misc]  # transformers is untyped, so its base class is Any`.

## The rules that are not negotiable

### 1. The prompt is frozen, and mirrored

`packages/typedecide/src/typedecide/prompt.py`, `train/prompt.py` and `web/prompt.js`
must produce **byte-identical** text. A fine-tune learns the exact prompt it is trained
on; if the browser sends a prompt that differs by one newline, accuracy degrades
silently and looks like bad data.

- Do not change the output of `prompt.py`. If a change is genuinely needed, change all
  three files in one pull request, explain in the description why every existing
  adapter should be considered invalid, and bump the minor version.
- `tests/test_prompt_parity.py` runs the real JavaScript through node and compares it
  with `typedecide.prompt`. `train/check_parity.py` does the same for `train/prompt.py`.
  CI fails if either is skipped.

```bash
python -m pytest tests/test_prompt_parity.py -rs   # from packages/typedecide
python train/check_parity.py                       # from the repository root
```

### 2. `import typedecide` stays light

The base install is `numpy` and `pyyaml`. `torch`, `transformers`, `peft`, `datasets`,
`optimum`, `onnx`, `onnxruntime` and `pyarrow` are optional extras and are imported
**inside the function that needs them**, never at module level. If one is missing, raise
the matching `TypeDecideError` subclass naming the extra to install. A test imports every
subpackage in a fresh interpreter and fails if a heavy module appears in `sys.modules`.

### 3. Errors are diagnoses

Raise a subclass of `TypeDecideError` from `errors.py`, never a bare `ValueError`. The
message names the row, the column and the offending value (clipped), and says what would
fix it. The CLI prints `error: ...` and exits 2 for these, with no traceback, so an
unhelpful message is all the user gets.

### 4. Security properties have tests, and the tests stay

- `trust_remote_code` is never passed to any Hugging Face call.
- YAML is read with `yaml.safe_load` only.
- No `eval`, `exec`, `pickle`, `subprocess` or `shell=True` in library code.
- Library code never calls `print` or `logging.basicConfig` (`cli.py` excepted) and never
  logs a decision's state text at any level; log ids.

`tests/test_package.py` enforces these by walking the AST of every source file.

### 5. Determinism

Every function that consumes randomness takes an explicit `seed`. Anything that feeds a
fingerprint, a split or a manifest iterates in a sorted, defined order. No global mutable
state; config objects are frozen dataclasses.

### 6. The public API is additive until 1.0

Anything listed in an `__all__`, any CLI subcommand or flag, and any dataclass field is
public. Add; do not rename, remove or change a signature without a deprecation cycle and
a CHANGELOG entry. `tests/test_cli.py` parses every documented flag as a tripwire.

The module contracts are written down in
[`packages/typedecide/CONTRACTS.md`](packages/typedecide/CONTRACTS.md). Read the section
for the module you are touching.

## How to add a loader

Loaders live in `src/typedecide/data/loaders.py` and all end in the same place: a row
mapping handed to `_decision_from_mapping`, which applies the `FieldMapping`, resolves
the answer, and builds the `Decision`.

1. Add the format name to `FORMATS`, and its file suffixes to `SUFFIX_FORMATS`.
2. Write `_load_<format>(path: Path, mapping: FieldMapping) -> list[Decision]` and
   register it in the `loaders` table in `load_decisions`.
3. **Stream.** Iterate rows or record batches; do not read the whole file into memory
   when the format allows otherwise (`_load_jsonl` and `_load_parquet` are the models).
4. Import any optional dependency inside the function and raise `DataError` naming the
   extra if it is missing. Add the extra to `pyproject.toml`.
5. Create one `_IndexedAnswers()` per source, pass it to every `_decision_from_mapping`
   call along with `position=len(decisions)`, and call `.settle(decisions, label)` at
   the end. That is what decides, once per file, whether numeric answers are 0-based or
   1-based, instead of guessing row by row.
6. Wrap the format's own parse errors in `DataError` with the file name and row number.
   `UnicodeDecodeError` and `OSError` are already wrapped by `load_decisions`.
7. Raise `DataError` if the source has no rows.
8. Add the format to the CLI's `--format` choices in `cli.py`.
9. Test it in `tests/test_data_loaders.py`: a round trip, a malformed file, an empty
   file, a field mapping, and the missing-dependency message (set
   `sys.modules["<dep>"] = None` with `monkeypatch`). Use `pytest.importorskip` for tests
   that need the real dependency.

## Commits and pull requests

- Branch from `main`. Keep a pull request to one logical change; a bug fix and a refactor
  are two pull requests.
- Write commit subjects in the imperative, under about 72 characters, saying what changed
  and where: `loaders: settle numeric answer columns once per file`. Use the body for the
  why. Conventional-commit prefixes are welcome but not required.
- **A bug fix comes with a regression test** that fails before the fix, and whose
  docstring says what used to go wrong. That docstring is the documentation of the bug.
- Update `packages/typedecide/CHANGELOG.md` under "Unreleased" for anything a user could
  notice.
- Fill in the pull request template. CI must be green, and a maintainer reviews every
  pull request before it merges. Expect a squash or rebase merge.
- Never commit credentials, model weights, or customer data. Test fixtures are synthetic.
  CI greps the tree for API-key patterns.

## Reporting bugs and asking for features

Use the issue templates. For a bug, the most useful thing you can attach is the smallest
dataset that reproduces it, with anything sensitive replaced, and the full `error:` line
or traceback.

## Licence

Contributions are accepted under the repository's [MIT licence](LICENSE). By opening a
pull request you confirm you have the right to contribute the code under it.
