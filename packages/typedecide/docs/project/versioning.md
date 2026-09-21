# Versioning and deprecation policy

## Semantic Versioning

`typedecide` follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
Version strings are [PEP 440](https://peps.python.org/pep-0440/): `X.Y.Z` for a
release and `X.Y.ZrcN` for a release candidate.

| Bump | When |
|---|---|
| Patch (`0.1.0` to `0.1.1`) | Bug fixes. No public API change |
| Minor (`0.1.0` to `0.2.0`) | Backwards-compatible additions |
| Major (`1.0.0` to `2.0.0`) | Breaking changes |

!!! warning "The pre-1.0 caveat"
    The current version is `0.1.0` and the package is classified as alpha. **While the
    version is `0.y.z`, the public API may change in a minor release.** A breaking
    change bumps the minor, not the major. Every such change is listed in the
    changelog under **Changed** or **Removed**, with the migration.

    In practice: pin a minor series in anything you depend on.

    ```text
    typedecide>=0.1,<0.2
    ```

The version lives in one place, `__version__` in `src/typedecide/__init__.py`, and
the package metadata reads it from there.

## What counts as public API

Public, and covered by the policy:

- Every name in the `__all__` of `typedecide`, `typedecide.data`,
  `typedecide.training`, `typedecide.evaluation` and `typedecide.export`. The
  [API index](../reference/api/index.md) lists them.
- The signatures, defaults and documented behaviour of those names.
- The exception hierarchy in `typedecide.errors`, including which class a documented
  failure raises.
- The CLI: subcommands, flags, defaults and [exit statuses](../reference/cli.md#exit-status).
- The **keys** of the files the library writes (`manifest.json`,
  `export_manifest.json`, `slot_map.json`, the `--json-out` payloads) and the JSONL
  decision format. `manifest.json` and `slot_map.json` carry their own integer
  `manifest_version` / `version` field. `SlotMap.from_dict` refuses a version it does
  not understand.
- Validation finding **codes** and their severities.
- **The prompt template.** A change to the rendered prompt changes the behaviour of
  every fine-tuned model, so it is treated as a breaking change.

Not public, and free to change in any release:

- Any name beginning with an underscore, and any module-level name not in `__all__`.
- The wording of error messages, log messages and validation messages. Match on
  exception classes and finding codes, never on text.
- The exact formatting of `DataReport.render()` and `EvalResult.render()`.
- Minimum versions of optional dependencies, which may rise in a minor release.
- Numerical results, within what upstream libraries (torch, transformers, ONNX Runtime)
  guarantee.

## Deprecation

When a public name or behaviour is to be removed:

1. It is marked deprecated in the docstring and listed under **Deprecated** in the
   changelog, with the replacement.
2. Where practical it keeps working for at least one further minor release.
3. It is then removed in a release that lists it under **Removed**.

A current example: `ScorerError` inherits from both `EvaluationError` and
`ConfigError`. The `ConfigError` base exists only so that `except ConfigError` written
against 0.1.0 keeps working. It is deprecated and will be dropped in a future major
version. New code should catch `EvaluationError`.

## Supported Python versions

Python 3.10 and newer (`requires-python = ">=3.10"`). The package metadata lists 3.10
to 3.13. Dropping a Python version is a minor-version change while pre-1.0, and is
recorded in the changelog.

## Releases

Releases are published to PyPI from GitHub Actions by Trusted Publishing, and release
candidates go to TestPyPI. A bad release is yanked, never deleted, so exact pins keep
working. See [Releasing a new version](../runbooks/releasing.md).
