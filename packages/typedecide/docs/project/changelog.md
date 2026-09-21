# Changelog

The changelog is a file in the repository, not a page of this site, so that it is
versioned with the code and ships inside the source distribution:

**[`packages/typedecide/CHANGELOG.md`](https://github.com/ardada2468/typedecide/blob/main/packages/typedecide/CHANGELOG.md)**

Each published version also has a
[GitHub Release](https://github.com/ardada2468/typedecide/releases) whose body is that
version's changelog section.

## Format

The file follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

- Unreleased work accumulates under `## [Unreleased]`.
- Each release heading reads `## [X.Y.Z] - YYYY-MM-DD`.
- Changes are grouped as **Added**, **Changed**, **Deprecated**, **Removed**, **Fixed**
  and **Security**.

The heading format is load-bearing. The release workflow extracts the section whose
heading matches the tag, uses it as the GitHub Release body, and fails the release if
it cannot find a non-empty one. See [Releasing a new version](../runbooks/releasing.md).

## What must be recorded

Anything a user could notice:

- A new, changed or removed public function, class, argument, CLI flag or exit status.
- A change to a default value (for example a `TrainConfig` default).
- A change to a file format the library writes: `manifest.json`,
  `export_manifest.json`, `slot_map.json`, the `--json-out` payloads, the JSONL written
  by `write_decisions`.
- A new or renamed validation finding code, or a change in a finding's severity.
- **Any change to the prompt template.** It alters the behaviour of every model users
  have already fine-tuned, so it goes under **Changed** with the migration:
  retrain. See [Prompt parity](../concepts/prompt-parity.md).

## Which version am I running?

```bash
python -c "import typedecide; print(typedecide.__version__)"
```

Training manifests record it under `versions.typedecide`.
