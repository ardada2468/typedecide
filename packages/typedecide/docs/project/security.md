# Security

## Reporting a vulnerability

Please report security problems privately, not in a public issue.

1. Use GitHub's private vulnerability reporting for the repository:
   [Report a vulnerability](https://github.com/ardada2468/typedecide/security/advisories/new).
2. If the repository has a `SECURITY.md`, it takes precedence over this page:
   [`SECURITY.md`](https://github.com/ardada2468/typedecide/security/policy).

Include the version (`python -c "import typedecide; print(typedecide.__version__)"`),
what you did, what happened, and what you expected. Remove any customer data from
reproduction files.

Only the latest released version receives fixes while the project is pre-1.0.

## What the library does with your data

- **No telemetry.** The library makes no network calls of its own.
- **Network access happens through the libraries you ask it to use**: `transformers`
  and `datasets` download models, tokenizers and datasets from the Hugging Face Hub
  when you pass an id and not a local path. Use local paths and those libraries'
  offline settings in an air-gapped environment.
- **Decision text stays local.** States are tokenized and scored in your process.
- **The browser demo has no backend.** Evidence typed into the page is scored in the
  visitor's tab. The page fetches static files, transformers.js from a CDN, web fonts,
  and the model files.

## Things to be careful with

**Model and adapter files are code-adjacent.** Loading a model, tokenizer or adapter
from the Hugging Face Hub or from disk runs through `transformers` and `peft`. Load
only checkpoints you trust, and prefer the `safetensors` format. `typedecide` does not
pass `trust_remote_code=True` anywhere.

**Manifests record your environment.** `manifest.json` stores the command line, the
platform string, library versions and the `output_dir` path. Arguments that look like
credentials are replaced with `[redacted]` before writing (the value after a flag named
like `--hf-token`, and any argument shaped like a Hugging Face, OpenAI-style or GitHub
token). That is pattern matching, not a guarantee, so review a manifest before
attaching it to a public issue, and pass secrets through environment variables, not
flags.

**Validation reports quote your data.** Loader errors quote the offending cell, and
findings list row ids. Treat `--json-out` reports and CI logs as containing data.

**Config files are parsed safely.** YAML is loaded with `yaml.safe_load`, and unknown
keys are rejected.

**Credentials.** The release workflow uses PyPI Trusted Publishing (OIDC). There is no
API token in the repository or its settings, and there should never be one. The CI
workflow greps the tree for strings that look like committed API keys.

## Supply chain

Release artefacts are built by GitHub Actions from a tagged commit, checked with
`twine check --strict`, smoke-tested from the built wheel, and uploaded by
`pypa/gh-action-pypi-publish`, which also publishes
[PEP 740](https://peps.python.org/pep-0740/) attestations tying each file to the
repository and workflow.
