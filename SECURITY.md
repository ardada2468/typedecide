# Security policy

## Supported versions

`typedecide` is pre-1.0. Security fixes are made on the latest released minor version
only; there are no long-term-support branches yet.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |
| < 0.1   | No        |

When 0.2.0 is released, 0.1.x stops receiving fixes. This table is updated in the same
pull request as each minor release.

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through GitHub Security Advisories:
<https://github.com/ardada2468/typedecide/security/advisories/new>

Include what you can of: the affected version (`python -c "import typedecide;
print(typedecide.__version__)"`), your Python version and platform, a minimal input
that reproduces the problem, and what an attacker gains from it.

What to expect:

- an acknowledgement within 3 working days;
- an initial assessment (accepted, needs more information, or not a vulnerability, with
  the reasoning) within 10 working days;
- for an accepted report, a fix and a published advisory crediting you unless you ask
  not to be named. We ask for up to 90 days before public disclosure, and will tell you
  if we need less or more.

This is a small open-source project maintained without a security team or a bounty
programme. Those timings are a commitment to respond, not a contractual SLA.

## What is in scope

The Python package in `packages/typedecide` as published to PyPI, and the release and CI
workflows in `.github/workflows`. In particular:

- **Parsing untrusted datasets.** `load_decisions` reads CSV, JSON, JSONL, Parquet and
  Hugging Face datasets that a user may have received from someone else. Code execution,
  path traversal, or unbounded memory or CPU from a crafted file is in scope.
- **Parsing configuration.** `TrainConfig.from_file` reads YAML and JSON. YAML is loaded
  with `yaml.safe_load` only; a way to construct arbitrary Python objects from a config
  file is a vulnerability.
- **Secrets reaching artefacts.** `manifest.json` and `export_manifest.json` are written
  beside models and travel with them. A credential ending up in one is in scope
  (`sys.argv` is redacted for this reason).
- **Supply chain of the release**: the PyPI Trusted Publishing workflow, and anything
  that could let a pull request publish a package or read a secret.

## What is out of scope

- Vulnerabilities in optional dependencies (`torch`, `transformers`, `peft`, `datasets`,
  `optimum`, `onnx`, `onnxruntime`, `pyarrow`). Report those upstream. CI runs a blocking
  `pip-audit` over the **base install** (`numpy`, `pyyaml`); the extras are yours to pin
  and audit, because the right pins depend on your accelerator.
- A model that answers a decision wrongly, is biased, or can be steered by text inside
  the evidence (prompt injection). Those are correctness and evaluation concerns, and the
  library's evaluation tools exist to measure them, but they are not vulnerabilities in
  this code. Do not use a typed readout as a security boundary.
- The contents of `bench/`, `paper/`, `train/` and `web/`, which are research and demo
  code that is not published to PyPI. Reports are still welcome as ordinary issues.
- Denial of service that needs the attacker to already control the Python process.

## Security-relevant design decisions

These are properties the test suite enforces, not aspirations.

- **Your data does not leave your machine during training or evaluation.** The library
  makes no network calls of its own and has no telemetry. Decisions are read from local
  files, tokenised locally, and trained on locally. The only network traffic is what
  Hugging Face libraries do when *you* name a model or dataset id that is not already in
  your local cache: downloading those weights or that dataset. Set `HF_HUB_OFFLINE=1`
  and pass local paths to guarantee none at all. Training reports to no experiment
  tracker (`report_to=[]`).
- **`trust_remote_code` is off and cannot be turned on through this library.** No call
  to `from_pretrained` or `load_dataset` passes it, so it stays at the Hugging Face
  default of `False`, and there is no config key, argument, environment variable or CLI
  flag that changes that. A model or dataset that requires a custom loading script is
  refused by the underlying library and the refusal is reported. A test walks the AST of
  every source file and fails if `trust_remote_code` is ever passed.
- **No dynamic code execution.** The same test fails on `eval`, `exec`, `compile`,
  `pickle`, `marshal`, `subprocess`, `os.system`, `shell=True` and `yaml.load`.
- **Logs carry ids, never evidence.** Library log lines at every level name decisions by
  id and never include the state text. Error messages may quote a clipped fragment of
  the offending cell, or a row's option ids, so that the row can be found; they never
  reproduce a state. The library never calls `logging.basicConfig` and installs only a
  `NullHandler`.
- **Datasets are written atomically** (temporary file, then rename), so an interrupted
  write cannot leave a truncated dataset that still parses.
- **Model weights are loaded through `transformers`.** Prefer models distributed as
  `safetensors`; a legacy pickle-format checkpoint executes code when loaded, which is a
  property of that file format and not something this library can make safe.

## A note on CSV

The library reads CSV and never writes it, so it cannot introduce spreadsheet formula
injection. It does pass cell contents through unchanged: if you export decisions to a
spreadsheet yourself, neutralise cells beginning with `=`, `+`, `-` or `@` as you would
for any user-supplied text.
