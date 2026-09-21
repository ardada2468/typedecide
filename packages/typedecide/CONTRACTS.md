# Module contracts

Everything in `typedecide` speaks the types in `schema.py` and builds prompts only
through `prompt.py`. Nothing re-implements either. This file is the interface each
module must provide so the pieces fit without anyone reading anyone else's code.

## Already written — do not modify

- `src/typedecide/schema.py` — `Option`, `Criterion`, `Decision`, `LETTERS`,
  `decision_from_dict`, `decision_to_dict`, `fingerprint`, `stable_id`.
  `Criterion.reordered(order)` permutes options. `Decision.grouping` is the
  leakage-safe split key.
- `src/typedecide/errors.py` — `TypeDecideError` and its subclasses. Raise these,
  never bare `ValueError`.
- `src/typedecide/prompt.py` — `head_text`, `tail_text`, `full_prompt`,
  `state_prefix_ids`, `answer_slots`, `INSTRUCTION`, `LETTERS`.
  **This file mirrors `web/prompt.js` byte for byte. Do not change its output.**

## House rules

- Python ≥ 3.10, full type hints, `from __future__ import annotations`.
- Frozen dataclasses for config. No global mutable state.
- Heavy imports (`torch`, `transformers`, `peft`, `optimum`, `onnx`, `pyarrow`) go
  **inside the function that needs them**, so `import typedecide` stays fast and the
  base install stays light. Raise the matching error class with the extra to install
  if one is missing.
- `logging.getLogger("typedecide.<module>")`. No `print` outside `cli.py`.
- Every function that consumes randomness takes an explicit `seed: int`.
- Messages name the problem and the fix. Line length 100. Ruff and mypy strict clean.
- Tests go in `packages/typedecide/tests/test_<module>.py`, pure-stdlib where possible
  so CI runs them without a GPU or model download. Use a fake tokenizer rather than
  downloading one.

## `data/` — loading, validating, splitting

```python
# data/loaders.py
def load_decisions(source: str | Path, *, format: str | None = None,
                   mapping: FieldMapping | None = None) -> list[Decision]
def write_decisions(decisions: Sequence[Decision], path: Path) -> None

@dataclass(frozen=True)
class FieldMapping:
    """Maps a customer's column names onto our schema, so bring-your-own-data does
    not mean rename-your-columns-first."""
    state: str = "state"; question: str = "question"; options: str = "options"
    answer: str = "answer_id"; group: str | None = None; id: str | None = None
```

Support `.jsonl`, `.json`, `.csv`, `.parquet`, and a HF dataset id. CSV needs a way to
express options: accept a delimited list column plus a parallel ids column, and say so
clearly when it is malformed. Infer `format` from the suffix when not given.

```python
# data/validate.py
@dataclass(frozen=True)
class Finding:  # severity: "error" | "warning" | "note"
    severity: str; code: str; message: str; decisions: tuple[str, ...] = ()

@dataclass(frozen=True)
class DataReport:
    findings: tuple[Finding, ...]; counts: dict[str, int]
    label_balance: dict[str, dict[str, int]]     # criterion key -> option id -> n
    @property
    def ok(self) -> bool                          # no findings of severity "error"
    def render(self) -> str                       # human-readable, for the CLI

def validate(decisions: Sequence[Decision], *, min_per_class: int = 5) -> DataReport
```

Check at least: unlabelled rows; a criterion whose gold is >70% one class; duplicate
states across different answers (contradictory labels); identical decision ids; option
sets that differ between decisions sharing a criterion key; states longer than a
configured token budget; classes with fewer than `min_per_class` examples.

```python
# data/split.py
def group_split(decisions, *, eval_fraction=0.2, seed=0) -> tuple[list[Decision], list[Decision]]
```
Split on `Decision.grouping`, never on rows. Refuse and explain if a split would leave
a class entirely absent from eval.

```python
# data/augment.py
def randomise_option_order(decisions, *, seed: int) -> list[Decision]
def all_rotations(decision: Decision) -> list[Decision]
def all_permutations(decision: Decision) -> list[Decision]
```
Order randomisation is the single most important preprocessing step here: without it a
model learns "answer A". Preserve `answer_id` through the permutation.

## `training/` — LoRA on the answer token

```python
# training/config.py
@dataclass(frozen=True)
class TrainConfig:
    base_model: str; output_dir: Path
    epochs: float = 2.0; batch_size: int = 4; grad_accum: int = 4
    learning_rate: float = 1e-4; lora_rank: int = 16; lora_alpha: int | None = None
    lora_dropout: float = 0.05; max_length: int = 1024; seed: int = 0
    randomise_option_order: bool = True
    @classmethod
    def from_file(cls, path: Path) -> TrainConfig     # yaml or json
    def to_manifest(self) -> dict[str, Any]

# training/trainer.py
@dataclass(frozen=True)
class TrainResult:
    adapter_dir: Path; manifest: dict[str, Any]; train_loss: float; eval_loss: float | None

def finetune(train: Sequence[Decision], config: TrainConfig,
             *, eval_set: Sequence[Decision] | None = None,
             progress: Callable[[dict[str, Any]], None] | None = None) -> TrainResult
```

**Loss lands only on the answer-letter token.** Build `input_ids = encode(prompt) +
[answer_token]` and `labels = [-100] * len(prompt) + [answer_token]`, where the answer
token comes from `prompt.answer_slots`. Everything else is masked. Training the model
to write prose it will never be asked for spends capacity on the wrong thing.

The manifest records: base model, config, dataset `fingerprint()`, row counts, seed,
library versions, start/end time, and the resolved answer-slot ids. Write it as
`manifest.json` beside the adapter. A result you cannot trace to its input is not a
result.

## `evaluation/` — scoring the way a sceptic would

```python
# evaluation/metrics.py
def balanced_accuracy(gold: Sequence[str], pred: Sequence[str]) -> float
def mean_group_balanced_accuracy(gold, pred, groups: Sequence[str]) -> float
def expected_calibration_error(probs: Sequence[float], correct: Sequence[bool], bins=10) -> float
def brier_score(prob_vectors: Sequence[Sequence[float]], gold_indices: Sequence[int]) -> float
def order_consistency(picks_per_decision: Sequence[Sequence[str]]) -> float

# evaluation/runner.py
@dataclass(frozen=True)
class EvalResult:
    accuracy: float; balanced_accuracy: float; order_consistency: float | None
    ece: float | None; brier: float | None
    per_criterion: dict[str, float]; letter_distribution: dict[str, float]
    n: int; manifest: dict[str, Any]
    def render(self) -> str

def evaluate(decisions, model_id: str, *, adapter: Path | None = None,
             debias: str = "none",          # "none" | "cyclic" | "permutation" | "calibrated"
             batch_size: int = 8, device: str | None = None, seed: int = 0) -> EvalResult
```

`debias` implements permutation averaging: score each ordering, average the probability
per **option id**, and pick the argmax of that. A position prior contributes equally to
every option under a cyclic or full set and cancels. `"calibrated"` divides out the
distribution obtained with a content-free state, cached per (question, ordering).

Report `order_consistency` whenever more than one ordering was scored: plain accuracy
hides a model that answers by position, and this is the number that exposes it.

Batched scoring must **left-pad** and pass explicit
`position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)`. Right-padding with
default positions silently corrupts batched results.

## `export/` — shipping it to a browser

```python
# export/onnx.py
@dataclass(frozen=True)
class ExportConfig:
    base_model: str; adapter: Path | None; output_dir: Path
    opset: int = 17; prune_head_to: Sequence[int] | None = None; keep_merged: bool = False

def export_onnx(config: ExportConfig) -> Path

# export/prune.py
def answer_letter_token_ids(model_id: str, n_options: int) -> list[int]
def prune_lm_head(model, keep: Sequence[int])   # returns model with a K-row unembedding

# export/quantize.py
def quantize(model_path: Path, *, mode: str = "q4f16") -> Path
```

`prune_lm_head` is the speed lever and needs care. A typed readout only ever reads the
logits of the answer letters, so the unembedding matrix can be sliced to those K rows.
For Qwen3-0.6B the LM head is ~151936 × 1024 ≈ 155M parameters, roughly a quarter of
the model, and the per-call logits tensor drops from `[1, seq, 151936]` to
`[1, seq, K]`. Emit a `slot_map.json` recording which original token id each kept row
corresponds to, because after pruning the readout must index rows 0..K-1, not vocab
ids. State plainly in the docstring that a pruned model **cannot generate text** and
cannot report the share of total vocabulary mass on the answer letters.
