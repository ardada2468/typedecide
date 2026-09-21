"""Telling someone their dataset is broken before they spend a night training on it.

The two checks that earn this module its place are the label balance and the
contradiction check. A criterion whose gold is 94% one class trains a model that has
learned the prior and nothing else, and it will still score 94%. Two rows with the
same evidence, the same question and different answers put a ceiling on accuracy that
no amount of training moves, and it is invisible in a metrics table.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..errors import DataError
from ..schema import Decision

LOGGER = logging.getLogger("typedecide.data.validate")

__all__ = [
    "DEFAULT_MAX_STATE_TOKENS",
    "DataReport",
    "Finding",
    "estimate_tokens",
    "validate",
]

#: Matches TrainConfig.max_length. Training truncates an over-long prompt from the LEFT
#: of the state (see `training.dataset.build_example`), so a state longer than this
#: loses its *start* -- the oldest evidence -- never the criterion, the options or
#: "Answer:". Evaluation does not truncate at all: it scores the prompt in full.
DEFAULT_MAX_STATE_TOKENS = 1024

#: Above this share of one class, a criterion is teaching the prior rather than the task.
IMBALANCE_THRESHOLD = 0.70

SEVERITIES = ("error", "warning", "note")
_MAX_LISTED = 8


@dataclass(frozen=True)
class Finding:
    """One thing worth saying about a dataset. `severity` is error, warning or note."""

    severity: str
    code: str
    message: str
    decisions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise DataError(
                f"Finding severity {self.severity!r} is not one of {', '.join(SEVERITIES)}."
            )

    def render(self) -> str:
        ids = ""
        if self.decisions:
            shown = ", ".join(self.decisions[:_MAX_LISTED])
            more = (
                f" (+{len(self.decisions) - _MAX_LISTED} more)"
                if len(self.decisions) > _MAX_LISTED
                else ""
            )
            ids = f"\n      rows: {shown}{more}"
        return f"  [{self.severity:<7}] {self.code}: {self.message}{ids}"


@dataclass(frozen=True)
class DataReport:
    """What `validate` found. `ok` is the one thing a caller must look at."""

    findings: tuple[Finding, ...]
    counts: dict[str, int] = field(default_factory=dict)
    label_balance: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)

    def by_severity(self, severity: str) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == severity)

    def render(self) -> str:
        lines = [
            "Dataset report: " + ("OK" if self.ok else "NOT USABLE AS GIVEN"),
            "  " + ", ".join(f"{k}={v}" for k, v in self.counts.items()),
        ]
        for severity in SEVERITIES:
            group = self.by_severity(severity)
            if group:
                lines.append(f"  -- {severity}s ({len(group)}) --")
                lines.extend(f.render() for f in group)
        if self.label_balance:
            lines.append("  -- label balance --")
            for key in sorted(self.label_balance):
                counts = self.label_balance[key]
                total = sum(counts.values()) or 1
                parts = [
                    f"{oid}={n} ({100 * n / total:.0f}%)" for oid, n in sorted(counts.items())
                ]
                lines.append(f"    {key}: " + ", ".join(parts))
        if self.ok and not self.findings:
            lines.append("  Nothing to report.")
        return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """A tokenizer-free estimate, so validation never downloads a model.

    Deliberately pessimistic: the larger of ~1.3 tokens per whitespace word and one
    token per four characters. It is a smoke alarm, not a scale.
    """
    words = len(text.split())
    return max(words + words // 3, (len(text) + 3) // 4)


def validate(
    decisions: Sequence[Decision],
    *,
    min_per_class: int = 5,
    max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS,
) -> DataReport:
    """Look for the problems that make a training run a waste of an afternoon."""
    if not decisions:
        return DataReport(
            findings=(
                Finding(
                    "error",
                    "empty",
                    "The dataset has no decisions in it. Fix: check the loader's source path "
                    "and any filter applied after loading.",
                ),
            ),
            counts={"decisions": 0, "labelled": 0, "unlabelled": 0},
            label_balance={},
        )

    findings: list[Finding] = []
    labelled = [d for d in decisions if d.labelled]
    unlabelled = [d for d in decisions if not d.labelled]

    findings.extend(_check_ids(decisions))
    findings.extend(_check_unlabelled(decisions, unlabelled))
    findings.extend(_check_option_sets(decisions))
    findings.extend(_check_contradictions(labelled))
    findings.extend(_check_duplicates(labelled))
    balance = _label_balance(decisions)
    findings.extend(_check_balance(balance))
    findings.extend(_check_rare_classes(balance, min_per_class))
    findings.extend(_check_state_length(decisions, max_state_tokens))
    findings.extend(_check_groups(decisions))

    order = {s: i for i, s in enumerate(SEVERITIES)}
    findings.sort(key=lambda f: (order[f.severity], f.code))

    counts = {
        "decisions": len(decisions),
        "labelled": len(labelled),
        "unlabelled": len(unlabelled),
        "criteria": len({d.criterion.key for d in decisions}),
        "groups": len({d.grouping for d in decisions}),
        "unique_states": len({d.state.strip() for d in decisions}),
        "errors": sum(1 for f in findings if f.severity == "error"),
        "warnings": sum(1 for f in findings if f.severity == "warning"),
    }
    LOGGER.info(
        "validated %d decisions: %d error(s), %d warning(s)",
        len(decisions),
        counts["errors"],
        counts["warnings"],
    )
    return DataReport(findings=tuple(findings), counts=counts, label_balance=balance)


# ---------------------------------------------------------------------------- checks


def _check_ids(decisions: Sequence[Decision]) -> list[Finding]:
    counts = Counter(d.id for d in decisions)
    repeated = sorted(i for i, n in counts.items() if n > 1)
    if not repeated:
        return []
    return [
        Finding(
            "error",
            "duplicate_id",
            f"{len(repeated)} decision id(s) appear more than once, so rows will overwrite each "
            "other in any id-keyed result and a group split cannot be trusted. Fix: make the id "
            "column unique, or drop it and let the loader hash the content.",
            tuple(repeated),
        )
    ]


def _check_unlabelled(
    decisions: Sequence[Decision], unlabelled: Sequence[Decision]
) -> list[Finding]:
    if not unlabelled:
        return []
    if len(unlabelled) == len(decisions):
        return [
            Finding(
                "error",
                "no_labels",
                f"None of the {len(decisions)} rows has an answer, so there is nothing to train "
                "or score against. Fix: fill the answer column, or pass "
                "FieldMapping(answer='<your column>') if the gold is under another name.",
            )
        ]
    return [
        Finding(
            "warning",
            "unlabelled",
            f"{len(unlabelled)} of {len(decisions)} rows have no answer. Training refuses "
            "them: `finetune` raises a TrainingError naming the rows rather than train on "
            "fewer than it was given. Evaluation skips them with a logged warning. Fix: "
            "label them, or drop them deliberately before training with "
            "`typedecide.training.labelled_only`, which also returns the ids it removed.",
            tuple(d.id for d in unlabelled),
        )
    ]


def _check_option_sets(decisions: Sequence[Decision]) -> list[Finding]:
    by_key: dict[str, dict[tuple[str, ...], list[str]]] = defaultdict(lambda: defaultdict(list))
    questions: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for decision in decisions:
        by_key[decision.criterion.key][tuple(sorted(decision.criterion.option_ids))].append(
            decision.id
        )
        questions[decision.criterion.key][decision.criterion.question.strip()].append(decision.id)

    findings: list[Finding] = []
    for key, variants in sorted(by_key.items()):
        if len(variants) > 1:
            shapes = sorted(variants, key=lambda ids: -len(variants[ids]))
            summary = "; ".join(
                f"{{{', '.join(shape)}}} on {len(variants[shape])} row(s)" for shape in shapes[:3]
            )
            offenders = [i for shape in shapes[1:] for i in variants[shape]]
            findings.append(
                Finding(
                    "error",
                    "option_set_mismatch",
                    f"Criterion {key!r} is asked with {len(variants)} different option sets: "
                    f"{summary}. A readout is defined over one closed set, so these are "
                    "different criteria wearing one name. Fix: give them distinct criterion "
                    "keys, or make the option set identical everywhere.",
                    tuple(sorted(offenders)),
                )
            )
    for key, texts in sorted(questions.items()):
        if len(texts) > 1:
            findings.append(
                Finding(
                    "warning",
                    "question_mismatch",
                    f"Criterion {key!r} is asked with {len(texts)} different question wordings, "
                    f"e.g. {sorted(texts)[0][:60]!r} and {sorted(texts)[1][:60]!r}. The prompt "
                    "changes between rows sharing a label space. Fix: settle on one wording, or "
                    "split the key.",
                    tuple(sorted(i for ids in list(texts.values())[1:] for i in ids)),
                )
            )
    return findings


def _check_contradictions(labelled: Sequence[Decision]) -> list[Finding]:
    """Same evidence, same question, different gold. The hard ceiling on accuracy."""
    by_state: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for decision in labelled:
        key = (decision.criterion.key, _normalise_state(decision.state))
        assert decision.answer_id is not None
        by_state[key][decision.answer_id].append(decision.id)

    findings: list[Finding] = []
    for (criterion_key, _), answers in sorted(by_state.items()):
        if len(answers) < 2:
            continue
        spread = ", ".join(
            f"{answer} on {len(ids)} row(s)" for answer, ids in sorted(answers.items())
        )
        findings.append(
            Finding(
                "error",
                "contradictory_labels",
                f"Criterion {criterion_key!r} has rows with identical evidence but different "
                f"answers: {spread}. No model can be right on both, so they set a hard ceiling "
                "on accuracy. Fix: reconcile the labels, or if the evidence really differs, "
                "include the part that distinguishes them in the state text.",
                tuple(sorted(i for ids in answers.values() for i in ids)),
            )
        )
    return findings


def _check_duplicates(labelled: Sequence[Decision]) -> list[Finding]:
    seen: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for decision in labelled:
        assert decision.answer_id is not None
        seen[
            (decision.criterion.key, _normalise_state(decision.state), decision.answer_id)
        ].append(decision.id)
    repeated = sorted(ids for ids in seen.values() if len(ids) > 1)
    if not repeated:
        return []
    total = sum(len(ids) - 1 for ids in repeated)
    return [
        Finding(
            "note",
            "duplicate_rows",
            f"{total} row(s) repeat evidence that is already present with the same answer. "
            "Duplicates inflate whichever split they land in and can leak across a split that "
            "is not grouped. Fix: de-duplicate, or give the copies a shared group_id.",
            tuple(i for ids in repeated for i in ids[1:]),
        )
    ]


def _label_balance(decisions: Sequence[Decision]) -> dict[str, dict[str, int]]:
    balance: dict[str, dict[str, int]] = {}
    for decision in decisions:
        counts = balance.setdefault(
            decision.criterion.key, {oid: 0 for oid in decision.criterion.option_ids}
        )
        for option_id in decision.criterion.option_ids:
            counts.setdefault(option_id, 0)
        if decision.answer_id is not None:
            counts[decision.answer_id] = counts.get(decision.answer_id, 0) + 1
    return balance


def _check_balance(balance: dict[str, dict[str, int]]) -> list[Finding]:
    findings: list[Finding] = []
    for key in sorted(balance):
        counts = balance[key]
        total = sum(counts.values())
        if total == 0:
            continue
        top, top_n = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        share = top_n / total
        if share > IMBALANCE_THRESHOLD and len(counts) > 1:
            findings.append(
                Finding(
                    "warning",
                    "imbalanced",
                    f"Criterion {key!r} answers {top!r} on {top_n} of {total} labelled rows "
                    f"({share:.0%}). Always guessing {top!r} would score {share:.0%}, so plain "
                    "accuracy will look fine while the model has learned the prior. Fix: "
                    "collect or down-sample towards balance, and read balanced_accuracy rather "
                    "than accuracy.",
                )
            )
    return findings


def _check_rare_classes(
    balance: dict[str, dict[str, int]], min_per_class: int
) -> list[Finding]:
    findings: list[Finding] = []
    for key in sorted(balance):
        counts = balance[key]
        if sum(counts.values()) == 0:
            continue
        unused = sorted(oid for oid, n in counts.items() if n == 0)
        rare = sorted(oid for oid, n in counts.items() if 0 < n < min_per_class)
        if unused:
            findings.append(
                Finding(
                    "warning",
                    "unused_option",
                    f"Criterion {key!r} offers {', '.join(unused)} but no row is labelled with "
                    "it. The model is asked to consider an answer it has never seen chosen, and "
                    "a split cannot cover it. Fix: label examples of it, or remove it from the "
                    "option set.",
                )
            )
        if rare:
            detail = ", ".join(f"{oid}={counts[oid]}" for oid in rare)
            findings.append(
                Finding(
                    "warning",
                    "rare_class",
                    f"Criterion {key!r} has class(es) below min_per_class={min_per_class}: "
                    f"{detail}. There are too few to both train on and hold out, so per-class "
                    "scores will be noise. Fix: collect more, merge the class, or lower "
                    "min_per_class deliberately.",
                )
            )
    return findings


def _check_state_length(decisions: Sequence[Decision], max_state_tokens: int) -> list[Finding]:
    long_rows = [(d.id, estimate_tokens(d.state)) for d in decisions]
    over = sorted((n, i) for i, n in long_rows if n > max_state_tokens)
    if not over:
        return []
    worst = over[-1]
    return [
        Finding(
            "warning",
            "long_state",
            f"{len(over)} row(s) have evidence estimated over the {max_state_tokens}-token "
            f"budget, the longest at ~{worst[0]} tokens. In training an over-long prompt is "
            "truncated from the left, so it is the START of the evidence that is lost -- the "
            "criterion, the options and 'Answer:' always survive. Evaluation does not "
            "truncate at all: it scores the full prompt, so the model is then scored on text "
            "it was never trained to read. Fix: raise TrainConfig.max_length, or summarise "
            "the evidence before loading.",
            tuple(i for _, i in reversed(over)),
        )
    ]


def _check_groups(decisions: Sequence[Decision]) -> list[Finding]:
    groups = {d.grouping for d in decisions}
    if len(groups) == 1 and len(decisions) > 1:
        return [
            Finding(
                "error",
                "single_group",
                f"All {len(decisions)} rows share the group_id {next(iter(groups))!r}, so a "
                "leakage-safe split has nowhere to cut and every row would land on one side. "
                "Fix: group by the thing that must not be split -- usually the state -- rather "
                "than by the dataset.",
            )
        ]
    return []


def _normalise_state(state: str) -> str:
    return " ".join(state.split()).casefold()
