"""Validation tests. The point is that the report catches things, not that it runs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from typedecide.data.validate import (  # noqa: E402
    DataReport,
    Finding,
    estimate_tokens,
    validate,
)
from typedecide.errors import DataError  # noqa: E402
from typedecide.schema import Criterion, Decision, Option  # noqa: E402

URGENCY = Criterion(
    key="urgency",
    question="How urgent is this ticket?",
    options=(Option("low", "Can wait a week"), Option("high", "Needs attention today")),
)


def make(
    ident: str,
    state: str,
    answer: str | None = "low",
    *,
    criterion: Criterion = URGENCY,
    group: str | None = None,
) -> Decision:
    return Decision(
        id=ident, state=state, criterion=criterion, answer_id=answer, group_id=group
    )


def balanced(n: int = 12) -> list[Decision]:
    return [
        make(f"t{i}", f"Ticket {i} describes a distinct problem.", "low" if i % 2 else "high")
        for i in range(n)
    ]


def codes(report: DataReport, severity: str | None = None) -> set[str]:
    return {
        f.code for f in report.findings if severity is None or f.severity == severity
    }


# ------------------------------------------------------------------------ happy path


def test_clean_dataset_is_ok() -> None:
    report = validate(balanced())
    assert report.ok
    assert codes(report, "error") == set()
    assert report.counts["decisions"] == 12
    assert report.counts["labelled"] == 12
    assert report.counts["groups"] == 12
    assert report.label_balance["urgency"] == {"low": 6, "high": 6}


def test_render_mentions_the_findings_and_the_balance() -> None:
    report = validate(balanced(4), min_per_class=1)
    rendered = report.render()
    assert "Dataset report: OK" in rendered
    assert "urgency" in rendered
    assert "low=2" in rendered


def test_empty_dataset_is_an_error_not_a_crash() -> None:
    report = validate([])
    assert not report.ok
    assert codes(report) == {"empty"}
    assert report.counts["decisions"] == 0


# ----------------------------------------------------------------- the two big ones


def test_contradictory_labels_are_an_error() -> None:
    rows = [
        *balanced(6),
        make("a", "The printer on floor two is jammed again.", "low"),
        make("b", "The printer on floor two is jammed again.", "high"),
    ]
    report = validate(rows, min_per_class=1)
    assert not report.ok
    finding = next(f for f in report.findings if f.code == "contradictory_labels")
    assert set(finding.decisions) == {"a", "b"}
    assert "urgency" in finding.message
    assert "hard ceiling" in finding.message


def test_contradiction_check_ignores_whitespace_and_case() -> None:
    rows = [
        make("a", "Printer   jammed\nagain.", "low"),
        make("b", "printer jammed again.", "high"),
    ]
    assert "contradictory_labels" in codes(validate(rows, min_per_class=1))


def test_agreeing_duplicates_are_only_a_note() -> None:
    rows = [
        *balanced(6),
        make("a", "The printer on floor two is jammed.", "low"),
        make("b", "The printer on floor two is jammed.", "low"),
    ]
    report = validate(rows, min_per_class=1)
    assert report.ok
    assert "duplicate_rows" in codes(report, "note")


def test_label_imbalance_is_flagged_with_the_share() -> None:
    rows = [make(f"t{i}", f"Ticket {i} text.", "low") for i in range(9)]
    rows.append(make("t9", "Ticket 9 text.", "high"))
    report = validate(rows, min_per_class=1)
    finding = next(f for f in report.findings if f.code == "imbalanced")
    assert "90%" in finding.message
    assert report.label_balance["urgency"] == {"low": 9, "high": 1}


def test_a_balanced_dataset_is_not_flagged_as_imbalanced() -> None:
    assert "imbalanced" not in codes(validate(balanced()))


# ------------------------------------------------------------------- the other checks


def test_duplicate_ids_are_an_error() -> None:
    rows = [make("same", "First ticket."), make("same", "Second ticket.", "high")]
    report = validate(rows, min_per_class=1)
    assert not report.ok
    assert next(f for f in report.findings if f.code == "duplicate_id").decisions == ("same",)


def test_partly_unlabelled_warns_and_wholly_unlabelled_errors() -> None:
    partial = [*balanced(6), make("u1", "An unlabelled ticket.", None)]
    report = validate(partial, min_per_class=1)
    assert report.ok
    assert next(f for f in report.findings if f.code == "unlabelled").decisions == ("u1",)
    assert report.counts["unlabelled"] == 1

    none_labelled = [make(f"u{i}", f"Ticket {i}.", None) for i in range(4)]
    blank = validate(none_labelled)
    assert not blank.ok
    assert "no_labels" in codes(blank, "error")


def test_option_set_mismatch_across_one_key_is_an_error() -> None:
    other = Criterion(
        key="urgency",
        question="How urgent is this ticket?",
        options=(
            Option("low", "Can wait a week"),
            Option("high", "Needs attention today"),
            Option("now", "Drop everything"),
        ),
    )
    rows = [*balanced(4), make("odd", "A three-way ticket.", "now", criterion=other)]
    report = validate(rows, min_per_class=1)
    assert not report.ok
    finding = next(f for f in report.findings if f.code == "option_set_mismatch")
    assert finding.decisions == ("odd",)
    assert "different criteria wearing one name" in finding.message


def test_question_wording_drift_is_a_warning() -> None:
    reworded = Criterion(
        key="urgency", question="How urgent is it?", options=URGENCY.options
    )
    rows = [*balanced(4), make("odd", "A reworded ticket.", "low", criterion=reworded)]
    report = validate(rows, min_per_class=1)
    assert report.ok
    assert "question_mismatch" in codes(report, "warning")


def test_rare_and_unused_classes_are_reported_separately() -> None:
    rows = [make(f"t{i}", f"Ticket {i}.", "low") for i in range(6)]
    report = validate(rows, min_per_class=5)
    assert "unused_option" in codes(report, "warning")
    assert "rare_class" not in codes(report)

    rows.append(make("t6", "Ticket six.", "high"))
    second = validate(rows, min_per_class=5)
    assert "rare_class" in codes(second, "warning")
    assert "high=1" in next(f for f in second.findings if f.code == "rare_class").message


def test_long_states_are_flagged_against_the_budget() -> None:
    rows = [*balanced(4), make("long", "word " * 400, "low")]
    report = validate(rows, min_per_class=1, max_state_tokens=64)
    finding = next(f for f in report.findings if f.code == "long_state")
    assert finding.decisions[0] == "long"
    assert "truncated" in finding.message


def test_one_group_for_everything_is_an_error() -> None:
    rows = [
        make(f"t{i}", f"Ticket {i}.", "low" if i % 2 else "high", group="all") for i in range(6)
    ]
    report = validate(rows, min_per_class=1)
    assert not report.ok
    assert "single_group" in codes(report, "error")


# ------------------------------------------------------------------------ mechanics


def test_estimate_tokens_grows_with_text() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three") >= 3
    assert estimate_tokens("word " * 100) > estimate_tokens("word " * 10)


def test_finding_rejects_an_unknown_severity() -> None:
    with pytest.raises(DataError, match="severity"):
        Finding("fatal", "code", "message")


def test_findings_are_ordered_errors_first() -> None:
    rows = [
        make("same", "The printer is jammed.", "low"),
        make("same", "The printer is jammed.", "high"),
        make("u", "Unlabelled ticket.", None),
    ]
    severities = [f.severity for f in validate(rows, min_per_class=1).findings]
    assert severities == sorted(severities, key=["error", "warning", "note"].index)


# ------------------------------------------------- messages must describe the code


def test_unlabelled_warning_says_what_training_actually_does() -> None:
    """The message promised unlabelled rows 'will be skipped by training'; `finetune`
    raises TrainingError on them. A diagnosis that is wrong is worse than none."""
    from typedecide.data.validate import validate as run
    from typedecide.schema import Criterion, Decision, Option
    from typedecide.training import labelled_only

    criterion = Criterion("k", "q?", (Option("a", "A"), Option("b", "B")))
    rows = [Decision(id=f"l{i}", state=f"s{i}", criterion=criterion, answer_id="ab"[i % 2])
            for i in range(10)]
    rows.append(Decision(id="blank", state="sx", criterion=criterion))
    (finding,) = [f for f in run(rows).findings if f.code == "unlabelled"]
    assert "skipped by training" not in finding.message
    assert "Training refuses" in finding.message and "TrainingError" in finding.message
    assert "labelled_only" in finding.message and "Evaluation skips" in finding.message
    assert finding.decisions == ("blank",)
    assert labelled_only(rows)[1] == ["blank"], "the helper the message points at exists"


def test_long_state_warning_names_the_end_that_is_actually_lost() -> None:
    """Training truncates from the LEFT, so the start of the evidence goes, not the
    tail; and evaluation does not truncate at all."""
    from typedecide.data.validate import validate as run
    from typedecide.schema import Criterion, Decision, Option

    criterion = Criterion("k", "q?", (Option("a", "A"), Option("b", "B")))
    rows = [Decision(id=f"d{i}", state="word " * (3000 if i == 0 else 5) + str(i),
                     criterion=criterion, answer_id="ab"[i % 2]) for i in range(10)]
    (finding,) = [f for f in run(rows).findings if f.code == "long_state"]
    message = finding.message
    assert "truncated from the left" in message and "START of the evidence" in message
    assert "Evaluation does not truncate" in message
    assert "end of a ticket" not in message
    assert finding.decisions == ("d0",)
