"""Running an evaluation, and reporting it the way a sceptic would read it.

The failure this module exists to catch is a model that answers by option position
rather than by reading the evidence. On a balanced set that model scores chance, and
"chance" in an accuracy column looks like a weak model rather than a broken one. So
three numbers are reported together and none of them is optional:

  balanced accuracy   is it right, with every class weighted equally
  order consistency   does the answer survive reordering the options
  letter distribution which letters did it actually pick

`debias` goes further than reporting. Scoring a decision under several orderings and
averaging the probability assigned to each *option id* cancels a position prior
exactly: under a cyclic or full permutation set every option spends the same time in
every slot, so whatever bonus the slot carries is added to every option equally and
falls out of the argmax. That repairs the readout. It does not repair the model --
order consistency and the letter histogram will still show the prior, which is why
they are reported even when debiasing is on.
"""

from __future__ import annotations

import itertools
import logging
import math
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ConfigError, DataError, ScorerError
from ..schema import LETTERS, Criterion, Decision, fingerprint
from . import metrics
from .readout import ScoreFn, ScoreRequest, load_readout

LOGGER = logging.getLogger("typedecide.evaluation.runner")

__all__ = ["DEBIAS_MODES", "EvalResult", "evaluate", "evaluate_with_scorer", "orderings"]

# The state used to measure what the model answers with no evidence at all. Short and
# content-free on purpose; it mirrors the contextual-calibration baseline used in
# bench/authored144.mjs so Python and browser numbers stay comparable.
NULL_STATE = "N/A"

DEBIAS_MODES = ("none", "cyclic", "permutation", "calibrated")

# 7! is 5040 forward passes per decision. Past this, say so rather than appearing to
# hang; cyclic averaging cancels a position prior just as exactly and costs n passes.
_MAX_PERMUTATION_OPTIONS = 6

_SEPARATOR = "|#|"


def orderings(n_options: int, debias: str) -> list[tuple[int, ...]]:
    """The option orderings to score for one decision under a debiasing mode.

    "none" scores the options as given. "cyclic" and "calibrated" score the n
    rotations. "permutation" scores all n! arrangements. Every mode but "none" puts
    each option in each position the same number of times, which is what makes the
    position prior cancel under averaging.
    """
    if debias not in DEBIAS_MODES:
        raise ConfigError(
            f"debias={debias!r} is not a mode. Choose one of: {', '.join(DEBIAS_MODES)}."
        )
    identity = tuple(range(n_options))
    if debias == "none":
        return [identity]
    if debias == "permutation":
        if n_options > _MAX_PERMUTATION_OPTIONS:
            raise ConfigError(
                f"debias='permutation' would score {n_options}! orderings of a "
                f"{n_options}-option decision. Use debias='cyclic', which cancels a "
                f"position prior with {n_options} orderings instead."
            )
        return list(itertools.permutations(range(n_options)))
    return [tuple((i + k) % n_options for i in range(n_options)) for k in range(n_options)]


@dataclass(frozen=True)
class EvalResult:
    """One evaluation, with everything needed to argue about it."""

    accuracy: float
    balanced_accuracy: float
    order_consistency: float | None
    ece: float | None
    brier: float | None
    per_criterion: dict[str, float]
    letter_distribution: dict[str, float]
    n: int
    manifest: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """A short human-readable block, for the CLI and for pasting into a log."""
        chance = self.manifest.get("chance")
        model = self.manifest.get("model", "(injected scorer)")
        adapter = self.manifest.get("adapter")
        lines = [
            f"model              {model}" + (f" + {adapter}" if adapter else ""),
            f"decisions          {self.n}",
            "debias             "
            f"{self.manifest.get('debias', 'none')}"
            f"   {self.manifest.get('orderings_scored', '?')} orderings scored",
            f"accuracy           {self.accuracy:.1%}"
            + (f"   (chance {chance:.1%})" if isinstance(chance, float) else ""),
            f"balanced accuracy  {self.balanced_accuracy:.1%}",
        ]
        if self.order_consistency is None:
            lines.append(
                "order consistency  n/a     only one ordering was scored, so nothing "
                "here would expose a position prior"
            )
        else:
            lines.append(
                f"order consistency  {self.order_consistency:.1%}"
                "   same answer under every ordering"
            )
        if self.ece is not None:
            lines.append(f"ece                {self.ece:.3f}")
        if self.brier is not None:
            lines.append(f"brier              {self.brier:.3f}")
        letters = "  ".join(f"{k}:{v:.0%}" for k, v in sorted(self.letter_distribution.items()))
        lines.append(f"letters picked     {letters}")
        if self.per_criterion:
            lines.append("\nper criterion (balanced accuracy):")
            for key, value in sorted(self.per_criterion.items()):
                lines.append(f"  {key:<20} {value:.3f}")
        return "\n".join(lines)


# --------------------------------------------------------------------- internals


def _null_key(criterion: Criterion) -> str:
    """Cache key for a content-free prior: the question and this exact ordering.

    The prior depends on the question and on which option sits in which slot, and on
    nothing else -- so one content-free forward pass serves every decision that asks
    that question with that arrangement.
    """
    ids = "/".join(criterion.option_ids)
    return f"{criterion.key}{_SEPARATOR}{criterion.question}{_SEPARATOR}{ids}"


def _score_all(score: ScoreFn, requests: Sequence[ScoreRequest]) -> list[list[float]]:
    """Call a scorer and check it kept its half of the bargain."""
    if not requests:
        return []
    raw = score(requests)
    # `ScorerError` is an `EvaluationError` that also inherits `ConfigError`, which is
    # what 0.1.0 raised here, so callers written against either keep working.
    try:
        rows = [[float(value) for value in vector] for vector in raw]
    except (TypeError, ValueError) as error:
        raise ScorerError(
            "The scorer did not return a sequence of numeric probability vectors "
            f"({error}). A scorer must return one list of floats per request, in order."
        ) from error
    if len(rows) != len(requests):
        raise ScorerError(
            f"The scorer returned {len(rows)} probability vectors for {len(requests)} "
            "requests. A scorer must return one vector per request, in order."
        )
    for index, (vector, request) in enumerate(zip(rows, requests, strict=True)):
        expected = len(request.criterion.options)
        if len(vector) != expected:
            raise ScorerError(
                f"The scorer returned {len(vector)} probabilities for request {index}, "
                f"which lists {expected} options."
            )
        if any(not math.isfinite(value) for value in vector):
            # NaN is not < 0, so it used to pass the check below, lose every comparison
            # in the argmax, and score the row as "first option" with no sign of trouble.
            raise ScorerError(
                f"The scorer returned a non-finite probability (NaN or inf) for request "
                f"{index}. That is usually a half-precision overflow in the model; score in "
                "float32 or bfloat16."
            )
        if any(value < 0 for value in vector):
            raise ScorerError(f"The scorer returned a negative probability for request {index}.")
    return rows


def _divide_out(probs: Sequence[float], prior: Sequence[float]) -> list[float]:
    """Contextual calibration: divide by the content-free prior, then renormalise."""
    adjusted = [value / max(prior[i], 1e-9) for i, value in enumerate(probs)]
    total = sum(adjusted)
    if total <= 0:
        return [1.0 / len(adjusted)] * len(adjusted)
    return [value / total for value in adjusted]


def _normalised(values: Sequence[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


# ------------------------------------------------------------------ the evaluation


def evaluate_with_scorer(
    decisions: Sequence[Decision],
    score: ScoreFn,
    *,
    debias: str = "none",
    seed: int = 0,
    manifest: dict[str, Any] | None = None,
) -> EvalResult:
    """Score decisions with an injected `ScoreFn`.

    `evaluate` is this function with a model attached. Keeping the scorer injectable
    is not only for tests: it is also how you score an ONNX export, a remote endpoint
    or a cached set of logits without reimplementing the debiasing.
    """
    if debias not in DEBIAS_MODES:
        raise ConfigError(
            f"debias={debias!r} is not a mode. Choose one of: {', '.join(DEBIAS_MODES)}."
        )

    rows = [d for d in decisions if d.labelled]
    skipped = len(decisions) - len(rows)
    if skipped:
        LOGGER.warning(
            "Skipping %d unlabelled decision(s): there is nothing to score them against.",
            skipped,
        )
    if not rows:
        raise DataError(
            "None of the decisions given have an answer_id, so there is nothing to "
            "score against. Evaluation needs a labelled set."
        )

    started = time.time()

    # One job per (decision, ordering). Building them all up front lets the scorer
    # batch across decisions as well as across orderings.
    orders_per_row = [orderings(len(d.criterion.options), debias) for d in rows]
    jobs: list[int] = []
    requests: list[ScoreRequest] = []
    for row_index, (decision, orders) in enumerate(zip(rows, orders_per_row, strict=True)):
        for order in orders:
            requests.append(ScoreRequest(decision.state, decision.criterion.reordered(order)))
            jobs.append(row_index)

    probs = _score_all(score, requests)

    priors: dict[str, list[float]] = {}
    if debias == "calibrated":
        null_requests: list[ScoreRequest] = []
        null_keys: list[str] = []
        for request in requests:
            key = _null_key(request.criterion)
            if key in priors:
                continue
            priors[key] = []  # reserve the slot so the key is scored exactly once
            null_keys.append(key)
            null_requests.append(ScoreRequest(NULL_STATE, request.criterion))
        null_probs = _score_all(score, null_requests)
        priors = dict(zip(null_keys, null_probs, strict=True))

    # Average the probability per OPTION ID, never per position. The whole point of
    # scoring several orderings is that position is the thing we are trying to
    # forget, so the accumulator is keyed by identity.
    totals = [dict.fromkeys(d.criterion.option_ids, 0.0) for d in rows]
    picks: list[list[str]] = [[] for _ in rows]
    picked_letters: list[str] = []

    for row_index, request, vector in zip(jobs, requests, probs, strict=True):
        criterion = request.criterion
        if debias == "calibrated":
            vector = _divide_out(vector, priors[_null_key(criterion)])
        share = 1.0 / len(orders_per_row[row_index])
        for position, option in enumerate(criterion.options):
            totals[row_index][option.id] += vector[position] * share
        best = max(range(len(vector)), key=vector.__getitem__)
        picks[row_index].append(criterion.options[best].id)
        picked_letters.append(LETTERS[best])

    gold = [d.answer_id or "" for d in rows]
    predictions = [max(total, key=total.__getitem__) for total in totals]
    groups = [d.criterion.key for d in rows]

    prob_vectors = [
        _normalised([totals[i][option_id] for option_id in d.criterion.option_ids])
        for i, d in enumerate(rows)
    ]
    confidences = [
        vector[d.criterion.index_of(predictions[i])] for i, (d, vector) in
        enumerate(zip(rows, prob_vectors, strict=True))
    ]
    hits = [predictions[i] == gold[i] for i in range(len(rows))]

    scored_orderings = max(len(orders) for orders in orders_per_row)
    consistency = (
        metrics.order_consistency(picks) if scored_orderings > 1 else None
    )
    widest = max(len(d.criterion.options) for d in rows)

    run_manifest: dict[str, Any] = {
        "debias": debias,
        "orderings_scored": scored_orderings,
        "forward_passes": len(requests) + len(priors),
        "decisions": len(rows),
        "unlabelled_skipped": skipped,
        "fingerprint": fingerprint(rows),
        "chance": metrics.chance_accuracy([len(d.criterion.options) for d in rows]),
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(started)),
        "seconds": round(time.time() - started, 3),
    }
    if manifest:
        run_manifest.update(manifest)

    return EvalResult(
        accuracy=metrics.accuracy(gold, predictions),
        balanced_accuracy=metrics.balanced_accuracy(gold, predictions),
        order_consistency=consistency,
        ece=metrics.expected_calibration_error(confidences, hits),
        brier=metrics.brier_score(prob_vectors, [d.answer_index for d in rows]),
        per_criterion={
            key: metrics.balanced_accuracy(
                [gold[i] for i in range(len(rows)) if groups[i] == key],
                [predictions[i] for i in range(len(rows)) if groups[i] == key],
            )
            for key in sorted(set(groups))
        },
        letter_distribution=metrics.letter_distribution(
            picked_letters, letters=list(LETTERS[:widest])
        ),
        n=len(rows),
        manifest=run_manifest,
    )


def evaluate(
    decisions: Sequence[Decision],
    model_id: str,
    *,
    adapter: Path | None = None,
    debias: str = "none",
    batch_size: int = 8,
    device: str | None = None,
    seed: int = 0,
) -> EvalResult:
    """Load a model and score `decisions` with it.

    `debias` selects how many orderings each decision is scored under; see
    `orderings`. Requires the `train` extra (torch, transformers, and peft if an
    adapter is given).
    """
    if debias not in DEBIAS_MODES:
        raise ConfigError(
            f"debias={debias!r} is not a mode. Choose one of: {', '.join(DEBIAS_MODES)}."
        )
    readout = load_readout(
        model_id, adapter=adapter, device=device, batch_size=batch_size, seed=seed
    )
    return evaluate_with_scorer(
        decisions,
        readout,
        debias=debias,
        seed=seed,
        manifest={
            "model": model_id,
            "adapter": None if adapter is None else str(adapter),
            "device": readout.device,
            "batch_size": batch_size,
        },
    )
