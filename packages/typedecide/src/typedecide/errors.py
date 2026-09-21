"""Every error this library raises, so a caller can catch what it means to catch.

Messages name the thing that is wrong and what would fix it. A stack trace is not a
diagnosis, and someone running a twelve-hour fine-tune deserves better than one.
"""

from __future__ import annotations


class TypeDecideError(Exception):
    """Base class. Catch this to catch anything this library raises deliberately."""


class SchemaError(TypeDecideError):
    """A decision, criterion or option is not well formed."""


class DataError(TypeDecideError):
    """A dataset could not be read, or is unusable as given."""


class PromptError(TypeDecideError):
    """The prompt and the tokenizer disagree, so a readout would be silently wrong."""


class TrainingError(TypeDecideError):
    """Training cannot start, or cannot be trusted if it did."""


class ExportError(TypeDecideError):
    """A model could not be turned into the artefact that was asked for."""


class ConfigError(TypeDecideError):
    """A configuration file or CLI invocation does not describe a runnable job."""


class EvaluationError(TypeDecideError):
    """An evaluation started from usable inputs and still could not produce a score.

    Distinct from `ConfigError` (the job was described wrongly) and `DataError` (the
    dataset is unusable): this is what goes wrong *while* scoring -- most often a
    scorer that breaks its contract by returning the wrong number of probability
    vectors, a vector of the wrong width, or values that are not probabilities.
    """


class ScorerError(EvaluationError, ConfigError):
    """A `ScoreFn` broke its contract: wrong shape, or values that are not probabilities.

    Catch `EvaluationError`. This class also inherits `ConfigError` purely for backward
    compatibility: 0.1.0 raised `ConfigError` for a misbehaving scorer, so
    `except ConfigError` written against that release keeps working. The `ConfigError`
    base is deprecated and will be dropped in a future major version.
    """
