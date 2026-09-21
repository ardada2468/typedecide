"""Small seams over optional dependencies whose spelling changed between releases.

Private. Nothing here imports torch or transformers; callers hand in the callable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def from_pretrained_with_dtype(loader: Callable[..., Any], model_id: str, dtype: Any) -> Any:
    """Call a transformers ``from_pretrained`` with the dtype keyword this release spells.

    transformers 4.56 renamed ``torch_dtype`` to ``dtype``. Older releases forward an
    unknown ``dtype`` to the model's ``__init__``, which raises a ``TypeError`` naming
    it; only that specific failure is retried with the old spelling, so an unrelated
    ``TypeError`` is never swallowed.

    ``trust_remote_code`` is deliberately not a parameter. It is never passed, so it
    stays at transformers' default of False: a model id is data, and loading it must
    not execute the Python published beside it.
    """
    try:
        return loader(model_id, dtype=dtype)
    except TypeError as error:
        if "dtype" not in str(error):
            raise
    return loader(model_id, torch_dtype=dtype)
