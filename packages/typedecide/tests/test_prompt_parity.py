"""typedecide.prompt must build byte-identical text to web/prompt.js.

If the two drift, a fine-tune optimises a prompt the browser never sends and accuracy
quietly collapses. `train/check_parity.py` guards `train/prompt.py`; this guards the
library's own copy, by running the real JavaScript through node.

Skips -- loudly, with the reason -- when node is not installed or when the tests are run
from an sdist, which does not carry `web/`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from typedecide import prompt
from typedecide.schema import LETTERS, Criterion, Option

PROMPT_JS = Path(__file__).resolve().parents[3] / "web" / "prompt.js"

TWENTY = [{"id": f"o{i}", "description": f"Option number {i}"} for i in range(20)]

CASES: list[tuple[str, dict[str, Any]]] = [
    ("Ticket #1 — charged twice for order A-9923.\nAttached: two statements.",
     {"question": "Which queue should handle this ticket?",
      "options": [{"id": "billing", "description": "Billing and payments"},
                  {"id": "auth", "description": "Account access and authentication"}]}),
    ("ग्राहक ने दो बार "
     "भुगतान किया।",
     {"question": "Is the customer dissatisfied?",
      "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}]}),
    ("  leading and trailing whitespace, a tab\there, CRLF\r\nand an emoji \U0001f600  ",
     {"question": "Quotes \"double\" 'single' `back` and a ${template} and a \\ backslash?",
      "options": [{"id": "a", "description": "  padded description  "},
                  {"id": "b", "description": "multi\nline"},
                  {"id": "c", "description": "éè 中文 العربية"}]}),
    ("Every letter the library renders.", {"question": "Which of twenty?", "options": TWENTY}),
]

SCRIPT = """
import { fullPrompt, headText, tailText, INSTRUCTION, LETTERS } from %s;
const cases = JSON.parse(process.env.PROMPT_CASES);
console.log(JSON.stringify({
  instruction: INSTRUCTION,
  letters: LETTERS,
  full: cases.map(([s, c]) => fullPrompt(s, c)),
  head: cases.map(([s]) => headText(s)),
  tail: cases.map(([, c]) => tailText(c)),
}));
"""


@pytest.fixture(scope="module")
def from_js() -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so web/prompt.js cannot be run")
    if not PROMPT_JS.is_file():
        pytest.skip(f"{PROMPT_JS} is not present (running outside the repository checkout)")
    result = subprocess.run(
        [node, "--input-type=module", "-e", SCRIPT % json.dumps(PROMPT_JS.as_uri())],
        capture_output=True, text=True, encoding="utf-8", timeout=60, check=False,
        env={**os.environ, "PROMPT_CASES": json.dumps(CASES)},
    )
    assert result.returncode == 0, f"node could not run web/prompt.js:\n{result.stderr}"
    loaded: dict[str, Any] = json.loads(result.stdout)
    return loaded


def test_constants_match(from_js: dict[str, Any]) -> None:
    assert from_js["instruction"] == prompt.INSTRUCTION
    assert from_js["letters"] == prompt.LETTERS == LETTERS


@pytest.mark.parametrize("index", range(len(CASES)))
def test_prompts_are_byte_identical(from_js: dict[str, Any], index: int) -> None:
    state, criterion = CASES[index]
    assert from_js["full"][index].encode("utf-8") == prompt.full_prompt(state, criterion).encode()
    assert from_js["head"][index] == prompt.head_text(state)
    assert from_js["tail"][index] == prompt.tail_text(criterion)


@pytest.mark.parametrize("index", range(len(CASES)))
def test_a_criterion_object_renders_like_its_dict(from_js: dict[str, Any], index: int) -> None:
    """The library passes `Criterion`s, the parity script passes dicts. Same bytes."""
    state, raw = CASES[index]
    criterion = Criterion(
        key="k", question=raw["question"],
        options=tuple(Option(o["id"], o["description"]) for o in raw["options"]),
    )
    assert prompt.full_prompt(state, criterion) == from_js["full"][index]
