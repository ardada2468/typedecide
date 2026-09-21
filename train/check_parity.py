"""Prove that web/prompt.js and train/prompt.py build byte-identical prompts.

Run this before every training run. A silent drift here is the single most
expensive mistake available in this project.
"""

import json
import os
import pathlib
import subprocess
import sys

import prompt as py

ROOT = pathlib.Path(__file__).resolve().parent.parent

CASES = [
    ("Ticket #1 — charged twice for order A-9923.\nAttached: two statements.",
     {"question": "Which queue should handle this ticket?",
      "options": [{"id": "billing", "description": "Billing and payments"},
                  {"id": "auth", "description": "Account access and authentication"}]}),
    ("ग्राहक ने दो बार भुगतान किया।",
     {"question": "Is the customer dissatisfied?",
      "options": [{"id": "yes", "description": "Yes"},
                  {"id": "no", "description": "No"}]}),
]

script = """
import { fullPrompt } from '%s';
const cases = JSON.parse(process.env.PROMPT_CASES);
console.log(JSON.stringify(cases.map(([s, c]) => fullPrompt(s, c))));
""" % (ROOT / "web" / "prompt.js").as_uri()

node = subprocess.run(
    ["node", "--input-type=module", "-e", script],
    capture_output=True, text=True,
    env={**os.environ, "PROMPT_CASES": json.dumps(CASES)},
)
if node.returncode:
    sys.exit(f"Could not run web/prompt.js through node:\n{node.stderr}")

from_js = json.loads(node.stdout)
from_py = [py.full_prompt(state, criterion) for state, criterion in CASES]

ok = True
for index, (a, b) in enumerate(zip(from_js, from_py)):
    if a != b:
        ok = False
        print(f"case {index} DIFFERS")
        print(f"  js: {a!r}")
        print(f"  py: {b!r}")

print("prompt parity: OK" if ok else "prompt parity: FAILED")
sys.exit(0 if ok else 1)
