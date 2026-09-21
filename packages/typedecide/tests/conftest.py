"""Put `src/` on the path so the suite runs from a checkout without installing.

CI installs the package properly; this is for `pytest` in a fresh clone.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
