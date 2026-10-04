"""Shared pytest setup.

Tests that need downloaded models will get a marker and be skipped when the
models are missing (added in Phase 1).
"""

import sys
from pathlib import Path

# Lets the tests import smallproof without `pip install -e .`
# (pyproject.toml sets the same pythonpath for pytest).
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
