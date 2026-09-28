"""Shared pytest setup.

Tests that need downloaded models will get a marker and be skipped when the
models are missing (added in Phase 1).
"""

import sys
from pathlib import Path

# Temporary: put the project root on the import path so `import core` works.
# Not needed once the package is installable (pyproject.toml, Phase 1).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
