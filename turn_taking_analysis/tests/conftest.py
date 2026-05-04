"""
pytest configuration for fusion_runner tests.

Adds the scripts/ directory to sys.path so `import fusion_runner` and
`import fusion_lib` work from any test file. Forces matplotlib to a
non-interactive backend so PNG saves work in headless environments.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

_HERE = Path(__file__).resolve().parent
_SCRIPTS_DIR = _HERE.parent / "scripts"

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
