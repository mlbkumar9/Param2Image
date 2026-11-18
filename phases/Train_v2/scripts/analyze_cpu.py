"""CPU-only analysis CLI: runs the colocated implementation module."""
from __future__ import annotations

import runpy
from pathlib import Path

_IMPL = Path(__file__).with_name("_analyze_cpu_impl.py")
if not _IMPL.exists():
    raise SystemExit(f"CPU analysis implementation not found: {_IMPL}")
runpy.run_path(str(_IMPL), run_name="__main__")
