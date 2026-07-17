from __future__ import annotations

import runpy
from pathlib import Path


def test_linnerud_example_runs() -> None:
    root = Path(__file__).resolve().parents[2]
    runpy.run_path(str(root / "examples" / "09_linnerud_real_data.py"), run_name="__main__")
