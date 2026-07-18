from __future__ import annotations

import runpy
from pathlib import Path

from threadpoolctl import threadpool_limits


def test_tobacco_example_runs() -> None:
    root = Path(__file__).resolve().parents[2]
    with threadpool_limits(limits=1):
        runpy.run_path(
            str(root / "examples" / "12_tobacco_real_data.py"),
            run_name="__main__",
        )
