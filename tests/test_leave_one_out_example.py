from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_leave_one_out_example_runs_and_reports_interpretable_results() -> None:
    root = _repository_root()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")

    completed = subprocess.run(
        [sys.executable, str(root / "examples" / "03_leave_one_out_validation.py")],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = completed.stdout

    assert "Observations: 12" in output
    assert "Leave-one-out splits: 12" in output
    assert "Complete OOF coverage: True" in output
    assert "OOF prediction shape: (12, 2)" in output
    assert "Estimate kind: selection-conditioned" in output
    selected = re.search(
        r"Selected pair: n_components=([12]), predictor_rank=([12])", output
    )
    assert selected is not None
    n_components, predictor_rank = map(int, selected.groups())
    assert 1 <= n_components <= predictor_rank <= 2

    cv_mse = re.search(r"Mean response-standardized CV-MSE: ([0-9.]+)", output)
    pooled_r2 = re.search(r"Pooled OOF R2 \(not mean foldwise R2\): (-?[0-9.]+)", output)
    assert cv_mse is not None and float(cv_mse.group(1)) >= 0.0
    assert pooled_r2 is not None and float(pooled_r2.group(1)) <= 1.0


def test_leave_one_out_example_is_linked_from_the_reference() -> None:
    root = _repository_root()
    path_reference = (root / "docs" / "path_analysis.md").read_text(encoding="utf-8")
    example_catalogue = (root / "docs" / "examples.md").read_text(encoding="utf-8")
    example_readme = (root / "examples" / "README.md").read_text(encoding="utf-8")

    assert "examples.md#leave-one-out-validation" in path_reference
    assert "examples/03_leave_one_out_validation.py" in example_catalogue
    assert "03_leave_one_out_validation.py" in example_readme
    assert "not mean foldwise $R^2$" in path_reference
