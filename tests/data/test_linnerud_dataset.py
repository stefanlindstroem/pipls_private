from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXPECTED_CHECKSUMS = {
    "exercise.csv": "cb8d8c24937643fa2459682efb86c5e667bcd6dd93109eef81964d9e9f11bf8c",
    "physiological.csv": "2bf7e05c1cd7d0adf0eca1e456941f624bed0a4fc96694d60d0ff7853ec5fcf7",
    "LICENSE.txt": "97a461834f569c24a4d78a74d37e35fe051f9a564dd43d066fd920a82706f441",
}


def _dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "datasets" / "linnerud"


def test_linnerud_tables_are_aligned_and_complete() -> None:
    data_dir = _dataset_dir()
    X = pd.read_csv(data_dir / "exercise.csv", sep=r"\s+")
    Y = pd.read_csv(data_dir / "physiological.csv", sep=r"\s+")

    assert X.shape == (20, 3)
    assert Y.shape == (20, 3)
    assert list(X.columns) == ["Chins", "Situps", "Jumps"]
    assert list(Y.columns) == ["Weight", "Waist", "Pulse"]
    assert not X.isna().to_numpy().any()
    assert not Y.isna().to_numpy().any()
    assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes)
    assert all(np.issubdtype(dtype, np.number) for dtype in Y.dtypes)


def test_linnerud_repository_files_match_recorded_checksums() -> None:
    data_dir = _dataset_dir()
    recorded = {}
    for line in (data_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        recorded[filename] = digest

    assert recorded == EXPECTED_CHECKSUMS
    for filename, expected in EXPECTED_CHECKSUMS.items():
        actual = hashlib.sha256((data_dir / filename).read_bytes()).hexdigest()
        assert actual == expected


def test_linnerud_example_reads_files_and_fits(capsys: pytest.CaptureFixture[str]) -> None:
    root = Path(__file__).resolve().parents[2]
    runpy.run_path(str(root / "examples" / "09_linnerud_real_data.py"), run_name="__main__")
    output = capsys.readouterr().out

    assert "X shape: (20, 3)" in output
    assert "Y shape: (20, 3)" in output
    assert "selected predictor rank:" in output
    assert "selection-conditioned pooled OOF R2:" in output
