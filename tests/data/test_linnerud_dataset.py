from __future__ import annotations

import hashlib
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

EXPECTED_CHECKSUMS = {
    "X.csv": "b40a7d43a02d401e764a44aa11be7ee28bc907960f73cf267014e95b27ddcc1f",
    "Y.csv": "8f8120e958a60b5f09466b575570f6eb36c7c35da58cca575ad14c0d0e9eeb86",
    "LICENSE.txt": "97a461834f569c24a4d78a74d37e35fe051f9a564dd43d066fd920a82706f441",
}


def _dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "datasets" / "linnerud"


def _metadata() -> dict[str, Any]:
    loaded = yaml.safe_load((_dataset_dir() / "metadata.yaml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_linnerud_uses_repository_dataset_layout() -> None:
    data_dir = _dataset_dir()

    assert (data_dir / "X.csv").is_file()
    assert (data_dir / "Y.csv").is_file()
    assert (data_dir / "metadata.yaml").is_file()
    assert not (data_dir / "exercise.csv").exists()
    assert not (data_dir / "physiological.csv").exists()
    assert not (data_dir / "checksums.sha256").exists()


def test_linnerud_tables_are_comma_delimited_aligned_and_complete() -> None:
    data_dir = _dataset_dir()
    X = pd.read_csv(data_dir / "X.csv")
    Y = pd.read_csv(data_dir / "Y.csv")

    assert X.shape == (20, 3)
    assert Y.shape == (20, 3)
    assert list(X.columns) == ["Chins", "Situps", "Jumps"]
    assert list(Y.columns) == ["Weight", "Waist", "Pulse"]
    assert not X.isna().to_numpy().any()
    assert not Y.isna().to_numpy().any()
    assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes)
    assert all(np.issubdtype(dtype, np.number) for dtype in Y.dtypes)

    assert X.iloc[0].tolist() == [5, 162, 60]
    assert Y.iloc[0].tolist() == [191, 36, 50]


def test_linnerud_metadata_matches_repository_contract() -> None:
    metadata = _metadata()

    assert metadata["schema_version"] == 1
    assert metadata["dataset"]["id"] == "linnerud"
    assert metadata["files"] == {"predictors": "X.csv", "responses": "Y.csv"}
    assert metadata["format"] == {
        "type": "csv",
        "delimiter": ",",
        "encoding": "utf-8",
        "header": True,
    }
    assert metadata["dimensions"] == {
        "n_samples": 20,
        "n_predictors": 3,
        "n_responses": 3,
    }
    assert [item["name"] for item in metadata["predictors"]] == [
        "Chins",
        "Situps",
        "Jumps",
    ]
    assert [item["name"] for item in metadata["responses"]] == [
        "Weight",
        "Waist",
        "Pulse",
    ]
    assert all(item["description"] for item in metadata["predictors"])
    assert all(item["description"] for item in metadata["responses"])
    assert metadata["sample_alignment"]["method"] == "positional"
    assert metadata["missing_values"] == {"predictors": "none", "responses": "none"}
    assert metadata["license"]["identifier"] == "BSD-3-Clause"
    assert metadata["preparation"]["description"]


def test_linnerud_repository_files_match_metadata_checksums() -> None:
    data_dir = _dataset_dir()
    recorded = _metadata()["integrity"]["sha256"]

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
