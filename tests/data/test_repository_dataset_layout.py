from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _dataset_directories() -> list[Path]:
    root = Path(__file__).resolve().parents[2] / "datasets"
    standard_names = ("X.csv", "Y.csv", "metadata.yaml")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and any((path / name).exists() for name in standard_names)
    )


def _load_metadata(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_all_repository_datasets_use_standard_files() -> None:
    dataset_dirs = _dataset_directories()
    assert dataset_dirs

    for data_dir in dataset_dirs:
        assert (data_dir / "X.csv").is_file(), data_dir
        assert (data_dir / "Y.csv").is_file(), data_dir
        assert (data_dir / "metadata.yaml").is_file(), data_dir


def test_all_repository_dataset_metadata_matches_tables() -> None:
    for data_dir in _dataset_directories():
        metadata = _load_metadata(data_dir / "metadata.yaml")
        X = pd.read_csv(data_dir / "X.csv")
        Y = pd.read_csv(data_dir / "Y.csv")

        assert metadata["schema_version"] == 1
        assert metadata["dataset"]["id"] == data_dir.name
        assert metadata["files"] == {"predictors": "X.csv", "responses": "Y.csv"}
        assert metadata["format"] == {
            "type": "csv",
            "delimiter": ",",
            "encoding": "utf-8",
            "header": True,
        }
        assert metadata["dimensions"] == {
            "n_samples": len(X),
            "n_predictors": X.shape[1],
            "n_responses": Y.shape[1],
        }
        assert [item["name"] for item in metadata["predictors"]] == list(X.columns)
        assert [item["name"] for item in metadata["responses"]] == list(Y.columns)
        assert all(item["description"] for item in metadata["predictors"])
        assert all(item["description"] for item in metadata["responses"])
        assert len(X) == len(Y)


def test_all_repository_dataset_integrity_hashes_are_current() -> None:
    for data_dir in _dataset_directories():
        metadata = _load_metadata(data_dir / "metadata.yaml")
        checksums = metadata["integrity"]["sha256"]
        assert "X.csv" in checksums
        assert "Y.csv" in checksums

        for filename, expected in checksums.items():
            actual = hashlib.sha256((data_dir / filename).read_bytes()).hexdigest()
            assert actual == expected
