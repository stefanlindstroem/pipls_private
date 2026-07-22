from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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
    assert loaded
    assert all(isinstance(key, str) and key for key in loaded)
    return loaded


def _assert_numeric_csv_format(path: Path) -> pd.DataFrame:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert "," in first_line
    assert "\t" not in first_line

    frame = pd.read_csv(path)
    assert not frame.empty
    assert frame.columns.is_unique
    assert all(isinstance(column, str) and column for column in frame.columns)
    assert all(np.issubdtype(dtype, np.number) for dtype in frame.dtypes)
    assert np.isfinite(frame.to_numpy()).all()
    return frame


def test_all_repository_datasets_use_standard_files() -> None:
    dataset_dirs = _dataset_directories()
    assert dataset_dirs

    for data_dir in dataset_dirs:
        assert (data_dir / "X.csv").is_file(), data_dir
        assert (data_dir / "Y.csv").is_file(), data_dir
        assert (data_dir / "metadata.yaml").is_file(), data_dir


def test_all_repository_dataset_files_have_supported_formats() -> None:
    for data_dir in _dataset_directories():
        X = _assert_numeric_csv_format(data_dir / "X.csv")
        Y = _assert_numeric_csv_format(data_dir / "Y.csv")
        _load_metadata(data_dir / "metadata.yaml")
        assert len(X) == len(Y)


def _collect_named_strings(value: object, name: str) -> set[str]:
    if isinstance(value, dict):
        collected: set[str] = set()
        for key, child in value.items():
            if key == name and isinstance(child, str) and child.strip():
                collected.add(child.strip())
            collected.update(_collect_named_strings(child, name))
        return collected
    if isinstance(value, list):
        collected: set[str] = set()
        for child in value:
            collected.update(_collect_named_strings(child, name))
        return collected
    return set()


def test_public_dataset_guide_links_metadata_source_dois() -> None:
    root = Path(__file__).resolve().parents[2]
    guide = (root / "docs" / "datasets.md").read_text(encoding="utf-8")

    for data_dir in _dataset_directories():
        metadata = _load_metadata(data_dir / "metadata.yaml")
        source = metadata.get("source")
        assert isinstance(source, dict), data_dir

        dois = _collect_named_strings(source, "doi")
        urls = _collect_named_strings(source, "url")
        assert dois, data_dir

        for doi in dois:
            url = f"https://doi.org/{doi}"
            assert url in urls, (data_dir, doi)
            assert f"]({url})" in guide, (data_dir, doi)
