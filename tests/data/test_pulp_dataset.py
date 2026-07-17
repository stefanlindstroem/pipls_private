from __future__ import annotations

import hashlib
import importlib.util
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

EXPECTED_CHECKSUMS = {
    "X.csv": "b26d1639339c406bf91070e2ad8ceb8df59d3ad1f90ece7c5cd5b18264d63b45",
    "Y.csv": "b600264d2319efce3500c6d7adbf5490b62e6d42a1e32af7df62709b1975ce93",
    "LICENSE.txt": "4e5cfb9cbb9214bfa8fb55ef52735799dddd4206749e5386e0d6a12ce965891d",
}


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dataset_dir() -> Path:
    return _root() / "datasets" / "pulp"


def _metadata() -> dict[str, Any]:
    loaded = yaml.safe_load((_dataset_dir() / "metadata.yaml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_pulp_tables_are_aligned_numeric_and_complete() -> None:
    X = pd.read_csv(_dataset_dir() / "X.csv")
    Y = pd.read_csv(_dataset_dir() / "Y.csv")

    assert X.shape == (46, 14)
    assert Y.shape == (46, 8)
    assert list(X.columns) == [
        "Shives", "Fines B", "L (arith)", "L (lw)", "L (llw)",
        "W (arith)", "W (lw)", "W (llw)", "C (arith)", "C (lw)",
        "C (llw)", "F (arith)", "F (lw)", "F (llw)",
    ]
    assert list(Y.columns) == [
        "CSF", "Density", "TI", "Elongation", "TEA", "TSI", "Tear index", "s"
    ]
    assert not X.isna().to_numpy().any()
    assert not Y.isna().to_numpy().any()
    assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes)
    assert all(np.issubdtype(dtype, np.number) for dtype in Y.dtypes)
    assert X.iloc[0].tolist() == pytest.approx(
        [1.815, 22.48, 0.78995, 1.5692, 2.3011, 19.564, 28.357, 33.237,
         0.12756, 0.15411, 0.15392, 0.047752, 0.061746, 0.0597]
    )
    assert Y.iloc[0].tolist() == pytest.approx([253.0, 313.0, 28.2, 1.77, 0.33, 2.97, 6.9, 41.7])


def test_pulp_metadata_records_selection_provenance_and_license() -> None:
    metadata = _metadata()
    assert metadata["schema_version"] == 1
    assert metadata["dataset"]["id"] == "pulp"
    assert metadata["dimensions"] == {
        "n_samples": 46,
        "n_predictors": 14,
        "n_responses": 8,
    }
    assert metadata["source"]["upstream_file"] == "data/pulp.csv"
    assert metadata["source"]["upstream_sha256"].startswith("e5aeb8da")
    assert metadata["source"]["publication"]["doi"] == "10.1016/j.compchemeng.2025.109143"
    assert metadata["license"]["identifier"] == "CC-BY-4.0"
    assert metadata["sample_alignment"]["method"] == "positional"
    assert metadata["missing_values"] == {"predictors": "none", "responses": "none"}
    assert "final k column" in metadata["preparation"]["description"]


def test_pulp_repository_files_match_metadata_checksums() -> None:
    recorded = _metadata()["integrity"]["sha256"]
    assert recorded == EXPECTED_CHECKSUMS
    for filename, expected in EXPECTED_CHECKSUMS.items():
        actual = hashlib.sha256((_dataset_dir() / filename).read_bytes()).hexdigest()
        assert actual == expected


def test_pulp_preparation_script_recreates_tables_from_source_shape(tmp_path: Path) -> None:
    script = _root() / "scripts" / "prepare_data" / "prepare_pulp.py"
    spec = importlib.util.spec_from_file_location("prepare_pulp", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    X = pd.read_csv(_dataset_dir() / "X.csv")
    Y = pd.read_csv(_dataset_dir() / "Y.csv")
    source = pd.DataFrame(index=range(46), columns=module.SOURCE_COLUMNS, dtype=float)
    source.iloc[:, :6] = 0.0
    source.loc[:, module.PREDICTOR_COLUMNS] = X.to_numpy()
    source.loc[:, module.RESPONSE_COLUMNS] = Y.to_numpy()
    source.loc[:, "k"] = 0.0
    source_path = tmp_path / "pulp.csv"
    source.to_csv(source_path, index=False)

    output_dir = tmp_path / "prepared"
    module.prepare_pulp(source_path, output_dir, verify_source=False)
    pd.testing.assert_frame_equal(pd.read_csv(output_dir / "X.csv"), X)
    pd.testing.assert_frame_equal(pd.read_csv(output_dir / "Y.csv"), Y)


def test_pulp_example_reads_files_and_fits(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(_root() / "examples" / "10_pulp_real_data.py"), run_name="__main__")
    output = capsys.readouterr().out
    assert "X shape: (46, 14)" in output
    assert "Y shape: (46, 8)" in output
    assert "selected parameters: {'n_components': 3, 'predictor_rank': 4}" in output
    assert "selection-conditioned pooled OOF R2:" in output
