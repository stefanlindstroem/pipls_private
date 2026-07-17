from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_plot_module() -> ModuleType:
    path = _repository_root() / "examples" / "plot_component_path.py"
    spec = importlib.util.spec_from_file_location("plot_component_path_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLOT = _load_plot_module()


def _write_component_path(path: Path) -> None:
    pd.DataFrame(
        {
            "n_components": [1, 2, 3],
            "predictor_rank": [4, 5, 5],
            "predictor_rank_policy": ["optimized"] * 3,
            "response_standardized_cv_mse_mean": [0.8, 0.5, 0.45],
            "response_standardized_cv_mse_fold_sd": [0.1, 0.08, 0.07],
            "n_splits": [5, 5, 5],
        }
    ).to_csv(path, index=False)


def test_plot_is_generated_by_reading_the_canonical_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "component_path.csv"
    pdf_path = tmp_path / "component_path.pdf"
    _write_component_path(csv_path)

    PLOT.plot_component_path(csv_path, pdf_path, title="Test component path")

    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert pdf_path.stat().st_size > 1_000


def test_plot_reader_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "incomplete.csv"
    pd.DataFrame({"n_components": [1]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        PLOT.read_component_path(csv_path)


def test_real_data_examples_use_two_stage_artifact_workflow() -> None:
    examples = [
        _repository_root() / "examples" / "10_pulp_real_data.py",
        _repository_root() / "examples" / "11_sugarcane_real_data.py",
    ]
    for path in examples:
        text = path.read_text(encoding="utf-8")
        assert "component_path_results_" in text
        assert "refit=False" in text
        assert "pd.read_csv(COMPONENT_PATH_CSV)" in text
        assert "predictor_rank=chosen_predictor_rank" in text
        assert "best_params_" not in text
        assert "pooled_oof_r2_" not in text
