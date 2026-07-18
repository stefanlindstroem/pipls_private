from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_example_module(filename: str, module_name: str) -> ModuleType:
    path = _repository_root() / "examples" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLOT = _load_example_module("plot_component_path.py", "plot_component_path_example")
PLS_PATH = _load_example_module("pls_component_path.py", "pls_component_path_example")


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


def _write_pls_component_path(path: Path, components: list[int] | None = None) -> None:
    values = [1, 2, 3] if components is None else components
    pd.DataFrame(
        {
            "n_components": values,
            "algorithm": ["NIPALS"] * len(values),
            "response_standardized_cv_mse_mean": [0.9, 0.6, 0.55][: len(values)],
            "response_standardized_cv_mse_fold_sd": [0.12, 0.09, 0.08][: len(values)],
            "n_splits": [5] * len(values),
        }
    ).to_csv(path, index=False)


def test_plot_is_generated_from_two_canonical_csvs(tmp_path: Path) -> None:
    csv_path = tmp_path / "component_path.csv"
    pls_csv_path = tmp_path / "pls_component_path.csv"
    pdf_path = tmp_path / "component_path.pdf"
    _write_component_path(csv_path)
    _write_pls_component_path(pls_csv_path)

    PLOT.plot_component_path(
        csv_path,
        pls_csv_path,
        pdf_path,
        title="Test component path",
    )

    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert pdf_path.stat().st_size > 1_000


def test_plot_reader_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "incomplete.csv"
    pd.DataFrame({"n_components": [1]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        PLOT.read_component_path(csv_path)


def test_plot_rejects_mismatched_component_counts(tmp_path: Path) -> None:
    csv_path = tmp_path / "component_path.csv"
    pls_csv_path = tmp_path / "pls_component_path.csv"
    pdf_path = tmp_path / "component_path.pdf"
    _write_component_path(csv_path)
    _write_pls_component_path(pls_csv_path, [1, 2])

    with pytest.raises(ValueError, match="same component counts"):
        PLOT.plot_component_path(
            csv_path,
            pls_csv_path,
            pdf_path,
            title="Test component path",
        )


def test_standard_pls_path_is_deterministic_and_machine_readable() -> None:
    rng = np.random.default_rng(1729)
    X = pd.DataFrame(rng.normal(size=(30, 6)))
    Y = pd.DataFrame(rng.normal(size=(30, 3)))

    first = PLS_PATH.evaluate_pls_component_path(X, Y, max_n_components=3)
    second = PLS_PATH.evaluate_pls_component_path(X, Y, max_n_components=3)

    pd.testing.assert_frame_equal(first, second)
    assert tuple(first.columns) == PLS_PATH.RESULT_COLUMNS
    assert first["n_components"].tolist() == [1, 2, 3]
    assert first["algorithm"].tolist() == ["NIPALS"] * 3
    assert first["n_splits"].tolist() == [5] * 3
    assert np.isfinite(first["response_standardized_cv_mse_mean"]).all()
    assert (first["response_standardized_cv_mse_fold_sd"] >= 0.0).all()


def test_nested_pls_path_matches_separate_pls_fits() -> None:
    rng = np.random.default_rng(2718)
    X = pd.DataFrame(rng.normal(size=(30, 6)))
    Y = pd.DataFrame(rng.normal(size=(30, 3)))
    path = PLS_PATH.evaluate_pls_component_path(X, Y, max_n_components=3)

    splitter = KFold(n_splits=5, shuffle=False)
    for n_components in range(1, 4):
        split_mse: list[float] = []
        for train, validation in splitter.split(X):
            model = PLSRegression(n_components=n_components, scale=True)
            model.fit(X.iloc[train], Y.iloc[train])
            prediction = model.predict(X.iloc[validation])
            response_scale = Y.iloc[train].std(axis=0, ddof=1).to_numpy()
            residual = Y.iloc[validation].to_numpy() - prediction
            split_mse.append(float(np.mean((residual / response_scale) ** 2)))

        row = path.loc[path["n_components"] == n_components].iloc[0]
        assert row["response_standardized_cv_mse_mean"] == pytest.approx(
            np.mean(split_mse)
        )
        assert row["response_standardized_cv_mse_fold_sd"] == pytest.approx(
            np.std(split_mse)
        )


def test_real_data_examples_use_two_stage_comparison_workflow() -> None:
    examples = [
        _repository_root() / "examples" / "10_pulp_real_data.py",
        _repository_root() / "examples" / "11_sugarcane_real_data.py",
        _repository_root() / "examples" / "12_tobacco_real_data.py",
    ]
    for path in examples:
        text = path.read_text(encoding="utf-8")
        assert "component_path_results_" in text
        assert "refit=False" in text
        assert "PLS_COMPONENT_PATH_CSV" in text
        assert "from pls_component_path import evaluate_pls_component_path" in text
        assert "from plot_component_path import plot_component_path" in text
        assert "evaluate_pls_component_path(" in text
        assert "plot_component_path(" in text
        assert 'pd.read_csv(COMPONENT_PATH_CSV).set_index("n_components")' in text
        assert "predictor_rank=chosen_predictor_rank" in text
        assert "subprocess" not in text
        assert "best_params_" not in text
        assert "pooled_oof_r2_" not in text


def test_tobacco_example_uses_full_svd_and_auto_search() -> None:
    path = _repository_root() / "examples" / "12_tobacco_real_data.py"
    text = path.read_text(encoding="utf-8")

    assert 'svd_solver="full"' in text
    assert 'search_method="auto"' in text
    assert 'svd_solver="randomized"' not in text


def test_example_helpers_are_importable_functions_not_command_line_wrappers() -> None:
    for filename in ["pls_component_path.py", "plot_component_path.py"]:
        text = (_repository_root() / "examples" / filename).read_text(encoding="utf-8")
        assert "argparse" not in text
        assert "subprocess" not in text
        assert "if __name__ ==" not in text
