from __future__ import annotations

import importlib.util
import pickle
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


def _load_example_module(relative_path: str, module_name: str) -> ModuleType:
    path = _repository_root() / "examples" / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLS_PATH = _load_example_module("_support/pls_component_path.py", "pls_component_path_example")


def test_standard_pls_path_is_deterministic_and_immutable() -> None:
    rng = np.random.default_rng(1729)
    X = pd.DataFrame(rng.normal(size=(30, 6)))
    Y = pd.DataFrame(rng.normal(size=(30, 3)))

    first = PLS_PATH.evaluate_pls_component_path(X, Y, max_n_components=3)
    second = PLS_PATH.evaluate_pls_component_path(X, Y, max_n_components=3)

    np.testing.assert_array_equal(first.n_components, second.n_components)
    np.testing.assert_allclose(first.cv_mse_mean, second.cv_mse_mean)
    np.testing.assert_allclose(first.cv_mse_fold_sd, second.cv_mse_fold_sd)
    assert first.n_components.tolist() == [1, 2, 3]
    assert first.algorithm == "NIPALS"
    assert first.n_splits == 5
    assert first.n_components.dtype == np.dtype(np.intp)
    assert first.cv_mse_mean.dtype == np.dtype(np.float64)
    assert first.cv_mse_fold_sd.dtype == np.dtype(np.float64)
    assert first.cv_mse_standard_error.dtype == np.dtype(np.float64)
    assert np.isfinite(first.cv_mse_mean).all()
    assert (first.cv_mse_fold_sd >= 0.0).all()
    assert not first.n_components.flags.writeable
    assert not first.cv_mse_mean.flags.writeable
    assert not first.cv_mse_fold_sd.flags.writeable
    assert not first.cv_mse_standard_error.flags.writeable
    np.testing.assert_allclose(
        first.cv_mse_standard_error,
        first.cv_mse_fold_sd / np.sqrt(first.n_splits - 1),
    )


def test_standard_pls_path_defensively_copies_and_pickles() -> None:
    components = np.array([1, 2, 3])
    means = np.array([0.9, 0.6, 0.55])
    fold_sd = np.array([0.12, 0.09, 0.08])
    path = PLS_PATH.PLSComponentPath(
        n_components=components,
        cv_mse_mean=means,
        cv_mse_fold_sd=fold_sd,
        algorithm="NIPALS",
        n_splits=5,
    )

    components[0] = 7
    means[0] = 7.0
    fold_sd[0] = 7.0
    assert path.n_components.tolist() == [1, 2, 3]
    assert path.cv_mse_mean.tolist() == [0.9, 0.6, 0.55]
    assert path.cv_mse_fold_sd.tolist() == [0.12, 0.09, 0.08]
    np.testing.assert_allclose(
        path.cv_mse_standard_error,
        np.array([0.12, 0.09, 0.08]) / np.sqrt(path.n_splits - 1),
    )

    restored = pickle.loads(pickle.dumps(path))
    np.testing.assert_array_equal(restored.n_components, path.n_components)
    np.testing.assert_allclose(restored.cv_mse_mean, path.cv_mse_mean)
    np.testing.assert_allclose(restored.cv_mse_fold_sd, path.cv_mse_fold_sd)
    assert not restored.n_components.flags.writeable
    assert not restored.cv_mse_mean.flags.writeable
    assert not restored.cv_mse_fold_sd.flags.writeable
    assert not restored.cv_mse_standard_error.flags.writeable


def test_standard_pls_path_rejects_invalid_arrays() -> None:
    with pytest.raises(ValueError, match="same length"):
        PLS_PATH.PLSComponentPath(
            n_components=[1, 2],
            cv_mse_mean=[0.9],
            cv_mse_fold_sd=[0.1, 0.1],
            algorithm="NIPALS",
            n_splits=5,
        )
    with pytest.raises(ValueError, match="strictly ascending"):
        PLS_PATH.PLSComponentPath(
            n_components=[1, 1],
            cv_mse_mean=[0.9, 0.8],
            cv_mse_fold_sd=[0.1, 0.1],
            algorithm="NIPALS",
            n_splits=5,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        PLS_PATH.PLSComponentPath(
            n_components=[1, 2],
            cv_mse_mean=[0.9, 0.8],
            cv_mse_fold_sd=[0.1, -0.1],
            algorithm="NIPALS",
            n_splits=5,
        )
    one_split_path = PLS_PATH.PLSComponentPath(
        n_components=[1, 2],
        cv_mse_mean=[0.9, 0.8],
        cv_mse_fold_sd=[0.0, 0.0],
        algorithm="NIPALS",
        n_splits=1,
    )
    with pytest.raises(ValueError, match="requires at least two"):
        _ = one_split_path.cv_mse_standard_error


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

        index = n_components - 1
        assert path.cv_mse_mean[index] == pytest.approx(np.mean(split_mse))
        assert path.cv_mse_fold_sd[index] == pytest.approx(np.std(split_mse))
        assert path.cv_mse_standard_error[index] == pytest.approx(
            np.std(split_mse, ddof=1) / np.sqrt(path.n_splits)
        )


def test_dedicated_example_compares_paths_directly_in_memory() -> None:
    examples_dir = _repository_root() / "examples"
    path = examples_dir / "04_pls_path_comparison.py"
    text = path.read_text(encoding="utf-8")

    assert "evaluate_pls_component_path(" in text
    assert "component_path_" in text
    assert "pipls_path" in text
    assert "pls_path" in text
    assert text.count("axis.errorbar(") == 2
    assert "plt.subplots(" in text
    assert "figure.savefig(" in text
    assert "plt.close(figure)" in text
    assert text.count('("pulp",') == 1
    assert text.count('("sugarcane",') == 1
    assert '"tobacco",' in text
    assert "refit=False" not in text
    assert ".to_csv(" not in text
    assert "component_path.csv" not in text
    assert "plot_component_path" not in text
    assert not (examples_dir / "_support" / "plot_component_path.py").exists()


def test_real_data_examples_use_pipls_only_component_paths() -> None:
    examples_dir = _repository_root() / "examples"
    pulp_text = (examples_dir / "05_pulp_real_data.py").read_text(encoding="utf-8")
    assert "component_path_" in pulp_text
    assert "refit=False" not in pulp_text
    assert "path.for_n_components(CHOSEN_N_COMPONENTS)" in pulp_text
    assert "path_search.predictor_rank_profile(selected.n_components)" in pulp_text
    assert 'cv_results["predictor_rank"]' not in pulp_text
    assert "axis.errorbar(" in pulp_text
    assert "component_path.csv" not in pulp_text
    assert "post_analysis.pdf" not in pulp_text
    assert ".to_csv(" not in pulp_text
    assert not (examples_dir / "_support" / "pulp_workflow.py").exists()

    sugarcane_text = (examples_dir / "06_sugarcane_real_data.py").read_text(
        encoding="utf-8"
    )
    assert "component_path_" in sugarcane_text
    assert "refit=False" not in sugarcane_text
    assert "path.for_n_components(CHOSEN_N_COMPONENTS)" in sugarcane_text
    assert "axis.errorbar(" in sugarcane_text
    assert "component_path.csv" not in sugarcane_text
    assert "post_analysis.pdf" not in sugarcane_text
    assert ".to_csv(" not in sugarcane_text

    tobacco_text = (examples_dir / "07_tobacco_real_data.py").read_text(encoding="utf-8")
    assert "component_path_" in tobacco_text
    assert "refit=False" not in tobacco_text
    assert "path.for_n_components(CHOSEN_N_COMPONENTS)" in tobacco_text
    assert "axis.errorbar(" in tobacco_text
    assert "component_path.csv" not in tobacco_text
    assert "post_analysis.pdf" not in tobacco_text
    assert ".to_csv(" not in tobacco_text
    assert not (examples_dir / "_support" / "fixed_model_oof.py").exists()
    assert not (examples_dir / "_support" / "post_analysis_artifacts.py").exists()

    for source in (pulp_text, sugarcane_text, tobacco_text):
        assert "evaluate_pls_component_path(" not in source
        assert "subprocess" not in source
        assert "pooled_oof_r2_" not in source


def test_tobacco_example_uses_full_svd_and_auto_search() -> None:
    text = (_repository_root() / "examples" / "07_tobacco_real_data.py").read_text(
        encoding="utf-8"
    )

    assert 'svd_solver="full"' in text
    assert 'search_method="auto"' in text
    assert 'svd_solver="randomized"' not in text


def test_pls_path_helper_is_not_a_command_line_or_table_wrapper() -> None:
    text = (
        _repository_root() / "examples" / "_support" / "pls_component_path.py"
    ).read_text(encoding="utf-8")

    assert "argparse" not in text
    assert "subprocess" not in text
    assert "if __name__ ==" not in text
    assert "-> PLSComponentPath" in text
    assert "return PLSComponentPath(" in text
    assert "pd.DataFrame(" not in text
    assert ".to_csv(" not in text
