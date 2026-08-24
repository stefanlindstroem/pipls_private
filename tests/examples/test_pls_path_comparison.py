from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from matplotlib.figure import Figure

from pipls import PiPLSRegression, PiPLSSearchCV


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_comparison_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    examples_dir = _repository_root() / "examples"
    monkeypatch.syspath_prepend(str(examples_dir))
    path = examples_dir / "03_pls_path_comparison.py"
    spec = importlib.util.spec_from_file_location("pls_path_comparison_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pulp_comparison_uses_one_protocol_for_all_three_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written_paths: list[Path] = []

    def _capture_savefig(
        self: Figure,
        path: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        del args, kwargs
        written_paths.append(Path(path))

    monkeypatch.setattr(Figure, "savefig", _capture_savefig)
    module = _load_comparison_module(monkeypatch)
    result = module.run_dataset("pulp")

    cross_search = result.cross_covariance_search
    least_squares_search = result.least_squares_search
    splits = result.cv_splits

    assert isinstance(cross_search, PiPLSSearchCV)
    assert isinstance(least_squares_search, PiPLSSearchCV)
    assert cross_search.cv is splits
    assert least_squares_search.cv is splits
    assert cross_search.n_splits_ == least_squares_search.n_splits_ == 5
    assert result.pls_path.n_splits == 5
    assert isinstance(cross_search.estimator, PiPLSRegression)
    assert isinstance(least_squares_search.estimator, PiPLSRegression)
    assert cross_search.estimator.response_subspace == "cross_covariance"
    assert least_squares_search.estimator.response_subspace == "least_squares"
    assert cross_search.search_method == least_squares_search.search_method == "exhaustive"

    cross_path = cross_search.component_path_
    least_squares_path = least_squares_search.component_path_
    np.testing.assert_array_equal(
        cross_path.n_components,
        least_squares_path.n_components,
    )
    np.testing.assert_array_equal(
        cross_path.n_components,
        result.pls_path.n_components,
    )
    assert np.isfinite(cross_path.cv_mse_mean).all()
    assert np.isfinite(least_squares_path.cv_mse_mean).all()
    assert np.isfinite(result.pls_path.cv_mse_mean).all()

    assert written_paths == [result.output_path]


def test_shared_evaluator_can_run_publication_default_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_comparison_module(monkeypatch)
    X, Y = module._load_dataset("pulp")

    evaluation = module.evaluate_pls_family_paths(
        "pulp",
        X,
        Y,
        response_subspaces=("cross_covariance",),
    )

    assert tuple(evaluation.pipls_searches) == ("cross_covariance",)
    search = evaluation.pipls_searches["cross_covariance"]
    assert search.cv is evaluation.cv_splits
    assert search.n_splits_ == evaluation.pls_path.n_splits == 5
    assert search.estimator.response_subspace == "cross_covariance"
    assert search.search_method == "exhaustive"
    np.testing.assert_array_equal(
        search.component_path_.n_components,
        evaluation.pls_path.n_components,
    )
    assert np.isfinite(search.component_path_.cv_mse_mean).all()
    assert np.isfinite(evaluation.pls_path.cv_mse_mean).all()


def test_synthetic_stress_case_is_fixed_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_comparison_module(monkeypatch)
    spec = module.SYNTHETIC_STRESS_SPEC

    assert spec.n_samples == 25
    assert spec.n_features == 40
    assert spec.n_targets == 10
    assert spec.n_shared == 5
    assert spec.n_predictor_specific == 15
    assert spec.n_response_specific == 0
    assert spec.noise == 0.3
    assert spec.random_state == 0

    first = module._make_synthetic_stress_case()
    second = module._make_synthetic_stress_case()

    assert first.X.shape == (25, 40)
    assert first.Y.shape == (25, 10)
    assert first.metadata["latent_dimensions"]["shared"] == 5
    assert first.metadata["latent_dimensions"]["predictor_specific"] == 15
    assert first.metadata["latent_dimensions"]["response_specific"] == 0
    assert first.metadata["noise"]["X"] == 0.3
    assert first.metadata["noise"]["Y"] == 0.3
    assert first.metadata["random_state"] == 0
    assert first.truth is not None
    assert first.truth.n_shared == 5
    assert first.truth.n_predictor_specific == 15
    assert first.truth.n_response_specific == 0

    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.Y, second.Y)
    np.testing.assert_array_equal(first.truth.x_signal, second.truth.x_signal)
    np.testing.assert_array_equal(first.truth.y_signal, second.truth.y_signal)


def test_synthetic_stress_comparison_uses_exhaustive_matched_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written_paths: list[Path] = []

    def _capture_savefig(
        self: Figure,
        path: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        del self, args, kwargs
        written_paths.append(Path(path))

    monkeypatch.setattr(Figure, "savefig", _capture_savefig)
    module = _load_comparison_module(monkeypatch)
    result = module.run_dataset(module.SYNTHETIC_STRESS_CASE)

    cross_search = result.cross_covariance_search
    least_squares_search = result.least_squares_search
    splits = result.cv_splits

    assert cross_search.cv is least_squares_search.cv is splits
    assert cross_search.search_method == least_squares_search.search_method == "exhaustive"
    assert cross_search.search_is_exhaustive_
    assert least_squares_search.search_is_exhaustive_
    assert cross_search.n_splits_ == least_squares_search.n_splits_ == 5
    assert result.pls_path.n_splits == 5

    cross_path = cross_search.component_path_
    least_squares_path = least_squares_search.component_path_
    expected_components = np.arange(1, 11, dtype=np.intp)
    np.testing.assert_array_equal(cross_path.n_components, expected_components)
    np.testing.assert_array_equal(least_squares_path.n_components, expected_components)
    np.testing.assert_array_equal(result.pls_path.n_components, expected_components)
    assert np.isfinite(cross_path.cv_mse_mean).all()
    assert np.isfinite(least_squares_path.cv_mse_mean).all()
    assert np.isfinite(result.pls_path.cv_mse_mean).all()

    assert written_paths == [result.output_path]


def test_dataset_search_templates_keep_matched_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_comparison_module(monkeypatch)
    from _support.pls_family_path_comparison import make_pipls_search

    splits = [
        (np.array([0, 1], dtype=np.intp), np.array([2], dtype=np.intp)),
        (np.array([1, 2], dtype=np.intp), np.array([0], dtype=np.intp)),
    ]

    for dataset in module.COMPARISON_CASES:
        cross_search = make_pipls_search(
            dataset,
            response_subspace="cross_covariance",
            cv_splits=splits,
        )
        least_squares_search = make_pipls_search(
            dataset,
            response_subspace="least_squares",
            cv_splits=splits,
        )

        assert cross_search.cv is least_squares_search.cv is splits
        assert cross_search.search_method == least_squares_search.search_method
        assert cross_search.n_jobs == least_squares_search.n_jobs
        assert isinstance(cross_search.estimator, PiPLSRegression)
        assert isinstance(least_squares_search.estimator, PiPLSRegression)
        assert cross_search.estimator.response_subspace == "cross_covariance"
        assert least_squares_search.estimator.response_subspace == "least_squares"
        assert cross_search.estimator.svd_solver == least_squares_search.estimator.svd_solver

        if dataset in ("pulp", module.SYNTHETIC_STRESS_CASE):
            assert cross_search.search_method == "exhaustive"
        else:
            assert cross_search.search_method == "adaptive"
        if dataset == "tobacco":
            assert cross_search.estimator.svd_solver == "full"
            assert cross_search.n_jobs == 1
        else:
            assert cross_search.estimator.svd_solver == "auto"
