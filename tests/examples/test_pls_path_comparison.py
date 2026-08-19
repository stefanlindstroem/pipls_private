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
    capsys: pytest.CaptureFixture[str],
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
    result = module.run_dataset("pulp")
    output = capsys.readouterr().out

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

    assert "shared validation protocol: 5 materialized folds" in output
    assert "cross_covariance (peer-reviewed default) predictor ranks" in output
    assert (
        "least_squares (software extension; not part of the peer-reviewed publication) "
        "predictor ranks"
        in output
    )
    assert "model-development evidence, not independent post-selection validation" in output
    assert written_paths == [result.output_path]
    assert result.output_path.name == "pulp_component_path_comparison.pdf"


def test_dataset_search_templates_keep_matched_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_comparison_module(monkeypatch)
    splits = [
        (np.array([0, 1], dtype=np.intp), np.array([2], dtype=np.intp)),
        (np.array([1, 2], dtype=np.intp), np.array([0], dtype=np.intp)),
    ]

    for dataset in module.DATASET_NAMES:
        cross_search = module._make_search(
            dataset,
            response_subspace="cross_covariance",
            cv_splits=splits,
        )
        least_squares_search = module._make_search(
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

        if dataset == "pulp":
            assert cross_search.search_method == "exhaustive"
        else:
            assert cross_search.search_method == "adaptive"
        if dataset == "tobacco":
            assert cross_search.estimator.svd_solver == "full"
            assert cross_search.n_jobs == 1
        else:
            assert cross_search.estimator.svd_solver == "auto"
