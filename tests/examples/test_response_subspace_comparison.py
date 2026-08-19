from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from pipls import PiPLSRegression, PiPLSSearchCV


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_response_subspace_example_uses_one_matched_cv_protocol(
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
    result = runpy.run_path(
        str(_repository_root() / "examples" / "07_response_subspace_comparison.py"),
        run_name="__main__",
    )
    output = capsys.readouterr().out

    splits = result["CV_SPLITS"]
    cross_search = result["cross_covariance_search"]
    least_squares_search = result["least_squares_search"]
    cross_selection = result["cross_covariance_selection"]
    least_squares_selection = result["least_squares_selection"]

    assert isinstance(cross_search, PiPLSSearchCV)
    assert isinstance(least_squares_search, PiPLSSearchCV)
    assert cross_search.cv is splits
    assert least_squares_search.cv is splits
    assert cross_search.n_splits_ == least_squares_search.n_splits_ == 5
    assert isinstance(cross_search.estimator, PiPLSRegression)
    assert isinstance(least_squares_search.estimator, PiPLSRegression)
    assert cross_search.estimator.response_subspace == "cross_covariance"
    assert least_squares_search.estimator.response_subspace == "least_squares"
    np.testing.assert_array_equal(
        cross_search.component_path_.n_components,
        least_squares_search.component_path_.n_components,
    )
    assert cross_selection.n_components == 3
    assert cross_selection.predictor_rank == 9
    assert least_squares_selection.n_components == 3
    assert least_squares_selection.predictor_rank == 9
    assert np.isfinite(cross_selection.cv_mse_mean)
    assert np.isfinite(least_squares_selection.cv_mse_mean)

    assert "Π-PLS response-subspace comparison" in output
    assert "Shared validation protocol: 5 materialized folds" in output
    assert "cross_covariance (peer-reviewed default)" in output
    assert (
        "least_squares (software extension; not part of the peer-reviewed publication)"
        in output
    )
    assert "model-development CV results, not independent post-selection validation" in output
    assert len(written_paths) == 1
    assert written_paths[0].name == "response_subspace_comparison.pdf"
