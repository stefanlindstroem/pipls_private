from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pipls._core import fit_pipls_core


def _center(array: np.ndarray) -> np.ndarray:
    return array - array.mean(axis=0, keepdims=True)


def test_core_shapes_and_prediction_method() -> None:
    rng = np.random.default_rng(12)
    X = _center(rng.normal(size=(15, 7)))
    Y = _center(rng.normal(size=(15, 4)))

    result = fit_pipls_core(X, Y, predictor_rank=5, n_components=3)

    assert result.Pi.shape == (7, 5)
    assert result.C.shape == (4, 3)
    assert result.W.shape == (5, 3)
    assert result.P.shape == (7, 3)
    assert result.D.shape == (3, 3)
    assert result.Q.shape == (4, 3)
    assert result.regression_map.shape == (7, 4)
    assert_allclose(result.predict(X), X @ result.regression_map)


def test_core_rejects_inadmissible_inputs() -> None:
    X = np.eye(4)
    Y = np.ones((4, 2))

    with pytest.raises(ValueError, match="same number of samples"):
        fit_pipls_core(X, Y[:3], predictor_rank=2, n_components=1)
    with pytest.raises(ValueError, match="positive integer"):
        fit_pipls_core(X, Y, predictor_rank=1.5, n_components=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"min\(predictor_rank, n_targets\)"):
        fit_pipls_core(X, Y, predictor_rank=2, n_components=3)
    with pytest.raises(ValueError, match="finite"):
        fit_pipls_core(np.array([[np.nan]]), np.ones((1, 1)), predictor_rank=1, n_components=1)


def test_core_handles_rank_deficiency_explicitly() -> None:
    base = np.arange(1.0, 9.0).reshape(4, 2)
    X = np.column_stack([base, base[:, 0] + base[:, 1]])
    Y = np.column_stack([base[:, 0], base[:, 1]])

    result = fit_pipls_core(X, Y, predictor_rank=2, n_components=2)
    assert result.x_rank == 2

    with pytest.raises(ValueError, match="numerical rank"):
        fit_pipls_core(X, Y, predictor_rank=3, n_components=2)
