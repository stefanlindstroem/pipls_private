from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import NotFittedError

from pipls.inspection import PLSLatentStructure, pls_latent_structure


def _fitted_pls() -> PLSRegression:
    rng = np.random.default_rng(441)
    X = rng.normal(size=(36, 7))
    Y = X[:, :3] @ np.array(
        [
            [1.0, -0.5],
            [0.4, 0.8],
            [-0.2, 0.3],
        ]
    ) + 0.1 * rng.normal(size=(36, 2))
    return PLSRegression(n_components=3, scale=True).fit(X, Y)


def test_pls_latent_structure_copies_public_fitted_arrays() -> None:
    model = _fitted_pls()

    structure = pls_latent_structure(model)

    assert isinstance(structure, PLSLatentStructure)
    np.testing.assert_array_equal(structure.x_scores, model.x_scores_)
    np.testing.assert_array_equal(structure.x_loadings, model.x_loadings_)
    np.testing.assert_array_equal(structure.y_loadings, model.y_loadings_)
    np.testing.assert_array_equal(structure.coefficients, model.coef_)
    assert structure.n_samples == 36
    assert structure.n_features == 7
    assert structure.n_targets == 2
    assert structure.n_components == 3


def test_pls_latent_structure_returns_defensive_read_only_arrays() -> None:
    model = _fitted_pls()
    structure = pls_latent_structure(model)
    original_score = float(structure.x_scores[0, 0])

    for values in (
        structure.x_scores,
        structure.x_loadings,
        structure.y_loadings,
        structure.coefficients,
    ):
        assert not values.flags.writeable

    model.x_scores_[0, 0] = original_score + 10.0
    assert structure.x_scores[0, 0] == original_score

    with pytest.raises(ValueError, match="read-only"):
        structure.x_loadings[0, 0] = 0.0


def test_pls_latent_structure_rejects_an_unfitted_estimator() -> None:
    with pytest.raises(NotFittedError):
        pls_latent_structure(PLSRegression(n_components=2))


def test_pls_latent_structure_rejects_another_estimator_type() -> None:
    with pytest.raises(TypeError, match="PLSRegression"):
        pls_latent_structure(object())  # type: ignore[arg-type]


def test_pls_latent_structure_rejects_inconsistent_coefficient_shape() -> None:
    model = _fitted_pls()
    model.coef_ = np.ones((model.y_loadings_.shape[0], model.x_loadings_.shape[0] - 1))

    with pytest.raises(ValueError, match=r"shape \(n_targets, n_features\)"):
        pls_latent_structure(model)


def test_pls_latent_structure_supports_one_response() -> None:
    rng = np.random.default_rng(817)
    X = rng.normal(size=(24, 5))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.normal(size=24)
    model = PLSRegression(n_components=2).fit(X, y)

    structure = pls_latent_structure(model)

    assert structure.y_loadings.shape == (1, 2)
    assert structure.coefficients.shape == (1, 5)
    assert structure.n_targets == 1


def test_pls_latent_structure_rejects_nonfinite_fitted_arrays() -> None:
    model = _fitted_pls()
    model.x_scores_[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite values"):
        pls_latent_structure(model)
