from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import NotFittedError

from pipls.inspection import (
    PLSBiplotCoordinates,
    PLSLatentStructure,
    PLSObservationDiagnostics,
    pls_biplot_coordinates,
    pls_latent_structure,
    pls_observation_diagnostics,
)


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


def _fitted_pls_with_data() -> tuple[PLSRegression, np.ndarray]:
    rng = np.random.default_rng(441)
    X = rng.normal(size=(36, 7))
    Y = X[:, :3] @ np.array(
        [
            [1.0, -0.5],
            [0.4, 0.8],
            [-0.2, 0.3],
        ]
    ) + 0.1 * rng.normal(size=(36, 2))
    return PLSRegression(n_components=3, scale=True).fit(X, Y), X


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


def test_pls_observation_diagnostics_match_explicit_public_method_calculation() -> None:
    model, X = _fitted_pls_with_data()

    diagnostics = pls_observation_diagnostics(model, X)

    training_scores = model.x_scores_
    center = np.mean(training_scores, axis=0)
    centered_training = training_scores - center
    covariance = centered_training.T @ centered_training / (len(training_scores) - 1)
    scores = model.transform(X)
    centered_scores = scores - center
    expected_distance = np.einsum(
        "ij,jk,ik->i",
        centered_scores,
        np.linalg.pinv(covariance),
        centered_scores,
    )
    reconstructed = model.inverse_transform(scores)
    expected_residual = np.sum((X - reconstructed) ** 2, axis=1)

    assert isinstance(diagnostics, PLSObservationDiagnostics)
    np.testing.assert_allclose(diagnostics.score_distance, expected_distance)
    np.testing.assert_allclose(
        diagnostics.x_reconstruction_residual,
        expected_residual,
    )
    assert diagnostics.n_samples == len(X)
    assert not diagnostics.score_distance.flags.writeable
    assert not diagnostics.x_reconstruction_residual.flags.writeable


def test_pls_observation_diagnostics_accept_external_observations() -> None:
    model, X = _fitted_pls_with_data()

    diagnostics = pls_observation_diagnostics(model, X[:5])

    assert diagnostics.score_distance.shape == (5,)
    assert diagnostics.x_reconstruction_residual.shape == (5,)


def test_pls_observation_diagnostics_rejects_wrong_feature_count() -> None:
    model, X = _fitted_pls_with_data()

    with pytest.raises(ValueError, match="fitted number of predictor columns"):
        pls_observation_diagnostics(model, X[:, :-1])


def test_pls_observation_diagnostics_rejects_unfitted_estimator() -> None:
    with pytest.raises(NotFittedError):
        pls_observation_diagnostics(PLSRegression(n_components=2), np.ones((4, 3)))


def test_pls_biplot_coordinates_preserve_selected_reconstruction_and_balance_norms() -> None:
    model = _fitted_pls()
    structure = pls_latent_structure(model)
    scores_before = structure.x_scores.copy()
    loadings_before = structure.x_loadings.copy()

    coordinates = pls_biplot_coordinates(structure, components=(0, 2))

    assert isinstance(coordinates, PLSBiplotCoordinates)
    expected = structure.x_scores[:, [0, 2]] @ structure.x_loadings[:, [0, 2]].T
    reconstructed = coordinates.sample_coordinates @ coordinates.predictor_coordinates.T
    np.testing.assert_allclose(reconstructed, expected)
    np.testing.assert_allclose(
        np.linalg.norm(coordinates.sample_coordinates, axis=0),
        np.linalg.norm(coordinates.predictor_coordinates, axis=0),
    )
    np.testing.assert_array_equal(coordinates.component_indices, np.array([0, 2]))
    assert coordinates.n_samples == structure.n_samples
    assert coordinates.n_features == structure.n_features
    for values in (
        coordinates.sample_coordinates,
        coordinates.predictor_coordinates,
        coordinates.component_indices,
        coordinates.scaling_factors,
    ):
        assert not values.flags.writeable
    np.testing.assert_array_equal(structure.x_scores, scores_before)
    np.testing.assert_array_equal(structure.x_loadings, loadings_before)


@pytest.mark.parametrize(
    ("components", "message"),
    [
        ((0,), "exactly two"),
        ((0, 0), "distinct"),
        ((0, 3), "must lie"),
    ],
)
def test_pls_biplot_coordinates_reject_invalid_component_selections(
    components: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pls_biplot_coordinates(_structure_for_biplot(), components=components)


def test_pls_biplot_coordinates_reject_zero_norm_loadings() -> None:
    structure = _structure_for_biplot()
    x_loadings = structure.x_loadings.copy()
    x_loadings[:, 1] = 0.0
    invalid = PLSLatentStructure(
        x_scores=structure.x_scores,
        x_loadings=x_loadings,
        y_loadings=structure.y_loadings,
        coefficients=structure.coefficients,
    )

    with pytest.raises(ValueError, match="X-loading columns"):
        pls_biplot_coordinates(invalid, components=(0, 1))


def _structure_for_biplot() -> PLSLatentStructure:
    return pls_latent_structure(_fitted_pls())
