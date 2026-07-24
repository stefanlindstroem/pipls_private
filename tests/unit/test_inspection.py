from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import pipls
import pipls.inspection as inspection
from pipls import PiPLSDecomposition
from pipls.inspection import pipls_display_factors, prediction_diagnostics


def _decomposition(
    predictor_directions: np.ndarray,
    response_directions: np.ndarray,
    dilation: np.ndarray,
) -> PiPLSDecomposition:
    _, n_components = predictor_directions.shape
    return PiPLSDecomposition(
        predictor_rotations=predictor_directions,
        dilation=dilation,
        response_rotations=response_directions,
        predictor_numerical_rank=n_components,
        predictor_numerical_rank_is_exact=True,
        rank_tolerance=1e-12,
        predictor_svd_solver="full",
    )


def test_inspection_names_are_submodule_exports_only() -> None:
    expected = {
        "BiplotCoordinates",
        "LatentStructure",
        "ObservationDiagnostics",
        "PiPLSDisplayFactors",
        "PredictionDiagnostics",
        "PredictionKind",
        "pipls_display_factors",
        "biplot_coordinates",
        "latent_structure",
        "observation_diagnostics",
        "prediction_diagnostics",
    }

    assert set(inspection.__all__) == expected
    assert expected.isdisjoint(pipls.__all__)


def test_pipls_display_factors_use_first_largest_predictor_entry_for_sign() -> None:
    predictor_directions = np.array(
        [
            [-2.0, 0.0, 0.0],
            [2.0, -3.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    response_directions = np.array(
        [
            [0.5, -0.25, 1.0],
            [-1.5, 0.75, 0.0],
        ]
    )
    dilation = np.array([3.0, 2.0, 0.0])
    decomposition = _decomposition(predictor_directions, response_directions, dilation)

    factors = pipls_display_factors(decomposition)

    np.testing.assert_array_equal(
        factors.predictor_directions,
        predictor_directions * np.array([-1.0, -1.0, 1.0]),
    )
    np.testing.assert_array_equal(
        factors.response_directions,
        response_directions * np.array([-1.0, -1.0, 1.0]),
    )
    np.testing.assert_array_equal(
        factors.weighted_response_directions,
        factors.response_directions * dilation,
    )
    assert factors.n_features == 3
    assert factors.n_targets == 2
    assert factors.n_components == 3
    assert not hasattr(factors, "component_signs")


def test_pipls_display_factors_are_defensive_read_only_copies() -> None:
    predictor_directions = np.eye(2)
    response_directions = np.eye(2)
    dilation = np.array([2.0, 1.0])
    decomposition = _decomposition(predictor_directions, response_directions, dilation)

    factors = pipls_display_factors(decomposition)

    for values in (
        factors.predictor_directions,
        factors.dilation,
        factors.response_directions,
        factors.weighted_response_directions,
    ):
        assert not values.flags.writeable
    assert not np.shares_memory(factors.predictor_directions, decomposition.predictor_rotations)
    assert not np.shares_memory(factors.response_directions, decomposition.response_rotations)
    assert not np.shares_memory(factors.dilation, decomposition.dilation)

    predictor_directions[0, 0] = 7.0
    response_directions[0, 0] = 8.0
    dilation[0] = 9.0
    assert factors.predictor_directions[0, 0] == 1.0
    assert factors.response_directions[0, 0] == 1.0
    assert factors.dilation[0] == 2.0

    with pytest.raises(ValueError, match="read-only"):
        factors.predictor_directions[0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        factors.dilation = np.ones(2)  # type: ignore[misc]


def test_public_decomposition_rejects_negative_dilation() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        _decomposition(np.eye(2), np.eye(2), np.array([2.0, -1.0]))


def test_pipls_display_factors_require_public_decomposition() -> None:
    with pytest.raises(TypeError, match="PiPLSDecomposition"):
        pipls_display_factors(object())  # type: ignore[arg-type]


def test_prediction_diagnostics_standardize_from_observed_responses() -> None:
    observed = np.array(
        [
            [1.0, 10.0],
            [3.0, 14.0],
            [5.0, 18.0],
        ]
    )
    predicted = np.array(
        [
            [0.0, 12.0],
            [4.0, 10.0],
            [5.0, 20.0],
        ]
    )

    diagnostics = prediction_diagnostics(
        observed,
        predicted,
        prediction_kind="external test predictions",
    )

    expected_center = np.array([3.0, 14.0])
    expected_scale = np.array([2.0, 4.0])
    expected_residual = observed - predicted
    np.testing.assert_allclose(diagnostics.response_centers, expected_center)
    np.testing.assert_allclose(diagnostics.response_scales, expected_scale)
    np.testing.assert_allclose(diagnostics.residual, expected_residual)
    np.testing.assert_allclose(
        diagnostics.observed_standardized,
        (observed - expected_center) / expected_scale,
    )
    np.testing.assert_allclose(
        diagnostics.predicted_standardized,
        (predicted - expected_center) / expected_scale,
    )
    np.testing.assert_allclose(
        diagnostics.residual_standardized,
        expected_residual / expected_scale,
    )
    np.testing.assert_allclose(
        diagnostics.standardized_rmse,
        np.sqrt(np.mean(np.square(expected_residual / expected_scale), axis=0)),
    )
    assert diagnostics.prediction_kind == "external test predictions"
    assert diagnostics.n_samples == 3
    assert diagnostics.n_targets == 2


def test_prediction_diagnostics_normalize_vector_inputs_to_two_dimensions() -> None:
    diagnostics = prediction_diagnostics(
        np.array([1.0, 3.0, 5.0]),
        np.array([[0.0], [4.0], [5.0]]),
        prediction_kind="fitted values",
    )

    assert diagnostics.observed.shape == (3, 1)
    assert diagnostics.predicted.shape == (3, 1)
    assert diagnostics.response_centers.shape == (1,)
    assert diagnostics.standardized_rmse.shape == (1,)


def test_prediction_diagnostics_are_defensive_read_only_copies() -> None:
    observed = np.array([1.0, 2.0, 4.0])
    predicted = np.array([1.5, 2.5, 3.5])

    diagnostics = prediction_diagnostics(
        observed,
        predicted,
        prediction_kind="fixed-parameter OOF predictions",
    )

    for values in (
        diagnostics.observed,
        diagnostics.predicted,
        diagnostics.residual,
        diagnostics.observed_standardized,
        diagnostics.predicted_standardized,
        diagnostics.residual_standardized,
        diagnostics.response_centers,
        diagnostics.response_scales,
        diagnostics.standardized_rmse,
    ):
        assert not values.flags.writeable

    observed[0] = 100.0
    predicted[0] = 100.0
    assert diagnostics.observed[0, 0] == 1.0
    assert diagnostics.predicted[0, 0] == 1.5

    with pytest.raises(ValueError, match="read-only"):
        diagnostics.residual[0, 0] = 0.0


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        (np.ones((3, 2)), np.ones((3, 1)), "identical shapes"),
        (np.ones((1, 1)), np.ones((1, 1)), "at least two observations"),
        (np.ones((3, 1)), np.zeros((3, 1)), "constant response columns"),
        (np.array([1.0, np.nan]), np.array([1.0, 2.0]), "finite values"),
        (np.ones((2, 1, 1)), np.ones((2, 1, 1)), "one- or two-dimensional"),
    ],
)
def test_prediction_diagnostics_reject_invalid_response_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prediction_diagnostics(y_true, y_pred, prediction_kind="fitted values")


def test_prediction_diagnostics_reject_unknown_prediction_kind() -> None:
    with pytest.raises(ValueError, match="prediction_kind must be one of"):
        prediction_diagnostics(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            prediction_kind="cross-validated predictions",  # type: ignore[arg-type]
        )
