from __future__ import annotations

import pickle
from collections.abc import Callable

import numpy as np
import pytest

from pipls.inspection import (
    BiplotCoordinates,
    LatentStructure,
    ObservationDiagnostics,
    PiPLSDisplayFactors,
    PredictionDiagnostics,
)


def _prediction_diagnostics() -> PredictionDiagnostics:
    observed = np.array([[1.0], [3.0], [5.0]], dtype=np.float32)
    predicted = np.array([[0.0], [4.0], [5.0]], dtype=np.float32)
    centers = np.array([3.0], dtype=np.float32)
    scales = np.array([2.0], dtype=np.float32)
    residual = observed - predicted
    return PredictionDiagnostics(
        observed=observed,
        predicted=predicted,
        residual=residual,
        observed_standardized=(observed - centers) / scales,
        predicted_standardized=(predicted - centers) / scales,
        residual_standardized=residual / scales,
        response_centers=centers,
        response_scales=scales,
        standardized_rmse=np.sqrt(np.mean(np.square(residual / scales), axis=0)),
        prediction_kind="external test predictions",
    )


def test_inspection_records_copy_arrays_and_revalidate_pickle() -> None:
    x_scores = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    x_loadings = np.eye(2, dtype=np.float32)
    y_loadings = np.eye(2, dtype=np.float32)
    coefficients = np.eye(2, dtype=np.float32)
    structure = LatentStructure(x_scores, x_loadings, y_loadings, coefficients)
    x_scores[0, 0] = 9.0

    coordinates = BiplotCoordinates(
        sample_coordinates=np.eye(2, dtype=np.float32),
        predictor_coordinates=np.eye(2, dtype=np.float32),
        component_indices=np.array([0, 1], dtype=np.int64),
        scaling_factors=np.ones(2, dtype=np.float32),
    )
    observations = ObservationDiagnostics(
        score_distance=np.array([1.0, 2.0], dtype=np.float32),
        x_reconstruction_residual=np.array([0.1, 0.2], dtype=np.float32),
    )
    factors = PiPLSDisplayFactors(
        predictor_directions=np.eye(2, dtype=np.float32),
        dilation=np.array([2.0, 1.0], dtype=np.float32),
        response_directions=np.eye(2, dtype=np.float32),
    )
    predictions = _prediction_diagnostics()

    assert structure.x_scores[0, 0] == 1.0
    for record in (structure, coordinates, observations, factors, predictions):
        restored = pickle.loads(pickle.dumps(record))
        for value in vars(restored).values():
            if isinstance(value, np.ndarray):
                assert value.dtype in (np.dtype(np.float64), np.dtype(np.intp))
                assert not value.flags.writeable


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (
            lambda: LatentStructure(
                np.ones((2, 2)),
                np.ones((3, 1)),
                np.ones((1, 2)),
                np.ones((1, 3)),
            ),
            "same number of components",
        ),
        (
            lambda: BiplotCoordinates(
                np.ones((2, 3)),
                np.ones((3, 2)),
                np.array([0, 1]),
                np.ones(2),
            ),
            "exactly two columns",
        ),
        (
            lambda: ObservationDiagnostics(np.array([1.0, -1.0]), np.ones(2)),
            "nonnegative",
        ),
    ],
)
def test_inspection_records_reject_invalid_direct_construction(
    constructor: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        constructor()


def test_display_factors_derive_weighted_response_directions() -> None:
    factors = PiPLSDisplayFactors(
        predictor_directions=np.eye(2, dtype=np.float32),
        dilation=np.array([2.0, 1.0], dtype=np.float32),
        response_directions=np.eye(2, dtype=np.float32),
    )

    weighted = factors.weighted_response_directions
    np.testing.assert_array_equal(weighted, np.diag([2.0, 1.0]))
    assert not weighted.flags.writeable
    assert "weighted_response_directions" not in vars(factors)


def test_display_factors_reject_unrepresentable_derived_weighting() -> None:
    with pytest.raises(ValueError, match="weighted_response_directions cannot be represented"):
        PiPLSDisplayFactors(
            predictor_directions=np.eye(2),
            dilation=np.array([2.0, 1.0]),
            response_directions=np.array(
                [[np.finfo(np.float64).max, 0.0], [0.0, 1.0]]
            ),
        )


def test_prediction_diagnostics_reject_inconsistent_direct_construction() -> None:
    diagnostics = _prediction_diagnostics()
    with pytest.raises(ValueError, match="residual is inconsistent"):
        PredictionDiagnostics(
            observed=diagnostics.observed,
            predicted=diagnostics.predicted,
            residual=np.zeros_like(diagnostics.residual),
            observed_standardized=diagnostics.observed_standardized,
            predicted_standardized=diagnostics.predicted_standardized,
            residual_standardized=diagnostics.residual_standardized,
            response_centers=diagnostics.response_centers,
            response_scales=diagnostics.response_scales,
            standardized_rmse=diagnostics.standardized_rmse,
            prediction_kind=diagnostics.prediction_kind,
        )
