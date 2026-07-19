from __future__ import annotations

import numpy as np

from pipls import PiPLSRegression
from pipls.inspection import pipls_display_factors


def test_display_sign_canonicalization_preserves_fitted_regression_map() -> None:
    rng = np.random.default_rng(1204)
    X = rng.normal(size=(24, 6))
    Y = rng.normal(size=(24, 3))
    model = PiPLSRegression(n_components=2, predictor_rank=3).fit(X, Y)
    decomposition = model.decomposition_
    predictor_before = decomposition.P.copy()
    response_before = decomposition.Q.copy()

    factors = pipls_display_factors(decomposition)
    display_map = (
        factors.predictor_directions
        @ np.diag(factors.dilation)
        @ factors.response_directions.T
    )

    np.testing.assert_allclose(display_map, decomposition.regression_map, atol=1e-14)
    np.testing.assert_allclose(
        factors.predictor_directions @ factors.weighted_response_directions.T,
        decomposition.regression_map,
        atol=1e-14,
    )
    np.testing.assert_array_equal(decomposition.P, predictor_before)
    np.testing.assert_array_equal(decomposition.Q, response_before)
    assert model.x_rotations_ is decomposition.P
    assert model.y_rotations_ is decomposition.Q
