from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from pipls._core import fit_pipls_core

_EXPECTED_DILATION = np.array([0.7318678674635507, 0.2860934054613716])
_EXPECTED_REGRESSION_MAP = np.array(
    [
        [0.0227515116210289, 0.0599525243839417, 0.0124157984852828],
        [-0.1621045910529089, -0.0923178863211663, -0.1302212944827721],
        [-0.5049202236976003, 0.0676477362676501, -0.4499077660476559],
        [-0.1229550816438217, 0.0125401683676110, -0.1090682976181716],
        [-0.0805912433251241, -0.2536619989695337, -0.0388296444243356],
    ]
)


def test_core_matches_frozen_trusted_reference() -> None:
    rng = np.random.default_rng(20260715)
    X = rng.normal(size=(8, 5))
    Y = rng.normal(size=(8, 3))
    X -= X.mean(axis=0, keepdims=True)
    Y -= Y.mean(axis=0, keepdims=True)

    result = fit_pipls_core(X, Y, predictor_rank=3, n_components=2)

    assert_allclose(np.diag(result.D), _EXPECTED_DILATION, rtol=2e-13, atol=2e-13)
    assert_allclose(
        result.standardized_regression_map,
        _EXPECTED_REGRESSION_MAP,
        rtol=2e-13,
        atol=2e-13,
    )
