from typing import Any

import numpy as np

from pipls import PiPLSRegression
from pipls.metrics import (
    neg_response_standardized_mean_squared_error,
    response_standardized_mean_squared_error,
)


def test_response_standardized_metric_uses_training_response_scale_when_scale_is_false() -> None:
    X = np.arange(24, dtype=np.float64).reshape(8, 3)
    y = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0])
    model = PiPLSRegression(
        n_components=1,
        predictor_rank=1,
        scale=False,
    ).fit(X, y)

    prediction = model.predict(X)
    expected = np.mean(((y - prediction) / np.std(y, ddof=1)) ** 2)

    assert response_standardized_mean_squared_error(model, X, y) == expected
    assert neg_response_standardized_mean_squared_error(model, X, y) == -expected


def test_metric_callable_has_sklearn_scorer_signature() -> None:
    def consume_scorer(scorer: Any) -> None:
        assert callable(scorer)

    consume_scorer(neg_response_standardized_mean_squared_error)
