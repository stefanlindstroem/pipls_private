import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipls import PiPLSRegression


def test_fixed_rank_estimator_runs_in_pipeline() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 6))
    Y = rng.normal(size=(30, 2))
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "regression",
                PiPLSRegression(
                    n_components=2,
                    predictor_rank=3,
                    scale=False,
                ),
            ),
        ]
    )

    prediction = pipeline.fit(X, Y).predict(X[:4])
    assert prediction.shape == (4, 2)
