from sklearn.base import clone

from pipls import PiPLSRegression


def test_estimator_is_cloneable() -> None:
    model = PiPLSRegression(
        n_components=1,
        predictor_rank=2,
        scale=False,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
