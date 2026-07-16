import numpy as np
import pytest

from pipls import PiPLSRegression


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    X = rng.normal(size=(40, 8))
    B = rng.normal(size=(8, 3))
    Y = X @ B + 0.05 * rng.normal(size=(40, 3))
    return X, Y


def test_fit_exposes_expected_fixed_rank_attributes() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=2, predictor_rank=4).fit(X, Y)

    assert model.predictor_rank_ == 4
    assert model.Pi_.shape == (8, 4)
    assert model.C_.shape == (3, 2)
    assert model.W_.shape == (4, 2)
    assert model.P_.shape == (8, 2)
    assert model.D_.shape == (2, 2)
    assert model.Q_.shape == (3, 2)
    assert model.coef_.shape == (3, 8)
    assert model.coef_matrix_.shape == (8, 3)
    assert model.intercept_.shape == (3,)
    assert model.x_scores_.shape == (40, 2)
    assert model.y_scores_.shape == (40, 2)


def test_prediction_matches_affine_attributes() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=2, predictor_rank=4).fit(X, Y)

    np.testing.assert_allclose(model.predict(X), X @ model.coef_.T + model.intercept_)


def test_transform_uses_fitted_centering_and_scaling() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=2, predictor_rank=4).fit(X, Y)

    x_scores, y_scores = model.transform(X, Y)
    np.testing.assert_allclose(x_scores, model.x_scores_)
    np.testing.assert_allclose(y_scores, model.y_scores_)


def test_scale_false_still_centers() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=2,
        predictor_rank=4,
        scale=False,
    ).fit(X, Y)

    np.testing.assert_allclose(model.x_scale_, 1.0)
    np.testing.assert_allclose(model.y_scale_, 1.0)
    np.testing.assert_allclose(np.mean(model.x_scores_, axis=0), 0.0, atol=1e-12)


def test_one_dimensional_response_round_trips_as_one_dimensional() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=1, predictor_rank=3).fit(X, Y[:, 0])

    assert model.predict(X).shape == (X.shape[0],)
    assert isinstance(model.score(X, Y[:, 0]), float)


def test_invalid_constructor_combination_is_rejected() -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="n_components <= predictor_rank"):
        PiPLSRegression(n_components=3, predictor_rank=2).fit(X, Y)


def test_wrong_feature_count_is_rejected_at_prediction() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=2, predictor_rank=4).fit(X, Y)

    with pytest.raises(ValueError, match="incompatible number of features"):
        model.predict(X[:, :-1])


def test_constant_columns_have_unit_scale() -> None:
    X, Y = _data()
    X[:, 0] = 3.0
    Y[:, 0] = -2.0
    model = PiPLSRegression(n_components=1, predictor_rank=3).fit(X, Y)

    assert model.x_scale_[0] == 1.0
    assert model.y_scale_[0] == 1.0


def test_max_rank_mode_uses_rule_derived_bound() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=2,
        predictor_rank="max",
        samples_per_predictor_rank=10,
    ).fit(X, Y)

    assert model.max_predictor_rank_ == 4
    assert model.predictor_rank_ == 4
    assert model.Pi_.shape == (X.shape[1], 4)


def test_samples_per_predictor_rank_changes_max_mode() -> None:
    X, Y = _data()
    lower_rank = PiPLSRegression(
        n_components=2,
        predictor_rank="max",
        samples_per_predictor_rank=20,
    ).fit(X, Y)
    higher_rank = PiPLSRegression(
        n_components=2,
        predictor_rank="max",
        samples_per_predictor_rank=8,
    ).fit(X, Y)

    assert lower_rank.predictor_rank_ == 2
    assert higher_rank.predictor_rank_ == 5


def test_explicit_rank_bypasses_rule_derived_bound() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=2,
        predictor_rank=4,
        samples_per_predictor_rank=100,
    ).fit(X, Y)

    assert model.max_predictor_rank_ == 1
    assert model.predictor_rank_ == 4


@pytest.mark.parametrize("value", [0, -1, np.inf, True])
def test_invalid_samples_per_predictor_rank_is_rejected(value: object) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="samples_per_predictor_rank"):
        PiPLSRegression(
            n_components=2,
            predictor_rank="max",
            samples_per_predictor_rank=value,  # type: ignore[arg-type]
        ).fit(X, Y)


def test_unimplemented_auto_rank_mode_is_rejected() -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match='positive integer or "max"'):
        PiPLSRegression(
            n_components=2,
            predictor_rank="auto",  # type: ignore[arg-type]
        ).fit(X, Y)
