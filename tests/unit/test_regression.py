import warnings

import numpy as np
import pytest

from pipls import PiPLSRegression, StatisticalSupportWarning


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
    assert model.max_predictor_rank_ == min(X.shape[1], X.shape[0] - 1)
    assert model.decomposition_.predictor_rotations.shape == (8, 2)
    assert model.decomposition_.dilation.shape == (2,)
    assert model.decomposition_.response_rotations.shape == (3, 2)
    assert model.coef_.shape == (3, 8)
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


def test_rank_above_centered_matrix_limit_is_rejected() -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="predictor_rank <= min"):
        PiPLSRegression(n_components=1, predictor_rank=X.shape[1] + 1).fit(X, Y)


def test_wrong_feature_count_is_rejected_at_prediction() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=2, predictor_rank=4).fit(X, Y)

    with pytest.raises(ValueError, match="expecting 8 features"):
        model.predict(X[:, :-1])


def test_constant_columns_have_unit_scale() -> None:
    X, Y = _data()
    X[:, 0] = 3.0
    Y[:, 0] = -2.0
    model = PiPLSRegression(n_components=1, predictor_rank=3).fit(X, Y)

    assert model.x_scale_[0] == 1.0
    assert model.y_scale_[0] == 1.0


def test_fixed_rank_pair_is_required_and_keyword_only() -> None:
    with pytest.raises(TypeError):
        PiPLSRegression()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        PiPLSRegression(n_components=1)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        PiPLSRegression(predictor_rank=1)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        PiPLSRegression(1, predictor_rank=1)  # type: ignore[call-arg]

    model = PiPLSRegression(n_components=1, predictor_rank=1)
    assert set(model.get_params()) == {
        "copy",
        "n_components",
        "predictor_rank",
        "random_state",
        "scale",
        "svd_solver",
    }


def test_fit_creates_no_selection_attributes() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=2, predictor_rank=4).fit(X, Y)

    for name in (
        "best_params_",
        "best_score_",
        "cv_results_",
        "validation_report_",
        "oof_predictions_",
        "predictor_rank_values_",
        "predictor_rank_search_history_",
    ):
        assert not hasattr(model, name)


def test_statistical_support_warning_uses_three_samples_per_rank_boundary() -> None:
    X, Y = _data()
    rng = np.random.default_rng(99)
    X_wide = np.column_stack([X, rng.normal(size=(X.shape[0], 6))])

    with warnings.catch_warnings():
        warnings.simplefilter("error", StatisticalSupportWarning)
        PiPLSRegression(n_components=2, predictor_rank=13).fit(X_wide, Y)

    with pytest.warns(StatisticalSupportWarning, match="recommended minimum of 3"):
        PiPLSRegression(n_components=2, predictor_rank=14).fit(X_wide, Y)

def test_small_auto_svd_uses_full_solver_and_reports_exact_rank() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=2,
        predictor_rank=3,
        svd_solver="auto",
    ).fit(X, Y)

    assert model.decomposition_.predictor_svd_solver == "full"
    assert model.decomposition_.predictor_numerical_rank_is_exact


def test_randomized_svd_estimator_is_reproducible_and_close_to_full() -> None:
    rng = np.random.default_rng(817)
    left, _ = np.linalg.qr(rng.normal(size=(100, 14)))
    right, _ = np.linalg.qr(rng.normal(size=(70, 14)))
    values = np.array(
        [35.0, 28.0, 21.0, 16.0, 12.0, 9.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1]
    )
    X = left @ np.diag(values) @ right.T
    Y = X @ rng.normal(size=(70, 3)) + 0.01 * rng.normal(size=(100, 3))

    full = PiPLSRegression(
        n_components=2,
        predictor_rank=6,
        svd_solver="full",
        random_state=None,
    ).fit(X, Y)
    first = PiPLSRegression(
        n_components=2,
        predictor_rank=6,
        svd_solver="randomized",
        random_state=23,
    ).fit(X, Y)
    second = PiPLSRegression(
        n_components=2,
        predictor_rank=6,
        svd_solver="randomized",
        random_state=23,
    ).fit(X, Y)

    assert first.decomposition_.predictor_svd_solver == "randomized"
    assert not first.decomposition_.predictor_numerical_rank_is_exact
    np.testing.assert_allclose(first.coef_, second.coef_)
    np.testing.assert_allclose(first.predict(X), second.predict(X))
    np.testing.assert_allclose(first.predict(X), full.predict(X), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("value", ["invalid", 1, None])
def test_rejects_invalid_svd_solver(value: object) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="svd_solver"):
        PiPLSRegression(
            n_components=1,
            predictor_rank=1,
            svd_solver=value,  # type: ignore[arg-type]
        ).fit(X, Y)


@pytest.mark.parametrize("value", [-1, True, 1.5, "seed"])
def test_rejects_invalid_random_state(value: object) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="random_state"):
        PiPLSRegression(
            n_components=1,
            predictor_rank=1,
            random_state=value,  # type: ignore[arg-type]
        ).fit(X, Y)


def test_randomized_svd_accepts_none_and_random_state_instances() -> None:
    X, Y = _data()
    none_model = PiPLSRegression(
        n_components=2,
        predictor_rank=3,
        svd_solver="randomized",
        random_state=None,
    ).fit(X, Y)
    first = PiPLSRegression(
        n_components=2,
        predictor_rank=3,
        svd_solver="randomized",
        random_state=np.random.RandomState(23),
    ).fit(X, Y)
    second = PiPLSRegression(
        n_components=2,
        predictor_rank=3,
        svd_solver="randomized",
        random_state=np.random.RandomState(23),
    ).fit(X, Y)

    assert none_model.decomposition_.predictor_svd_solver == "randomized"
    np.testing.assert_allclose(first.coef_, second.coef_)
