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


def test_default_predictor_rank_mode_is_auto() -> None:
    assert PiPLSRegression().predictor_rank == "auto"


def test_auto_rank_mode_exposes_diagnostics() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=2,
        predictor_rank="auto",
        samples_per_predictor_rank=8,
        cv=4,
        n_jobs=1,
    ).fit(X, Y)

    assert model.max_predictor_rank_ == 4
    np.testing.assert_array_equal(model.predictor_rank_values_, np.array([2, 3, 4]))
    assert model.predictor_rank_ in model.predictor_rank_values_
    assert model.n_splits_ == 4
    assert model.cv_n_train_min_ == 30
    results = model.predictor_rank_cv_results_
    assert results["mean_test_score"].shape == (3,)
    assert results["mean_response_standardized_mse"].shape == (3,)
    selected_index = int(np.flatnonzero(model.predictor_rank_values_ == model.predictor_rank_)[0])
    assert model.best_score_ == pytest.approx(results["mean_test_score"][selected_index])
    assert model.best_response_standardized_mse_ == pytest.approx(
        results["mean_response_standardized_mse"][selected_index]
    )


def test_auto_rank_refits_selected_rank_on_all_data() -> None:
    X, Y = _data()
    automatic = PiPLSRegression(
        n_components=2,
        predictor_rank="auto",
        samples_per_predictor_rank=8,
        cv=4,
        n_jobs=1,
    ).fit(X, Y)
    explicit = PiPLSRegression(
        n_components=2,
        predictor_rank=automatic.predictor_rank_,
    ).fit(X, Y)

    np.testing.assert_allclose(automatic.coef_, explicit.coef_)
    np.testing.assert_allclose(automatic.intercept_, explicit.intercept_)
    np.testing.assert_allclose(automatic.predict(X), explicit.predict(X))


def test_auto_rank_uses_smallest_materialized_training_fold() -> None:
    X, Y = _data()
    splits = [
        (np.arange(8), np.arange(8, 12)),
        (np.arange(6), np.arange(12, 16)),
    ]
    model = PiPLSRegression(
        n_components=2,
        predictor_rank="auto",
        samples_per_predictor_rank=2,
        cv=splits,
        n_jobs=1,
    ).fit(X[:16], Y[:16])

    assert model.cv_n_train_min_ == 6
    assert model.max_predictor_rank_ == 3
    np.testing.assert_array_equal(model.predictor_rank_values_, np.array([2, 3]))


def test_auto_rank_uses_fold_local_preprocessing() -> None:
    X, Y = _data()
    splits = [
        (np.arange(8), np.arange(8, 12)),
        (np.arange(4, 12), np.arange(0, 4)),
    ]
    automatic = PiPLSRegression(
        n_components=2,
        predictor_rank="auto",
        samples_per_predictor_rank=4,
        cv=splits,
        n_jobs=1,
    ).fit(X[:12], Y[:12])
    manual = PiPLSRegression(n_components=2, predictor_rank=2).fit(
        X[:8],
        Y[:8],
    )
    response_scale = np.std(Y[:8], axis=0, ddof=1)
    expected = np.mean(
        ((Y[8:12] - manual.predict(X[8:12])) / response_scale[None, :]) ** 2
    )

    assert automatic.predictor_rank_cv_results_[
        "split0_response_standardized_mse"
    ][0] == pytest.approx(expected)


def test_auto_rank_uses_smallest_rank_for_equal_scores() -> None:
    X, Y = _data()

    def constant_scorer(
        estimator: PiPLSRegression,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
    ) -> float:
        del estimator, X_validation, y_validation
        return 1.0

    model = PiPLSRegression(
        n_components=2,
        predictor_rank="auto",
        samples_per_predictor_rank=5,
        cv=3,
        scoring=constant_scorer,
        n_jobs=1,
    ).fit(X, Y)

    assert model.predictor_rank_ == 2


@pytest.mark.parametrize("value", [0, True, 1.5])
def test_auto_rank_rejects_invalid_n_jobs(value: object) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="n_jobs"):
        PiPLSRegression(
            n_components=2,
            predictor_rank="auto",
            n_jobs=value,  # type: ignore[arg-type]
        ).fit(X, Y)


def test_auto_rank_rejects_unknown_scorer() -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="Unknown scoring"):
        PiPLSRegression(
            n_components=2,
            predictor_rank="auto",
            scoring="not_a_scorer",
        ).fit(X, Y)


def test_auto_rank_accepts_sklearn_scorer_and_parallel_candidates() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=2,
        predictor_rank="auto",
        samples_per_predictor_rank=8,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=2,
    ).fit(X, Y)

    assert model.predictor_rank_ in model.predictor_rank_values_
    assert np.isfinite(model.best_score_)
    assert np.isfinite(model.best_response_standardized_mse_)
