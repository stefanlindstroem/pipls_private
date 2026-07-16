from __future__ import annotations

import warnings

import numpy as np
import pytest

from pipls import PiPLSRegression, StatisticalSupportWarning

_MAX_RANDOM_STATE = int(np.iinfo(np.uint32).max)


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260716)
    X = rng.normal(size=(24, 6))
    Y = X @ rng.normal(size=(6, 3)) + 0.02 * rng.normal(size=(24, 3))
    return X, Y


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        np.bool_(False),
        1.0,
        np.float64(1.0),
        "1",
        None,
        np.nan,
        np.inf,
        [],
        np.asarray([1]),
    ],
)
def test_n_components_rejects_nonpositive_and_noninteger_values(value: object) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="n_components must be a positive integer"):
        PiPLSRegression(n_components=value, predictor_rank=2).fit(X, Y)  # type: ignore[arg-type]


def test_n_components_accepts_numpy_integer() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=np.int64(2), predictor_rank=3).fit(X, Y)
    assert model.P_.shape[1] == 2


def test_n_components_rejects_more_components_than_response_columns() -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="number of response columns"):
        PiPLSRegression(n_components=4, predictor_rank=4).fit(X, Y)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        np.bool_(True),
        2.0,
        np.float64(2.0),
        "2",
        "exhaustive",
        None,
        np.inf,
        [],
        np.asarray([2]),
    ],
)
def test_predictor_rank_rejects_invalid_integer_and_mode_values(value: object) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="predictor_rank must be a positive integer"):
        PiPLSRegression(n_components=1, predictor_rank=value).fit(X, Y)  # type: ignore[arg-type]


def test_predictor_rank_accepts_numpy_integer() -> None:
    X, Y = _data()
    model = PiPLSRegression(n_components=2, predictor_rank=np.int64(3)).fit(X, Y)
    assert model.predictor_rank_ == 3


def test_predictor_rank_rejects_rank_above_matrix_dimensions() -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="predictor_rank <= min"):
        PiPLSRegression(n_components=1, predictor_rank=7).fit(X, Y)


def test_predictor_rank_rejects_rank_above_numerical_rank() -> None:
    X, Y = _data()
    X[:, 2:] = X[:, :1]
    with pytest.raises(ValueError, match="numerical rank"):
        PiPLSRegression(n_components=1, predictor_rank=3).fit(X, Y)


@pytest.mark.parametrize("mode", ["max", "optimal", "auto"])
def test_rule_modes_reject_upper_bound_below_n_components(mode: str) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="n_components"):
        PiPLSRegression(
            n_components=2,
            predictor_rank=mode,  # type: ignore[arg-type]
            samples_per_predictor_rank=1000,
            cv=2,
            n_jobs=1,
        ).fit(X, Y)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        np.bool_(False),
        np.nan,
        np.inf,
        -np.inf,
        "5",
        None,
        1 + 0j,
        [],
    ],
)
def test_samples_per_predictor_rank_rejects_nonpositive_or_nonfinite_values(
    value: object,
) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="samples_per_predictor_rank"):
        PiPLSRegression(
            n_components=1,
            predictor_rank="max",
            samples_per_predictor_rank=value,  # type: ignore[arg-type]
        ).fit(X, Y)


@pytest.mark.parametrize("mode", ["max", "optimal", "auto"])
def test_low_samples_per_predictor_rank_warns_once_for_rule_modes(mode: str) -> None:
    X, Y = _data()
    with pytest.warns(
        StatisticalSupportWarning,
        match="not have sufficient statistical support to be trusted",
    ) as records:
        PiPLSRegression(
            n_components=1,
            predictor_rank=mode,  # type: ignore[arg-type]
            samples_per_predictor_rank=4,
            cv=2,
            n_jobs=1,
        ).fit(X, Y)
    assert len(records) == 1


def test_samples_per_predictor_rank_five_does_not_warn() -> None:
    X, Y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", StatisticalSupportWarning)
        PiPLSRegression(
            n_components=1,
            predictor_rank="max",
            samples_per_predictor_rank=5,
        ).fit(X, Y)


def test_low_samples_per_predictor_rank_does_not_warn_when_rank_is_explicit() -> None:
    X, Y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", StatisticalSupportWarning)
        PiPLSRegression(
            n_components=1,
            predictor_rank=2,
            samples_per_predictor_rank=1,
        ).fit(X, Y)


def test_extremely_small_positive_samples_per_predictor_rank_saturates_safely() -> None:
    X, Y = _data()
    tiny_positive = np.nextafter(0.0, 1.0)
    with pytest.warns(StatisticalSupportWarning):
        model = PiPLSRegression(
            n_components=1,
            predictor_rank="max",
            samples_per_predictor_rank=tiny_positive,
        ).fit(X, Y)
    assert model.max_predictor_rank_ == min(X.shape)
    assert model.predictor_rank_ == min(X.shape)


def test_extremely_large_samples_per_predictor_rank_gives_rank_one() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=1,
        predictor_rank="max",
        samples_per_predictor_rank=np.finfo(np.float64).max,
    ).fit(X, Y)
    assert model.max_predictor_rank_ == 1
    assert model.predictor_rank_ == 1


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(False), 0, 1, -1, 2.0, np.float64(2.0), np.nan, np.inf, "5"],
)
def test_cv_rejects_invalid_scalar_values(value: object) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="cv must"):
        PiPLSRegression(
            n_components=1,
            predictor_rank="auto",
            cv=value,
            n_jobs=1,
        ).fit(X, Y)


def test_cv_accepts_numpy_integer_and_materializes_requested_splits() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=1,
        predictor_rank="auto",
        cv=np.int64(3),
        n_jobs=1,
    ).fit(X, Y)
    assert model.n_splits_ == 3


def test_cv_rejects_more_splits_than_samples() -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="number of splits"):
        PiPLSRegression(
            n_components=1,
            predictor_rank="auto",
            cv=X.shape[0] + 1,
            n_jobs=1,
        ).fit(X, Y)


def test_cv_rejects_empty_split_iterable() -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="at least one split"):
        PiPLSRegression(
            n_components=1,
            predictor_rank="auto",
            cv=[],
            n_jobs=1,
        ).fit(X, Y)


@pytest.mark.parametrize(
    "value",
    [0, True, np.bool_(False), 1.0, np.float64(-1.0), "1", np.nan, np.inf, []],
)
def test_n_jobs_rejects_invalid_values_even_when_rank_is_explicit(value: object) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="n_jobs"):
        PiPLSRegression(
            n_components=1,
            predictor_rank=2,
            n_jobs=value,  # type: ignore[arg-type]
        ).fit(X, Y)


@pytest.mark.parametrize("value", [None, 1, -1, -100, np.int64(2)])
def test_n_jobs_accepts_none_and_nonzero_integers(value: object) -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=1,
        predictor_rank=2,
        n_jobs=value,  # type: ignore[arg-type]
    ).fit(X, Y)
    assert model.predictor_rank_ == 2


@pytest.mark.parametrize(
    "value",
    [
        -1,
        True,
        np.bool_(False),
        1.0,
        np.float64(1.0),
        "1",
        _MAX_RANDOM_STATE + 1,
        np.uint64(_MAX_RANDOM_STATE + 1),
        [],
    ],
)
def test_random_state_rejects_invalid_or_out_of_range_values(value: object) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="random_state"):
        PiPLSRegression(
            n_components=1,
            predictor_rank=2,
            svd_solver="auto",
            random_state=value,  # type: ignore[arg-type]
        ).fit(X, Y)


@pytest.mark.parametrize("solver", ["auto", "randomized"])
def test_random_state_none_is_rejected_when_solver_may_randomize(solver: str) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="random_state"):
        PiPLSRegression(
            n_components=1,
            predictor_rank=2,
            svd_solver=solver,  # type: ignore[arg-type]
            random_state=None,
        ).fit(X, Y)


def test_random_state_none_is_accepted_for_full_solver() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=1,
        predictor_rank=2,
        svd_solver="full",
        random_state=None,
    ).fit(X, Y)
    assert model.svd_solver_ == "full"


def test_random_state_accepts_maximum_uint32_seed() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=1,
        predictor_rank=2,
        svd_solver="randomized",
        random_state=np.uint64(_MAX_RANDOM_STATE),
    ).fit(X, Y)
    assert model.svd_solver_ == "randomized"


@pytest.mark.parametrize("parameter", ["scale", "copy"])
@pytest.mark.parametrize("value", [0, 1, 1.0, "true", None, []])
def test_boolean_parameters_reject_nonboolean_values(parameter: str, value: object) -> None:
    X, Y = _data()
    kwargs = {parameter: value}
    with pytest.raises(ValueError, match=f"{parameter} must be boolean"):
        PiPLSRegression(
            n_components=1,
            predictor_rank=2,
            **kwargs,  # type: ignore[arg-type]
        ).fit(X, Y)


def test_numpy_boolean_parameters_are_accepted() -> None:
    X, Y = _data()
    model = PiPLSRegression(
        n_components=1,
        predictor_rank=2,
        scale=np.bool_(True),
        copy=np.bool_(False),
    ).fit(X, Y)
    assert model.predictor_rank_ == 2


@pytest.mark.parametrize("value", [1, None, [], np.asarray(["full"])])
def test_svd_solver_rejects_nonstring_values_cleanly(value: object) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="svd_solver"):
        PiPLSRegression(
            n_components=1,
            predictor_rank=2,
            svd_solver=value,  # type: ignore[arg-type]
        ).fit(X, Y)


@pytest.mark.parametrize("value", [1, [], np.asarray(["neg_mean_squared_error"])])
def test_scoring_rejects_nonstring_noncallable_values_cleanly(value: object) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="scoring"):
        PiPLSRegression(
            n_components=1,
            predictor_rank="auto",
            scoring=value,  # type: ignore[arg-type]
            cv=2,
            n_jobs=1,
        ).fit(X, Y)
