from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pipls import (
    PiPLSComponentPath,
    PiPLSComponentResult,
    PiPLSPredictorRankProfile,
)


def _component_path() -> PiPLSComponentPath:
    return PiPLSComponentPath(
        n_components=np.array([1, 2, 4], dtype=np.int64),
        predictor_rank=np.array([3, 4, 5], dtype=np.int64),
        predictor_rank_policy="optimized",
        mean_test_score=np.array([-0.8, -0.5, -0.45], dtype=np.float32),
        cv_mse_mean=np.array([0.8, 0.5, 0.45], dtype=np.float32),
        cv_mse_fold_sd=np.array([0.1, 0.08, 0.07], dtype=np.float32),
        n_splits=np.int64(5),
    )


def test_component_path_makes_aligned_read_only_defensive_copies() -> None:
    n_components = np.array([1, 2], dtype=np.int64)
    predictor_rank = np.array([2, 3], dtype=np.int64)
    mean_test_score = np.array([-0.6, -0.4], dtype=np.float32)
    cv_mse_mean = np.array([0.6, 0.4], dtype=np.float32)
    cv_mse_fold_sd = np.array([0.2, 0.1], dtype=np.float32)

    path = PiPLSComponentPath(
        n_components=n_components,
        predictor_rank=predictor_rank,
        predictor_rank_policy="fixed",
        mean_test_score=mean_test_score,
        cv_mse_mean=cv_mse_mean,
        cv_mse_fold_sd=cv_mse_fold_sd,
        n_splits=np.int64(4),
    )
    n_components[0] = 99
    predictor_rank[0] = 99
    mean_test_score[0] = 99.0
    cv_mse_mean[0] = 99.0
    cv_mse_fold_sd[0] = 99.0

    arrays = (
        path.n_components,
        path.predictor_rank,
        path.mean_test_score,
        path.cv_mse_mean,
        path.cv_mse_fold_sd,
    )
    assert all(array.shape == (2,) for array in arrays)
    assert path.n_components.dtype == np.dtype(np.intp)
    assert path.predictor_rank.dtype == np.dtype(np.intp)
    assert path.mean_test_score.dtype == np.dtype(np.float64)
    assert path.cv_mse_mean.dtype == np.dtype(np.float64)
    assert path.cv_mse_fold_sd.dtype == np.dtype(np.float64)
    assert path.cv_mse_standard_error.dtype == np.dtype(np.float64)
    assert all(not array.flags.writeable for array in arrays)
    assert path.predictor_rank_policy == "fixed"
    assert path.n_splits == 4
    assert type(path.n_splits) is int
    np.testing.assert_array_equal(path.n_components, np.array([1, 2]))
    np.testing.assert_array_equal(path.predictor_rank, np.array([2, 3]))
    np.testing.assert_allclose(path.mean_test_score, np.array([-0.6, -0.4]))
    np.testing.assert_allclose(path.cv_mse_mean, np.array([0.6, 0.4]))
    np.testing.assert_allclose(path.cv_mse_fold_sd, np.array([0.2, 0.1]))
    np.testing.assert_allclose(
        path.cv_mse_standard_error,
        np.array([0.2, 0.1]) / np.sqrt(3.0),
    )
    assert not path.cv_mse_standard_error.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        path.cv_mse_mean[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        path.n_components = np.array([1])  # type: ignore[misc]


def test_component_path_requires_aligned_ascending_valid_values() -> None:
    kwargs = {
        "n_components": [1, 2],
        "predictor_rank": [2, 3],
        "predictor_rank_policy": "optimized",
        "mean_test_score": [-0.6, -0.4],
        "cv_mse_mean": [0.6, 0.4],
        "cv_mse_fold_sd": [0.2, 0.1],
        "n_splits": 4,
    }

    with pytest.raises(ValueError, match="same length"):
        PiPLSComponentPath(**{**kwargs, "predictor_rank": [2]})
    with pytest.raises(ValueError, match="strictly ascending"):
        PiPLSComponentPath(**{**kwargs, "n_components": [2, 1]})
    with pytest.raises(ValueError, match="must be one of"):
        PiPLSComponentPath(**{**kwargs, "predictor_rank_policy": "unknown"})
    one_split_path = PiPLSComponentPath(**{**kwargs, "n_splits": 1})
    with pytest.raises(ValueError, match="requires at least two"):
        _ = one_split_path.cv_mse_standard_error


def test_component_path_scalar_lookup_returns_frozen_python_values() -> None:
    path = _component_path()

    selected = path.for_n_components(np.int64(2))

    assert isinstance(selected, PiPLSComponentResult)
    assert selected.n_components == 2
    assert selected.predictor_rank == 4
    assert selected.predictor_rank_policy == "optimized"
    assert selected.mean_test_score == pytest.approx(-0.5)
    assert selected.cv_mse_mean == pytest.approx(0.5)
    assert selected.cv_mse_fold_sd == pytest.approx(0.08)
    assert selected.cv_mse_standard_error == pytest.approx(0.08 / np.sqrt(4.0))
    assert selected.n_splits == 5
    assert type(selected.n_components) is int
    assert type(selected.predictor_rank) is int
    assert type(selected.mean_test_score) is float
    with pytest.raises(FrozenInstanceError):
        selected.predictor_rank = 5  # type: ignore[misc]


def test_component_path_minimum_cv_mse_result_uses_exact_stored_rows() -> None:
    path = PiPLSComponentPath(
        n_components=[1, 2, 4],
        predictor_rank=[2, 4, 6],
        predictor_rank_policy="optimized",
        mean_test_score=[-0.5, -0.4, -0.4],
        cv_mse_mean=[0.5, 0.4, 0.4],
        cv_mse_fold_sd=[0.1, 0.08, 0.07],
        n_splits=5,
    )

    result = path.minimum_cv_mse_result()

    assert result == path.for_n_components(2)
    assert result.predictor_rank == 4
    assert result.predictor_rank_policy == "optimized"


def test_component_path_one_standard_error_result_uses_reference_row_se() -> None:
    path = PiPLSComponentPath(
        n_components=[1, 2, 3, 4],
        predictor_rank=[2, 3, 5, 6],
        predictor_rank_policy="optimized",
        mean_test_score=[-0.48, -0.45, -0.40, -0.42],
        cv_mse_mean=[0.48, 0.45, 0.40, 0.42],
        cv_mse_fold_sd=[1.0, 0.4, 0.08, 0.2],
        n_splits=5,
    )

    result = path.one_standard_error_result()

    assert result == path.for_n_components(3)
    assert result.predictor_rank == 5


def test_component_path_one_standard_error_result_returns_smallest_eligible_count() -> None:
    path = PiPLSComponentPath(
        n_components=[1, 2, 4],
        predictor_rank=[2, 3, 5],
        predictor_rank_policy="optimized",
        mean_test_score=[-0.50, -0.44, -0.40],
        cv_mse_mean=[0.50, 0.44, 0.40],
        cv_mse_fold_sd=[0.1, 0.1, 0.10],
        n_splits=5,
    )

    result = path.one_standard_error_result()

    assert result == path.for_n_components(2)


def test_component_path_one_standard_error_result_uses_no_tolerance() -> None:
    minimum = 0.4
    reference_standard_error = 0.04
    threshold = minimum + reference_standard_error
    just_above_threshold = np.nextafter(threshold, np.inf)
    path = PiPLSComponentPath(
        n_components=[1, 2, 3],
        predictor_rank=[2, 3, 4],
        predictor_rank_policy="optimized",
        mean_test_score=[-0.5, -just_above_threshold, -minimum],
        cv_mse_mean=[0.5, just_above_threshold, minimum],
        cv_mse_fold_sd=[0.1, 0.1, 2.0 * reference_standard_error],
        n_splits=5,
    )

    result = path.one_standard_error_result()

    assert result.n_components == 3


def test_component_path_recommendations_handle_split_and_range_edges() -> None:
    one_split = PiPLSComponentPath(
        n_components=[1, 2],
        predictor_rank=[2, 3],
        predictor_rank_policy="optimized",
        mean_test_score=[-0.4, -0.5],
        cv_mse_mean=[0.4, 0.5],
        cv_mse_fold_sd=[0.0, 0.1],
        n_splits=1,
    )
    assert one_split.minimum_cv_mse_result().n_components == 1
    with pytest.raises(ValueError, match="requires at least two"):
        one_split.one_standard_error_result()

    overflowing = PiPLSComponentPath(
        n_components=[1],
        predictor_rank=[1],
        predictor_rank_policy="fixed",
        mean_test_score=[-1.0e308],
        cv_mse_mean=[1.0e308],
        cv_mse_fold_sd=[1.0e308],
        n_splits=2,
    )
    with pytest.raises(ValueError, match="threshold must be finite"):
        overflowing.one_standard_error_result()


def test_component_path_lookup_rejects_unevaluated_or_noninteger_counts() -> None:
    path = _component_path()

    with pytest.raises(ValueError, match=r"n_components=3 was not evaluated.*\[1, 2, 4\]"):
        path.for_n_components(3)
    with pytest.raises(ValueError, match="must be an integer"):
        path.for_n_components(2.0)  # type: ignore[arg-type]


def test_component_path_is_pickleable_with_read_only_arrays() -> None:
    restored = pickle.loads(pickle.dumps(_component_path()))

    assert isinstance(restored, PiPLSComponentPath)
    np.testing.assert_array_equal(restored.n_components, np.array([1, 2, 4]))
    assert not restored.n_components.flags.writeable
    assert restored.predictor_rank_policy == "optimized"
    assert restored.n_splits == 5
    assert restored.for_n_components(4).predictor_rank == 5
    assert restored.minimum_cv_mse_result() == restored.for_n_components(4)
    assert restored.one_standard_error_result() == restored.for_n_components(4)
    np.testing.assert_allclose(
        restored.cv_mse_standard_error,
        _component_path().cv_mse_fold_sd / np.sqrt(4.0),
    )


def _predictor_rank_profile() -> PiPLSPredictorRankProfile:
    return PiPLSPredictorRankProfile(
        n_components=2,
        predictor_rank=np.array([2, 3, 4], dtype=np.int64),
        mean_test_score=np.array([-0.6, -0.5, -0.4], dtype=np.float32),
        cv_mse_mean=np.array([0.6, 0.5, 0.4], dtype=np.float32),
        cv_mse_fold_sd=np.array([0.10, 0.09, 0.08], dtype=np.float32),
        predictor_rank_policy="optimized",
        n_splits=5,
    )


def test_predictor_rank_profile_makes_read_only_defensive_copies() -> None:
    predictor_rank = np.array([2, 3], dtype=np.int64)
    mean_test_score = np.array([-0.6, -0.4], dtype=np.float32)
    cv_mse_mean = np.array([0.6, 0.4], dtype=np.float32)
    cv_mse_fold_sd = np.array([0.2, 0.1], dtype=np.float32)

    profile = PiPLSPredictorRankProfile(
        n_components=np.int64(2),
        predictor_rank=predictor_rank,
        mean_test_score=mean_test_score,
        cv_mse_mean=cv_mse_mean,
        cv_mse_fold_sd=cv_mse_fold_sd,
        predictor_rank_policy="optimized",
        n_splits=np.int64(4),
    )
    predictor_rank[0] = 99
    mean_test_score[0] = 99.0
    cv_mse_mean[0] = 99.0
    cv_mse_fold_sd[0] = 99.0

    arrays = (
        profile.predictor_rank,
        profile.mean_test_score,
        profile.cv_mse_mean,
        profile.cv_mse_fold_sd,
    )
    assert all(array.shape == (2,) for array in arrays)
    assert all(not array.flags.writeable for array in arrays)
    assert profile.n_components == 2
    assert type(profile.n_components) is int
    assert profile.predictor_rank_policy == "optimized"
    assert profile.n_splits == 4
    assert type(profile.n_splits) is int
    np.testing.assert_array_equal(profile.predictor_rank, np.array([2, 3]))
    np.testing.assert_allclose(profile.cv_mse_mean, np.array([0.6, 0.4]))
    np.testing.assert_allclose(
        profile.cv_mse_standard_error,
        np.array([0.2, 0.1]) / np.sqrt(3.0),
    )
    assert profile.cv_mse_standard_error.dtype == np.dtype(np.float64)
    assert not profile.cv_mse_standard_error.flags.writeable
    assert profile.selected_result.predictor_rank == 3

    with pytest.raises(ValueError, match="read-only"):
        profile.cv_mse_mean[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        profile.n_components = 3  # type: ignore[misc]


def test_predictor_rank_profile_validates_alignment_and_shared_values() -> None:
    profile = _predictor_rank_profile()
    kwargs = {
        "n_components": profile.n_components,
        "predictor_rank": profile.predictor_rank,
        "mean_test_score": profile.mean_test_score,
        "cv_mse_mean": profile.cv_mse_mean,
        "cv_mse_fold_sd": profile.cv_mse_fold_sd,
        "predictor_rank_policy": profile.predictor_rank_policy,
        "n_splits": profile.n_splits,
    }

    with pytest.raises(ValueError, match="same length"):
        PiPLSPredictorRankProfile(**{**kwargs, "cv_mse_mean": [0.6, 0.5]})
    one_split_profile = PiPLSPredictorRankProfile(
        **{
            **kwargs,
            "cv_mse_fold_sd": [0.10, 0.09, 0.0],
            "n_splits": 1,
        }
    )
    with pytest.raises(ValueError, match="requires at least two"):
        _ = one_split_profile.cv_mse_standard_error
    with pytest.raises(ValueError, match="strictly ascending"):
        PiPLSPredictorRankProfile(**{**kwargs, "predictor_rank": [2, 4, 3]})
    with pytest.raises(ValueError, match="smaller than n_components"):
        PiPLSPredictorRankProfile(**{**kwargs, "n_components": 3})
    with pytest.raises(ValueError, match="must be one of"):
        PiPLSPredictorRankProfile(
            **{**kwargs, "predictor_rank_policy": "unknown"}
        )


def test_predictor_rank_profile_derives_selected_with_fitted_tie_rule() -> None:
    profile = PiPLSPredictorRankProfile(
        n_components=2,
        predictor_rank=[2, 3, 4],
        mean_test_score=[1.0, 1.0 - 5.0e-13, 0.5],
        cv_mse_mean=[0.6, 0.5, 0.4],
        cv_mse_fold_sd=[0.10, 0.09, 0.08],
        predictor_rank_policy="optimized",
        n_splits=5,
    )

    selected = profile.selected_result

    assert selected == PiPLSComponentResult(
        n_components=2,
        predictor_rank=2,
        predictor_rank_policy="optimized",
        mean_test_score=1.0,
        cv_mse_mean=0.6,
        cv_mse_fold_sd=0.10,
        n_splits=5,
    )


def test_predictor_rank_profile_is_pickleable_with_read_only_arrays() -> None:
    restored = pickle.loads(pickle.dumps(_predictor_rank_profile()))

    assert isinstance(restored, PiPLSPredictorRankProfile)
    np.testing.assert_array_equal(restored.predictor_rank, np.array([2, 3, 4]))
    assert not restored.predictor_rank.flags.writeable
    assert restored.predictor_rank_policy == "optimized"
    assert restored.selected_result.predictor_rank == 4
    np.testing.assert_allclose(
        restored.cv_mse_standard_error,
        _predictor_rank_profile().cv_mse_fold_sd / np.sqrt(4.0),
    )
