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
        predictor_rank_policy=np.array(["optimized"] * 3, dtype=object),
        mean_test_score=np.array([-0.8, -0.5, -0.45], dtype=np.float32),
        cv_mse_mean=np.array([0.8, 0.5, 0.45], dtype=np.float32),
        cv_mse_fold_sd=np.array([0.1, 0.08, 0.07], dtype=np.float32),
        n_splits=np.array([5, 5, 5], dtype=np.int64),
    )


def test_component_path_makes_aligned_read_only_defensive_copies() -> None:
    n_components = np.array([1, 2], dtype=np.int64)
    predictor_rank = np.array([2, 3], dtype=np.int64)
    predictor_rank_policy = np.array(["fixed", "fixed"], dtype=object)
    mean_test_score = np.array([-0.6, -0.4], dtype=np.float32)
    cv_mse_mean = np.array([0.6, 0.4], dtype=np.float32)
    cv_mse_fold_sd = np.array([0.2, 0.1], dtype=np.float32)
    n_splits = np.array([4, 5], dtype=np.int64)

    path = PiPLSComponentPath(
        n_components=n_components,
        predictor_rank=predictor_rank,
        predictor_rank_policy=predictor_rank_policy,
        mean_test_score=mean_test_score,
        cv_mse_mean=cv_mse_mean,
        cv_mse_fold_sd=cv_mse_fold_sd,
        n_splits=n_splits,
    )
    n_components[0] = 99
    predictor_rank[0] = 99
    predictor_rank_policy[0] = "optimized"
    mean_test_score[0] = 99.0
    cv_mse_mean[0] = 99.0
    cv_mse_fold_sd[0] = 99.0
    n_splits[0] = 99

    arrays = (
        path.n_components,
        path.predictor_rank,
        path.predictor_rank_policy,
        path.mean_test_score,
        path.cv_mse_mean,
        path.cv_mse_fold_sd,
        path.n_splits,
    )
    assert all(array.shape == (2,) for array in arrays)
    assert path.n_components.dtype == np.dtype(np.intp)
    assert path.predictor_rank.dtype == np.dtype(np.intp)
    assert path.predictor_rank_policy.dtype.kind == "U"
    assert path.mean_test_score.dtype == np.dtype(np.float64)
    assert path.cv_mse_mean.dtype == np.dtype(np.float64)
    assert path.cv_mse_fold_sd.dtype == np.dtype(np.float64)
    assert path.cv_mse_standard_error.dtype == np.dtype(np.float64)
    assert path.n_splits.dtype == np.dtype(np.intp)
    assert all(not array.flags.writeable for array in arrays)
    np.testing.assert_array_equal(path.n_components, np.array([1, 2]))
    np.testing.assert_array_equal(path.predictor_rank, np.array([2, 3]))
    assert path.predictor_rank_policy.tolist() == ["fixed", "fixed"]
    np.testing.assert_allclose(path.mean_test_score, np.array([-0.6, -0.4]))
    np.testing.assert_allclose(path.cv_mse_mean, np.array([0.6, 0.4]))
    np.testing.assert_allclose(path.cv_mse_fold_sd, np.array([0.2, 0.1]))
    np.testing.assert_allclose(
        path.cv_mse_standard_error,
        np.array([0.2 / np.sqrt(3.0), 0.1 / np.sqrt(4.0)]),
    )
    assert not path.cv_mse_standard_error.flags.writeable
    np.testing.assert_array_equal(path.n_splits, np.array([4, 5]))

    with pytest.raises(ValueError, match="read-only"):
        path.cv_mse_mean[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        path.n_components = np.array([1])  # type: ignore[misc]


def test_component_path_requires_aligned_ascending_valid_arrays() -> None:
    kwargs = {
        "n_components": [1, 2],
        "predictor_rank": [2, 3],
        "predictor_rank_policy": ["optimized", "optimized"],
        "mean_test_score": [-0.6, -0.4],
        "cv_mse_mean": [0.6, 0.4],
        "cv_mse_fold_sd": [0.2, 0.1],
        "n_splits": [4, 4],
    }

    with pytest.raises(ValueError, match="same length"):
        PiPLSComponentPath(**{**kwargs, "predictor_rank": [2]})
    with pytest.raises(ValueError, match="strictly ascending"):
        PiPLSComponentPath(**{**kwargs, "n_components": [2, 1]})
    with pytest.raises(ValueError, match="unsupported values"):
        PiPLSComponentPath(
            **{**kwargs, "predictor_rank_policy": ["optimized", "unknown"]}
        )
    one_split_path = PiPLSComponentPath(**{**kwargs, "n_splits": [4, 1]})
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
    assert restored.for_n_components(4).predictor_rank == 5
    np.testing.assert_allclose(
        restored.cv_mse_standard_error,
        _component_path().cv_mse_fold_sd / np.sqrt(4.0),
    )


def _predictor_rank_profile() -> PiPLSPredictorRankProfile:
    selected = PiPLSComponentResult(
        n_components=2,
        predictor_rank=4,
        predictor_rank_policy="optimized",
        mean_test_score=-0.4,
        cv_mse_mean=0.4,
        cv_mse_fold_sd=0.08,
        n_splits=5,
    )
    return PiPLSPredictorRankProfile(
        n_components=2,
        predictor_rank=np.array([2, 3, 4], dtype=np.int64),
        mean_test_score=np.array([-0.6, -0.5, -0.4], dtype=np.float32),
        cv_mse_mean=np.array([0.6, 0.5, 0.4], dtype=np.float32),
        cv_mse_fold_sd=np.array([0.10, 0.09, 0.08], dtype=np.float32),
        n_splits=5,
        selected=selected,
    )


def test_predictor_rank_profile_makes_read_only_defensive_copies() -> None:
    predictor_rank = np.array([2, 3], dtype=np.int64)
    mean_test_score = np.array([-0.6, -0.4], dtype=np.float32)
    cv_mse_mean = np.array([0.6, 0.4], dtype=np.float32)
    cv_mse_fold_sd = np.array([0.2, 0.1], dtype=np.float32)
    selected = PiPLSComponentResult(
        n_components=2,
        predictor_rank=3,
        predictor_rank_policy="optimized",
        mean_test_score=-0.4,
        cv_mse_mean=0.4,
        cv_mse_fold_sd=0.1,
        n_splits=4,
    )

    profile = PiPLSPredictorRankProfile(
        n_components=np.int64(2),
        predictor_rank=predictor_rank,
        mean_test_score=mean_test_score,
        cv_mse_mean=cv_mse_mean,
        cv_mse_fold_sd=cv_mse_fold_sd,
        n_splits=np.int64(4),
        selected=selected,
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

    with pytest.raises(ValueError, match="read-only"):
        profile.cv_mse_mean[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        profile.n_components = 3  # type: ignore[misc]


def test_predictor_rank_profile_validates_alignment_and_selected_row() -> None:
    profile = _predictor_rank_profile()
    kwargs = {
        "n_components": profile.n_components,
        "predictor_rank": profile.predictor_rank,
        "mean_test_score": profile.mean_test_score,
        "cv_mse_mean": profile.cv_mse_mean,
        "cv_mse_fold_sd": profile.cv_mse_fold_sd,
        "n_splits": profile.n_splits,
        "selected": profile.selected,
    }

    with pytest.raises(ValueError, match="same length"):
        PiPLSPredictorRankProfile(**{**kwargs, "cv_mse_mean": [0.6, 0.5]})
    one_split_selected = PiPLSComponentResult(
        n_components=2,
        predictor_rank=4,
        predictor_rank_policy="optimized",
        mean_test_score=-0.4,
        cv_mse_mean=0.4,
        cv_mse_fold_sd=0.0,
        n_splits=1,
    )
    one_split_profile = PiPLSPredictorRankProfile(
        **{
            **kwargs,
            "cv_mse_fold_sd": [0.10, 0.09, 0.0],
            "n_splits": 1,
            "selected": one_split_selected,
        }
    )
    with pytest.raises(ValueError, match="requires at least two"):
        _ = one_split_profile.cv_mse_standard_error
    with pytest.raises(ValueError, match="strictly ascending"):
        PiPLSPredictorRankProfile(**{**kwargs, "predictor_rank": [2, 4, 3]})
    with pytest.raises(ValueError, match="must match n_components"):
        PiPLSPredictorRankProfile(**{**kwargs, "n_components": 1})
    inconsistent = PiPLSComponentResult(
        n_components=2,
        predictor_rank=4,
        predictor_rank_policy="optimized",
        mean_test_score=-0.3,
        cv_mse_mean=0.4,
        cv_mse_fold_sd=0.08,
        n_splits=5,
    )
    with pytest.raises(ValueError, match="score values must match"):
        PiPLSPredictorRankProfile(**{**kwargs, "selected": inconsistent})


def test_predictor_rank_profile_is_pickleable_with_read_only_arrays() -> None:
    restored = pickle.loads(pickle.dumps(_predictor_rank_profile()))

    assert isinstance(restored, PiPLSPredictorRankProfile)
    np.testing.assert_array_equal(restored.predictor_rank, np.array([2, 3, 4]))
    assert not restored.predictor_rank.flags.writeable
    assert restored.selected.predictor_rank == 4
    np.testing.assert_allclose(
        restored.cv_mse_standard_error,
        _predictor_rank_profile().cv_mse_fold_sd / np.sqrt(4.0),
    )
