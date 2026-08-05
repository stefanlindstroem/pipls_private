from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pipls.component_path import (
    PiPLSComponentPath,
    PiPLSPredictorRankEvidence,
    PiPLSPredictorRankProfile,
    PiPLSSelection,
)


def _component_path() -> PiPLSComponentPath:
    return PiPLSComponentPath(
        n_components=np.array([1, 2, 4], dtype=np.int64),
        predictor_rank=np.array([3, 4, 5], dtype=np.int64),
        predictor_rank_policy="optimized",
        mean_test_score=np.array([-0.8, -0.5, -0.45], dtype=np.float32),
        cv_mse_mean=np.array([0.8, 0.5, 0.45], dtype=np.float32),
        cv_mse_std=np.array([0.1, 0.08, 0.07], dtype=np.float32),
        n_splits=np.int64(5),
    )


def test_component_path_makes_aligned_read_only_defensive_copies() -> None:
    n_components = np.array([1, 2], dtype=np.int64)
    predictor_rank = np.array([2, 3], dtype=np.int64)
    mean_test_score = np.array([-0.6, -0.4], dtype=np.float32)
    cv_mse_mean = np.array([0.6, 0.4], dtype=np.float32)
    cv_mse_std = np.array([0.2, 0.1], dtype=np.float32)

    path = PiPLSComponentPath(
        n_components=n_components,
        predictor_rank=predictor_rank,
        predictor_rank_policy="fixed",
        mean_test_score=mean_test_score,
        cv_mse_mean=cv_mse_mean,
        cv_mse_std=cv_mse_std,
        n_splits=np.int64(4),
    )
    n_components[0] = 99
    predictor_rank[0] = 99
    mean_test_score[0] = 99.0
    cv_mse_mean[0] = 99.0
    cv_mse_std[0] = 99.0

    arrays = (
        path.n_components,
        path.predictor_rank,
        path.mean_test_score,
        path.cv_mse_mean,
        path.cv_mse_std,
    )
    assert all(array.shape == (2,) for array in arrays)
    assert path.n_components.dtype == np.dtype(np.intp)
    assert path.predictor_rank.dtype == np.dtype(np.intp)
    assert path.mean_test_score.dtype == np.dtype(np.float64)
    assert path.cv_mse_mean.dtype == np.dtype(np.float64)
    assert path.cv_mse_std.dtype == np.dtype(np.float64)
    assert not hasattr(path, "cv_mse_fold_sd")
    assert not hasattr(path, "cv_mse_standard_error")
    assert all(not array.flags.writeable for array in arrays)
    assert path.predictor_rank_policy == "fixed"
    assert path.n_splits == 4
    assert type(path.n_splits) is int
    np.testing.assert_array_equal(path.n_components, np.array([1, 2]))
    np.testing.assert_array_equal(path.predictor_rank, np.array([2, 3]))
    np.testing.assert_allclose(path.mean_test_score, np.array([-0.6, -0.4]))
    np.testing.assert_allclose(path.cv_mse_mean, np.array([0.6, 0.4]))
    np.testing.assert_allclose(path.cv_mse_std, np.array([0.2, 0.1]))

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
        "cv_mse_std": [0.2, 0.1],
        "n_splits": 4,
    }

    with pytest.raises(ValueError, match="same length"):
        PiPLSComponentPath(**{**kwargs, "predictor_rank": [2]})
    with pytest.raises(ValueError, match="strictly ascending"):
        PiPLSComponentPath(**{**kwargs, "n_components": [2, 1]})
    with pytest.raises(ValueError, match="must be one of"):
        PiPLSComponentPath(**{**kwargs, "predictor_rank_policy": "unknown"})
    one_split_path = PiPLSComponentPath(**{**kwargs, "n_splits": 1})
    assert not hasattr(one_split_path, "cv_mse_standard_error")


def test_component_path_is_pickleable_with_read_only_arrays() -> None:
    restored = pickle.loads(pickle.dumps(_component_path()))

    assert isinstance(restored, PiPLSComponentPath)
    np.testing.assert_array_equal(restored.n_components, np.array([1, 2, 4]))
    assert not restored.n_components.flags.writeable
    assert restored.predictor_rank_policy == "optimized"
    assert restored.n_splits == 5
    np.testing.assert_array_equal(restored.predictor_rank, np.array([3, 4, 5]))
    np.testing.assert_allclose(restored.mean_test_score, np.array([-0.8, -0.5, -0.45]))
    np.testing.assert_allclose(restored.cv_mse_mean, np.array([0.8, 0.5, 0.45]))
    np.testing.assert_allclose(restored.cv_mse_std, np.array([0.1, 0.08, 0.07]))
    assert not hasattr(restored, "cv_mse_standard_error")


def _predictor_rank_profile() -> PiPLSPredictorRankProfile:
    return PiPLSPredictorRankProfile(
        n_components=2,
        predictor_rank=np.array([2, 3, 4], dtype=np.int64),
        mean_test_score=np.array([-0.6, -0.5, -0.4], dtype=np.float32),
        cv_mse_mean=np.array([0.6, 0.5, 0.4], dtype=np.float32),
        cv_mse_std=np.array([0.10, 0.09, 0.08], dtype=np.float32),
        predictor_rank_policy="optimized",
        n_splits=5,
    )


def test_predictor_rank_profile_makes_read_only_defensive_copies() -> None:
    predictor_rank = np.array([2, 3], dtype=np.int64)
    mean_test_score = np.array([-0.6, -0.4], dtype=np.float32)
    cv_mse_mean = np.array([0.6, 0.4], dtype=np.float32)
    cv_mse_std = np.array([0.2, 0.1], dtype=np.float32)

    profile = PiPLSPredictorRankProfile(
        n_components=np.int64(2),
        predictor_rank=predictor_rank,
        mean_test_score=mean_test_score,
        cv_mse_mean=cv_mse_mean,
        cv_mse_std=cv_mse_std,
        predictor_rank_policy="optimized",
        n_splits=np.int64(4),
    )
    predictor_rank[0] = 99
    mean_test_score[0] = 99.0
    cv_mse_mean[0] = 99.0
    cv_mse_std[0] = 99.0

    arrays = (
        profile.predictor_rank,
        profile.mean_test_score,
        profile.cv_mse_mean,
        profile.cv_mse_std,
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
    assert not hasattr(profile, "cv_mse_fold_sd")
    assert not hasattr(profile, "cv_mse_standard_error")
    assert profile.selection.predictor_rank == 3

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
        "cv_mse_std": profile.cv_mse_std,
        "predictor_rank_policy": profile.predictor_rank_policy,
        "n_splits": profile.n_splits,
    }

    with pytest.raises(ValueError, match="same length"):
        PiPLSPredictorRankProfile(**{**kwargs, "cv_mse_mean": [0.6, 0.5]})
    one_split_profile = PiPLSPredictorRankProfile(
        **{
            **kwargs,
            "cv_mse_std": [0.10, 0.09, 0.0],
            "n_splits": 1,
        }
    )
    assert not hasattr(one_split_profile, "cv_mse_standard_error")
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
        cv_mse_std=[0.10, 0.09, 0.08],
        predictor_rank_policy="optimized",
        n_splits=5,
    )

    selected = profile.selection

    assert selected == PiPLSSelection(
        n_components=2,
        predictor_rank=2,
        predictor_rank_policy="optimized",
        mean_test_score=1.0,
        cv_mse_mean=0.6,
        cv_mse_std=0.10,
        n_splits=5,
    )


def test_predictor_rank_profile_is_pickleable_with_read_only_arrays() -> None:
    restored = pickle.loads(pickle.dumps(_predictor_rank_profile()))

    assert isinstance(restored, PiPLSPredictorRankProfile)
    np.testing.assert_array_equal(restored.predictor_rank, np.array([2, 3, 4]))
    assert not restored.predictor_rank.flags.writeable
    assert restored.predictor_rank_policy == "optimized"
    assert restored.selection.predictor_rank == 4
    assert not hasattr(restored, "cv_mse_standard_error")


def test_predictor_rank_evidence_is_immutable_validated_and_pickleable() -> None:
    evidence = PiPLSPredictorRankEvidence(
        reference_predictor_rank=np.int64(3),
        reference_mean_test_score=np.float32(-0.40),
        reference_cv_mse_mean=np.float32(0.40),
        reference_cv_mse_std=np.float32(0.08),
        relative_tolerance=np.float32(0.10),
        absolute_tolerance=np.inf,
    )

    assert type(evidence.reference_predictor_rank) is int
    assert type(evidence.reference_mean_test_score) is float
    assert evidence.score_threshold == pytest.approx(-0.44)
    with pytest.raises(FrozenInstanceError):
        evidence.reference_predictor_rank = 2  # type: ignore[misc]
    assert pickle.loads(pickle.dumps(evidence)) == evidence


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference_predictor_rank", 0, "positive integer"),
        ("reference_mean_test_score", np.inf, "finite real"),
        ("reference_cv_mse_mean", -0.1, "nonnegative"),
        ("reference_cv_mse_std", np.nan, "finite real"),
        ("relative_tolerance", -0.1, "finite nonnegative"),
        ("absolute_tolerance", -np.inf, "nonnegative real"),
    ],
)
def test_predictor_rank_evidence_rejects_invalid_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = {
        "reference_predictor_rank": 3,
        "reference_mean_test_score": -0.4,
        "reference_cv_mse_mean": 0.4,
        "reference_cv_mse_std": 0.08,
        "relative_tolerance": 0.1,
        "absolute_tolerance": np.inf,
    }
    with pytest.raises(ValueError, match=message):
        PiPLSPredictorRankEvidence(**{**kwargs, field: value})


def test_predictor_rank_profile_exposes_exact_and_tolerant_selections() -> None:
    evidence = PiPLSPredictorRankEvidence(
        reference_predictor_rank=3,
        reference_mean_test_score=1.0,
        reference_cv_mse_mean=0.40,
        reference_cv_mse_std=0.08,
        relative_tolerance=0.10,
        absolute_tolerance=np.inf,
    )
    profile = PiPLSPredictorRankProfile(
        n_components=1,
        predictor_rank=[1, 2, 3],
        mean_test_score=[0.91, 0.95, 1.0],
        cv_mse_mean=[0.50, 0.45, 0.40],
        cv_mse_std=[0.10, 0.09, 0.08],
        predictor_rank_policy="optimized",
        n_splits=5,
        predictor_rank_evidence=evidence,
    )

    assert profile.reference_selection.predictor_rank == 3
    assert profile.reference_selection.predictor_rank_evidence is None
    assert profile.selection.predictor_rank == 1
    assert profile.selection.predictor_rank_evidence is evidence


def test_component_path_aligns_predictor_rank_evidence_with_rows() -> None:
    evidence = (
        PiPLSPredictorRankEvidence(2, 1.0, 0.4, 0.08, 0.1, np.inf),
        PiPLSPredictorRankEvidence(3, 0.9, 0.5, 0.09, 0.1, np.inf),
    )
    path = PiPLSComponentPath(
        n_components=[1, 2],
        predictor_rank=[1, 2],
        predictor_rank_policy="optimized",
        mean_test_score=[0.91, 0.9],
        cv_mse_mean=[0.45, 0.5],
        cv_mse_std=[0.09, 0.09],
        n_splits=5,
        predictor_rank_evidence=evidence,
    )

    assert isinstance(path.predictor_rank_evidence, tuple)
    assert path._selection_at_index(0).predictor_rank_evidence is evidence[0]
    restored = pickle.loads(pickle.dumps(path))
    assert restored.predictor_rank_evidence == evidence

    with pytest.raises(ValueError, match="one-for-one"):
        PiPLSComponentPath(
            n_components=[1, 2],
            predictor_rank=[1, 2],
            predictor_rank_policy="optimized",
            mean_test_score=[0.91, 0.9],
            cv_mse_mean=[0.45, 0.5],
            cv_mse_std=[0.09, 0.09],
            n_splits=5,
            predictor_rank_evidence=evidence[:1],
        )


def test_predictor_rank_evidence_rejects_selection_beyond_reference_rank() -> None:
    evidence = PiPLSPredictorRankEvidence(2, 1.0, 0.4, 0.08, 0.1, np.inf)

    with pytest.raises(ValueError, match="must not exceed"):
        PiPLSSelection(
            n_components=1,
            predictor_rank=3,
            predictor_rank_policy="optimized",
            mean_test_score=0.95,
            cv_mse_mean=0.45,
            cv_mse_std=0.09,
            n_splits=5,
            predictor_rank_evidence=evidence,
        )

    with pytest.raises(ValueError, match="must not exceed"):
        PiPLSComponentPath(
            n_components=[1],
            predictor_rank=[3],
            predictor_rank_policy="optimized",
            mean_test_score=[0.95],
            cv_mse_mean=[0.45],
            cv_mse_std=[0.09],
            n_splits=5,
            predictor_rank_evidence=(evidence,),
        )
