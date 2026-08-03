from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from pipls import (
    PiPLSComponentPath,
    PiPLSComponentResult,
    PiPLSDecomposition,
    PiPLSPredictorRankProfile,
    PiPLSValidationReport,
)


def _component_result() -> PiPLSComponentResult:
    return PiPLSComponentResult(
        n_components=np.int64(2),
        predictor_rank=np.int64(3),
        predictor_rank_policy="optimized",
        mean_test_score=np.float32(-0.4),
        cv_mse_mean=np.float32(0.4),
        cv_mse_fold_sd=np.float32(0.1),
        n_splits=np.int64(5),
    )


def _decomposition() -> PiPLSDecomposition:
    return PiPLSDecomposition(
        predictor_directions=np.eye(3, 2, dtype=np.float32),
        dilation=np.array([2.0, 1.0], dtype=np.float32),
        response_directions=np.eye(2, dtype=np.float32),
        predictor_numerical_rank=np.int64(3),
        predictor_numerical_rank_is_exact=np.bool_(True),
        rank_tolerance=np.float32(1e-6),
        predictor_svd_solver="full",
    )


def _validation_result() -> PiPLSComponentResult:
    return PiPLSComponentResult(
        n_components=np.int64(1),
        predictor_rank=np.int64(2),
        predictor_rank_policy="optimized",
        mean_test_score=np.float32(-0.5),
        cv_mse_mean=np.float32(0.5),
        cv_mse_fold_sd=np.float32(0.1),
        n_splits=np.int64(3),
    )


def _validation_report() -> PiPLSValidationReport:
    return PiPLSValidationReport(
        selected_result=_validation_result(),
        estimate_kind="selection-conditioned",
        is_leave_one_out=np.bool_(False),
        oof_predictions=np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]]),
        oof_prediction_counts=np.array([1, 0, 2], dtype=np.int64),
        pooled_oof_r2=np.float32(0.25),
    )


def test_cv_mse_standard_error_is_derived_not_stored_state() -> None:
    for result_type in (
        PiPLSComponentResult,
        PiPLSPredictorRankProfile,
        PiPLSComponentPath,
    ):
        assert "cv_mse_standard_error" not in {field.name for field in fields(result_type)}

    assert "one_standard_error_threshold" not in {
        field.name for field in fields(PiPLSComponentResult)
    }
    assert "selected_result" not in {
        field.name for field in fields(PiPLSPredictorRankProfile)
    }
    assert {
        "n_components",
        "predictor_rank",
        "n_splits",
        "mean_test_score",
        "cv_mse_mean",
    }.isdisjoint(field.name for field in fields(PiPLSValidationReport))


def test_component_result_validates_and_normalizes_python_scalars() -> None:
    result = _component_result()

    assert type(result.n_components) is int
    assert type(result.predictor_rank) is int
    assert type(result.mean_test_score) is float
    assert type(result.cv_mse_mean) is float
    assert type(result.cv_mse_fold_sd) is float
    assert type(result.cv_mse_standard_error) is float
    assert result.cv_mse_standard_error == pytest.approx(0.05)
    assert type(result.n_splits) is int
    assert result.rule is None
    assert result.reference_minimum is None
    assert result.one_standard_error_threshold is None
    with pytest.raises(FrozenInstanceError):
        result.n_components = 1  # type: ignore[misc]

    restored = pickle.loads(pickle.dumps(result))
    assert restored == result
    assert type(restored.predictor_rank) is int


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("n_components", 0, "positive integer"),
        ("predictor_rank", 1, "must not exceed"),
        ("predictor_rank_policy", "unknown", "must be one of"),
        ("mean_test_score", np.inf, "finite real"),
        ("cv_mse_mean", -0.1, "nonnegative"),
        ("cv_mse_fold_sd", np.nan, "finite real"),
        ("n_splits", 0, "positive integer"),
        ("rule", "unknown", "must be one of"),
    ],
)
def test_component_result_rejects_invalid_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = {
        "n_components": 2,
        "predictor_rank": 3,
        "predictor_rank_policy": "optimized",
        "mean_test_score": -0.4,
        "cv_mse_mean": 0.4,
        "cv_mse_fold_sd": 0.1,
        "n_splits": 5,
    }

    with pytest.raises(ValueError, match=message):
        PiPLSComponentResult(**{**kwargs, field: value})  # type: ignore[arg-type]


def test_component_result_records_one_standard_error_provenance() -> None:
    minimum = PiPLSComponentResult(
        n_components=3,
        predictor_rank=5,
        predictor_rank_policy="optimized",
        mean_test_score=-0.40,
        cv_mse_mean=0.40,
        cv_mse_fold_sd=0.08,
        n_splits=5,
        rule="minimum_cv_mse",
    )
    selection = PiPLSComponentResult(
        n_components=2,
        predictor_rank=4,
        predictor_rank_policy="optimized",
        mean_test_score=-0.43,
        cv_mse_mean=0.43,
        cv_mse_fold_sd=0.10,
        n_splits=5,
        rule="one_standard_error",
        reference_minimum=minimum,
    )

    assert selection.rule == "one_standard_error"
    assert selection.reference_minimum is minimum
    assert selection.one_standard_error_threshold == pytest.approx(0.44)

    restored = pickle.loads(pickle.dumps(selection))
    assert restored == selection
    assert restored.reference_minimum == minimum
    assert restored.one_standard_error_threshold == selection.one_standard_error_threshold


def test_component_result_rejects_invalid_selection_provenance() -> None:
    minimum = PiPLSComponentResult(
        n_components=3,
        predictor_rank=5,
        predictor_rank_policy="optimized",
        mean_test_score=-0.40,
        cv_mse_mean=0.40,
        cv_mse_fold_sd=0.08,
        n_splits=5,
        rule="minimum_cv_mse",
    )
    base = {
        "n_components": 2,
        "predictor_rank": 4,
        "predictor_rank_policy": "optimized",
        "mean_test_score": -0.43,
        "cv_mse_mean": 0.43,
        "cv_mse_fold_sd": 0.10,
        "n_splits": 5,
    }

    with pytest.raises(ValueError, match="required"):
        PiPLSComponentResult(**base, rule="one_standard_error")
    with pytest.raises(ValueError, match="defined only"):
        PiPLSComponentResult(**base, reference_minimum=minimum)
    with pytest.raises(TypeError, match="PiPLSComponentResult or None"):
        PiPLSComponentResult(
            **base,
            rule="one_standard_error",
            reference_minimum=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match='rule="minimum_cv_mse"'):
        PiPLSComponentResult(
            **base,
            rule="one_standard_error",
            reference_minimum=_component_result(),
        )
    with pytest.raises(ValueError, match="must not exceed the one-standard-error"):
        PiPLSComponentResult(
            **{**base, "cv_mse_mean": 0.45},
            rule="one_standard_error",
            reference_minimum=minimum,
        )


def test_path_records_reject_nonfinite_scores_and_noninteger_index_arrays() -> None:
    path_kwargs = {
        "n_components": [1, 2],
        "predictor_rank": [2, 3],
        "predictor_rank_policy": "optimized",
        "mean_test_score": [-0.5, -0.4],
        "cv_mse_mean": [0.5, 0.4],
        "cv_mse_fold_sd": [0.1, 0.1],
        "n_splits": 3,
    }
    with pytest.raises(ValueError, match="finite"):
        PiPLSComponentPath(**{**path_kwargs, "mean_test_score": [-0.5, np.nan]})
    with pytest.raises(ValueError, match="nonnegative"):
        PiPLSComponentPath(**{**path_kwargs, "cv_mse_mean": [0.5, -0.1]})
    with pytest.raises(ValueError, match="contain integers"):
        PiPLSComponentPath(**{**path_kwargs, "n_components": [1.0, 2.0]})
    one_split_path = PiPLSComponentPath(**{**path_kwargs, "n_splits": 1})
    with pytest.raises(ValueError, match="requires at least two"):
        _ = one_split_path.cv_mse_standard_error

    profile_kwargs = {
        "n_components": 2,
        "predictor_rank": [2, 3],
        "mean_test_score": [-0.5, -0.4],
        "cv_mse_mean": [0.5, 0.4],
        "cv_mse_fold_sd": [0.1, 0.1],
        "predictor_rank_policy": "optimized",
        "n_splits": 5,
    }
    with pytest.raises(ValueError, match="finite"):
        PiPLSPredictorRankProfile(
            **{**profile_kwargs, "mean_test_score": [-0.5, np.inf]}
        )
    with pytest.raises(ValueError, match="nonnegative"):
        PiPLSPredictorRankProfile(**{**profile_kwargs, "cv_mse_mean": [0.5, -0.1]})


def test_decomposition_uses_mathematical_direction_field_names() -> None:
    assert tuple(field.name for field in fields(PiPLSDecomposition)) == (
        "predictor_directions",
        "dilation",
        "response_directions",
        "predictor_numerical_rank",
        "predictor_numerical_rank_is_exact",
        "rank_tolerance",
        "predictor_svd_solver",
    )


def test_decomposition_makes_defensive_read_only_copies_and_revalidates_pickle() -> None:
    predictor_directions = np.eye(3, 2, dtype=np.float32)
    dilation = np.array([2.0, 1.0], dtype=np.float32)
    response_directions = np.eye(2, dtype=np.float32)
    decomposition = PiPLSDecomposition(
        predictor_directions=predictor_directions,
        dilation=dilation,
        response_directions=response_directions,
        predictor_numerical_rank=3,
        predictor_numerical_rank_is_exact=True,
        rank_tolerance=1e-12,
        predictor_svd_solver="full",
    )
    predictor_directions[0, 0] = 9.0
    dilation[0] = 9.0
    response_directions[0, 0] = 9.0

    assert decomposition.predictor_directions.dtype == np.dtype(np.float64)
    assert decomposition.dilation.dtype == np.dtype(np.float64)
    assert decomposition.response_directions.dtype == np.dtype(np.float64)
    assert decomposition.predictor_directions[0, 0] == 1.0
    assert decomposition.dilation[0] == 2.0
    assert decomposition.response_directions[0, 0] == 1.0
    assert all(
        not array.flags.writeable
        for array in (
            decomposition.predictor_directions,
            decomposition.dilation,
            decomposition.response_directions,
            decomposition.standardized_regression_map,
        )
    )

    restored = pickle.loads(pickle.dumps(decomposition))
    assert isinstance(restored, PiPLSDecomposition)
    assert not restored.predictor_directions.flags.writeable
    np.testing.assert_array_equal(restored.dilation, decomposition.dilation)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("predictor_directions", np.ones((3, 3)), "same number of components"),
        ("response_directions", np.ones((2, 1)), "same number of components"),
        ("dilation", [2.0, -1.0], "nonnegative"),
        ("dilation", [2.0, np.inf], "finite"),
        ("predictor_numerical_rank", 1, "not be smaller"),
        ("predictor_numerical_rank_is_exact", 1, "must be boolean"),
        ("rank_tolerance", -1.0, "nonnegative"),
        ("predictor_svd_solver", "auto", "must be one of"),
    ],
)
def test_decomposition_rejects_invalid_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = {
        "predictor_directions": np.eye(3, 2),
        "dilation": np.array([2.0, 1.0]),
        "response_directions": np.eye(2),
        "predictor_numerical_rank": 3,
        "predictor_numerical_rank_is_exact": True,
        "rank_tolerance": 1e-12,
        "predictor_svd_solver": "full",
    }

    with pytest.raises(ValueError, match=message):
        PiPLSDecomposition(**{**kwargs, field: value})  # type: ignore[arg-type]


def test_decomposition_requires_solver_and_rank_exactness_to_agree() -> None:
    with pytest.raises(ValueError, match="must be true"):
        PiPLSDecomposition(
            predictor_directions=np.eye(2),
            dilation=np.ones(2),
            response_directions=np.eye(2),
            predictor_numerical_rank=2,
            predictor_numerical_rank_is_exact=False,
            rank_tolerance=1e-12,
            predictor_svd_solver="full",
        )
    randomized = PiPLSDecomposition(
        predictor_directions=np.eye(2),
        dilation=np.ones(2),
        response_directions=np.eye(2),
        predictor_numerical_rank=2,
        predictor_numerical_rank_is_exact=False,
        rank_tolerance=1e-12,
        predictor_svd_solver="randomized",
    )
    assert randomized.predictor_svd_solver == "randomized"


def test_validation_report_normalizes_and_freezes_oof_arrays() -> None:
    report = _validation_report()

    assert report.selected_result == _validation_result()
    assert type(report.n_components) is int
    assert type(report.predictor_rank) is int
    assert type(report.n_splits) is int
    assert type(report.mean_test_score) is float
    assert type(report.cv_mse_mean) is float
    assert type(report.is_selection_conditioned) is bool
    assert type(report.has_complete_oof_coverage) is bool
    assert type(report.is_leave_one_out) is bool
    assert report.oof_predictions is not None
    assert report.oof_prediction_counts is not None
    assert not report.oof_predictions.flags.writeable
    assert not report.oof_prediction_counts.flags.writeable
    assert report.is_selection_conditioned
    assert not report.has_complete_oof_coverage

    restored = pickle.loads(pickle.dumps(report))
    assert isinstance(restored, PiPLSValidationReport)
    assert restored.selected_result == report.selected_result
    assert restored.oof_predictions is not None
    assert not restored.oof_predictions.flags.writeable
    np.testing.assert_array_equal(
        restored.oof_prediction_counts,
        report.oof_prediction_counts,
    )


def test_validation_report_requires_component_result() -> None:
    with pytest.raises(TypeError, match="selected_result must be a PiPLSComponentResult"):
        PiPLSValidationReport(
            selected_result=object(),  # type: ignore[arg-type]
            estimate_kind="selection-conditioned",
            is_leave_one_out=False,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("estimate_kind", "unknown", "must be one of"),
        ("is_leave_one_out", 1, "must be boolean"),
        ("pooled_oof_r2", np.inf, "finite real"),
    ],
)
def test_validation_report_rejects_invalid_scalar_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = {
        "selected_result": _validation_result(),
        "estimate_kind": "selection-conditioned",
        "is_leave_one_out": False,
        "oof_predictions": [1.0, 2.0],
        "oof_prediction_counts": [1, 1],
        "pooled_oof_r2": 0.2,
    }

    with pytest.raises(ValueError, match=message):
        PiPLSValidationReport(**{**kwargs, field: value})  # type: ignore[arg-type]


def test_validation_report_enforces_oof_coverage_representation() -> None:
    kwargs = {
        "selected_result": _validation_result(),
        "estimate_kind": "selection-conditioned",
        "is_leave_one_out": False,
    }
    with pytest.raises(ValueError, match="required"):
        PiPLSValidationReport(**kwargs, oof_predictions=[1.0, 2.0])
    with pytest.raises(ValueError, match="nonnegative"):
        PiPLSValidationReport(
            **kwargs,
            oof_predictions=[1.0, 2.0],
            oof_prediction_counts=[1, -1],
        )
    with pytest.raises(ValueError, match="Covered OOF predictions must be finite"):
        PiPLSValidationReport(
            **kwargs,
            oof_predictions=[np.nan, 2.0],
            oof_prediction_counts=[1, 1],
        )
    with pytest.raises(ValueError, match="Uncovered OOF predictions must be NaN"):
        PiPLSValidationReport(
            **kwargs,
            oof_predictions=[1.0, 2.0],
            oof_prediction_counts=[0, 1],
        )
    with pytest.raises(ValueError, match="contain integers"):
        PiPLSValidationReport(
            **kwargs,
            oof_predictions=[1.0, 2.0],
            oof_prediction_counts=[1.0, 1.0],
        )
    with pytest.raises(ValueError, match="at least one row"):
        PiPLSValidationReport(
            **kwargs,
            oof_predictions=[],
            oof_prediction_counts=[],
        )
    with pytest.raises(ValueError, match="at least two rows"):
        PiPLSValidationReport(
            **kwargs,
            oof_predictions=[1.0, np.nan],
            oof_prediction_counts=[1, 0],
            pooled_oof_r2=0.0,
        )
