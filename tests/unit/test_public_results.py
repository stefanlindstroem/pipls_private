from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError, fields
from types import ModuleType

import numpy as np
import pytest

import pipls
import pipls.exceptions as exceptions_module
import pipls.regression as regression_module
import pipls.search as search_module
from pipls.component_path import (
    PiPLSComponentPath,
    PiPLSPredictorRankProfile,
    PiPLSSelection,
)
from pipls.decomposition import PiPLSDecomposition
from pipls.validation import PiPLSOOFReport


def _selection() -> PiPLSSelection:
    return PiPLSSelection(
        n_components=np.int64(2),
        predictor_rank=np.int64(3),
        predictor_rank_policy="optimized",
        mean_test_score=np.float32(-0.4),
        cv_mse_mean=np.float32(0.4),
        cv_mse_std=np.float32(0.1),
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


def _validation_result() -> PiPLSSelection:
    return PiPLSSelection(
        n_components=np.int64(1),
        predictor_rank=np.int64(2),
        predictor_rank_policy="optimized",
        mean_test_score=np.float32(-0.5),
        cv_mse_mean=np.float32(0.5),
        cv_mse_std=np.float32(0.1),
        n_splits=np.int64(3),
    )


@pytest.mark.parametrize(
    ("module", "expected_name"),
    [
        (regression_module, "PiPLSRegression"),
        (search_module, "PiPLSSearchCV"),
        (exceptions_module, "PredictorRankSupportWarning"),
    ],
)
def test_primary_modules_declare_exact_exports(
    module: ModuleType,
    expected_name: str,
) -> None:
    namespace: dict[str, object] = {}
    exec(f"from {module.__name__} import *", namespace)

    assert module.__all__ == [expected_name]
    assert {name for name in namespace if name != "__builtins__"} == {expected_name}


def test_result_records_are_public_only_from_focused_modules() -> None:
    removed_top_level_names = (
        "PiPLSComponentPath",
        "PiPLSSelection",
        "PiPLSPredictorRankProfile",
        "PiPLSDecomposition",
        "PiPLSOOFReport",
    )

    assert set(pipls.__all__) == {
        "PiPLSRegression",
        "PiPLSSearchCV",
        "PredictorRankSupportWarning",
        "__version__",
    }
    assert all(not hasattr(pipls, name) for name in removed_top_level_names)


def test_selection_terminology_has_no_pre_release_aliases() -> None:
    profile = PiPLSPredictorRankProfile(
        n_components=1,
        predictor_rank=np.array([1]),
        mean_test_score=np.array([-0.5]),
        cv_mse_mean=np.array([0.5]),
        cv_mse_std=np.array([0.1]),
        predictor_rank_policy="optimized",
        n_splits=3,
    )

    assert not hasattr(pipls, "PiPLSComponentResult")
    assert not hasattr(profile, "selected_result")
    assert isinstance(profile.selection, PiPLSSelection)



def test_derived_result_properties_are_not_stored_state() -> None:
    for result_type in (
        PiPLSSelection,
        PiPLSPredictorRankProfile,
        PiPLSComponentPath,
    ):
        assert "cv_mse_standard_error" not in {field.name for field in fields(result_type)}

    assert "cv_mse_threshold" not in {
        field.name for field in fields(PiPLSSelection)
    }
    assert "selection" not in {
        field.name for field in fields(PiPLSPredictorRankProfile)
    }
    derived_report_fields = {
        "n_components",
        "predictor_rank",
        "n_splits",
        "mean_test_score",
        "cv_mse_mean",
        "has_complete_oof_coverage",
    }
    assert derived_report_fields.isdisjoint(
        field.name for field in fields(PiPLSOOFReport)
    )


def test_selection_validates_and_normalizes_python_scalars() -> None:
    result = _selection()

    assert type(result.n_components) is int
    assert type(result.predictor_rank) is int
    assert type(result.mean_test_score) is float
    assert type(result.cv_mse_mean) is float
    assert type(result.cv_mse_std) is float
    assert not hasattr(result, "cv_mse_fold_sd")
    assert not hasattr(result, "cv_mse_standard_error")
    assert type(result.n_splits) is int
    assert result.rule is None
    assert result.reference_minimum is None
    assert result.relative_tolerance is None
    assert result.absolute_tolerance is None
    assert result.cv_mse_threshold is None
    assert not hasattr(result, "one_standard_error_threshold")
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
        ("cv_mse_std", np.nan, "finite real"),
        ("n_splits", 0, "positive integer"),
        ("rule", "unknown", "must be one of"),
        ("rule", "one_standard_error", "must be one of"),
    ],
)
def test_selection_rejects_invalid_fields(
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
        "cv_mse_std": 0.1,
        "n_splits": 5,
    }

    with pytest.raises(ValueError, match=message):
        PiPLSSelection(**{**kwargs, field: value})  # type: ignore[arg-type]


def test_selection_records_tolerance_provenance() -> None:
    minimum = PiPLSSelection(
        n_components=3,
        predictor_rank=5,
        predictor_rank_policy="optimized",
        mean_test_score=-0.40,
        cv_mse_mean=0.40,
        cv_mse_std=0.08,
        n_splits=5,
    )
    selection = PiPLSSelection(
        n_components=2,
        predictor_rank=4,
        predictor_rank_policy="optimized",
        mean_test_score=-0.43,
        cv_mse_mean=0.43,
        cv_mse_std=0.10,
        n_splits=5,
        rule="minimum_cv_mse",
        reference_minimum=minimum,
        relative_tolerance=0.10,
        absolute_tolerance=np.inf,
    )

    assert selection.rule == "minimum_cv_mse"
    assert selection.reference_minimum is minimum
    assert selection.relative_tolerance == pytest.approx(0.10)
    assert np.isposinf(selection.absolute_tolerance)
    assert selection.cv_mse_threshold == pytest.approx(0.44)
    assert not hasattr(selection, "one_standard_error_threshold")

    restored = pickle.loads(pickle.dumps(selection))
    assert restored == selection
    assert restored.reference_minimum == minimum
    assert restored.cv_mse_threshold == selection.cv_mse_threshold

def test_selection_rejects_invalid_selection_provenance() -> None:
    minimum = PiPLSSelection(
        n_components=3,
        predictor_rank=5,
        predictor_rank_policy="optimized",
        mean_test_score=-0.40,
        cv_mse_mean=0.40,
        cv_mse_std=0.08,
        n_splits=5,
    )
    base = {
        "n_components": 2,
        "predictor_rank": 4,
        "predictor_rank_policy": "optimized",
        "mean_test_score": -0.43,
        "cv_mse_mean": 0.43,
        "cv_mse_std": 0.10,
        "n_splits": 5,
    }

    with pytest.raises(ValueError, match="reference_minimum is required"):
        PiPLSSelection(
            **base,
            rule="minimum_cv_mse",
            relative_tolerance=0.10,
            absolute_tolerance=np.inf,
        )
    with pytest.raises(ValueError, match="required for.*minimum_cv_mse"):
        PiPLSSelection(
            **base,
            rule="minimum_cv_mse",
            reference_minimum=minimum,
        )
    with pytest.raises(ValueError, match="effective CV-MSE threshold"):
        PiPLSSelection(
            **{**base, "cv_mse_mean": 0.45},
            rule="minimum_cv_mse",
            reference_minimum=minimum,
            relative_tolerance=0.10,
            absolute_tolerance=np.inf,
        )
    with pytest.raises(ValueError, match="defined only"):
        PiPLSSelection(**base, reference_minimum=minimum)
    with pytest.raises(TypeError, match="PiPLSSelection or None"):
        PiPLSSelection(
            **base,
            rule="minimum_cv_mse",
            reference_minimum=object(),  # type: ignore[arg-type]
            relative_tolerance=0.10,
            absolute_tolerance=np.inf,
        )
    with pytest.raises(ValueError, match="unruled path row"):
        ruled_reference = PiPLSSelection(
            **base,
            rule="minimum_cv_mse",
            reference_minimum=minimum,
            relative_tolerance=0.10,
            absolute_tolerance=np.inf,
        )
        PiPLSSelection(
            **base,
            rule="minimum_cv_mse",
            reference_minimum=ruled_reference,
            relative_tolerance=0.10,
            absolute_tolerance=np.inf,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("relative_tolerance", -0.1, "finite nonnegative"),
        ("relative_tolerance", np.inf, "finite nonnegative"),
        ("relative_tolerance", True, "finite nonnegative"),
        ("absolute_tolerance", -0.1, "nonnegative real"),
        ("absolute_tolerance", np.nan, "nonnegative real"),
        ("absolute_tolerance", -np.inf, "nonnegative real"),
        ("absolute_tolerance", False, "nonnegative real"),
    ],
)
def test_selection_rejects_invalid_tolerance_provenance(
    field: str,
    value: object,
    message: str,
) -> None:
    minimum = _selection()
    kwargs = {
        "n_components": 2,
        "predictor_rank": 3,
        "predictor_rank_policy": "optimized",
        "mean_test_score": -0.4,
        "cv_mse_mean": 0.4,
        "cv_mse_std": 0.1,
        "n_splits": 5,
        "rule": "minimum_cv_mse",
        "reference_minimum": minimum,
        "relative_tolerance": 0.0,
        "absolute_tolerance": np.inf,
    }

    with pytest.raises(ValueError, match=message):
        PiPLSSelection(**{**kwargs, field: value})  # type: ignore[arg-type]


def test_path_records_reject_nonfinite_scores_and_noninteger_index_arrays() -> None:
    path_kwargs = {
        "n_components": [1, 2],
        "predictor_rank": [2, 3],
        "predictor_rank_policy": "optimized",
        "mean_test_score": [-0.5, -0.4],
        "cv_mse_mean": [0.5, 0.4],
        "cv_mse_std": [0.1, 0.1],
        "n_splits": 3,
    }
    with pytest.raises(ValueError, match="finite"):
        PiPLSComponentPath(**{**path_kwargs, "mean_test_score": [-0.5, np.nan]})
    with pytest.raises(ValueError, match="nonnegative"):
        PiPLSComponentPath(**{**path_kwargs, "cv_mse_mean": [0.5, -0.1]})
    with pytest.raises(ValueError, match="contain integers"):
        PiPLSComponentPath(**{**path_kwargs, "n_components": [1.0, 2.0]})
    one_split_path = PiPLSComponentPath(**{**path_kwargs, "n_splits": 1})
    assert not hasattr(one_split_path, "cv_mse_standard_error")

    profile_kwargs = {
        "n_components": 2,
        "predictor_rank": [2, 3],
        "mean_test_score": [-0.5, -0.4],
        "cv_mse_mean": [0.5, 0.4],
        "cv_mse_std": [0.1, 0.1],
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



def test_oof_report_normalizes_and_freezes_arrays() -> None:
    selection = _validation_result()
    report = PiPLSOOFReport(
        selection=selection,
        is_leave_one_out=np.bool_(False),
        oof_predictions=np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]]),
        oof_prediction_counts=np.array([1, 0, 2], dtype=np.int64),
        pooled_oof_r2=np.float32(0.25),
    )

    assert report.selection == selection
    assert type(report.has_complete_oof_coverage) is bool
    assert type(report.is_leave_one_out) is bool
    assert not report.oof_predictions.flags.writeable
    assert not report.oof_prediction_counts.flags.writeable
    assert not report.has_complete_oof_coverage

    restored = pickle.loads(pickle.dumps(report))
    assert isinstance(restored, PiPLSOOFReport)
    assert restored.selection == report.selection
    assert not restored.oof_predictions.flags.writeable
    np.testing.assert_array_equal(
        restored.oof_prediction_counts,
        report.oof_prediction_counts,
    )


def test_oof_report_requires_selection() -> None:
    with pytest.raises(TypeError, match="selection must be a PiPLSSelection"):
        PiPLSOOFReport(
            selection=object(),  # type: ignore[arg-type]
            is_leave_one_out=False,
            oof_predictions=[1.0, 2.0],  # type: ignore[arg-type]
            oof_prediction_counts=[1, 1],  # type: ignore[arg-type]
        )


def test_oof_report_enforces_oof_coverage_representation() -> None:
    kwargs = {
        "selection": _validation_result(),
        "is_leave_one_out": False,
    }
    with pytest.raises(TypeError):
        PiPLSOOFReport(**kwargs)
    with pytest.raises(ValueError, match="oof_predictions are required"):
        PiPLSOOFReport(
            **kwargs,
            oof_predictions=None,  # type: ignore[arg-type]
            oof_prediction_counts=[1, 1],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="oof_prediction_counts are required"):
        PiPLSOOFReport(
            **kwargs,
            oof_predictions=[1.0, 2.0],  # type: ignore[arg-type]
            oof_prediction_counts=None,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="nonnegative"):
        PiPLSOOFReport(
            **kwargs,
            oof_predictions=[1.0, 2.0],
            oof_prediction_counts=[1, -1],
        )
    with pytest.raises(ValueError, match="Covered OOF predictions must be finite"):
        PiPLSOOFReport(
            **kwargs,
            oof_predictions=[np.nan, 2.0],
            oof_prediction_counts=[1, 1],
        )
    with pytest.raises(ValueError, match="Uncovered OOF predictions must be NaN"):
        PiPLSOOFReport(
            **kwargs,
            oof_predictions=[1.0, 2.0],
            oof_prediction_counts=[0, 1],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("is_leave_one_out", 1, "must be boolean"),
        ("pooled_oof_r2", np.inf, "finite real"),
    ],
)
def test_oof_report_rejects_invalid_scalar_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = {
        "selection": _validation_result(),
        "is_leave_one_out": False,
        "oof_predictions": [1.0, 2.0],
        "oof_prediction_counts": [1, 1],
        "pooled_oof_r2": 0.2,
    }

    with pytest.raises(ValueError, match=message):
        PiPLSOOFReport(**{**kwargs, field: value})  # type: ignore[arg-type]
