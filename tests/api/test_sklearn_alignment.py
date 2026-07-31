from __future__ import annotations

import importlib.util
import inspect
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

import pipls
from pipls import (
    PiPLSComponentPath,
    PiPLSDecomposition,
    PiPLSRegression,
    PiPLSSearchCV,
)


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260718)
    X = rng.normal(size=(48, 7))
    Y = X @ rng.normal(size=(7, 3)) + 0.05 * rng.normal(size=(48, 3))
    return X, Y


def _fixed_estimator() -> PiPLSRegression:
    return PiPLSRegression(
        n_components=2,
        predictor_rank=3,
        scale=False,
        svd_solver="full",
        random_state=None,
    )


def test_search_cv_is_the_only_public_selection_class_name() -> None:
    assert pipls.PiPLSSearchCV is PiPLSSearchCV
    assert importlib.util.find_spec("pipls.search") is not None
    assert importlib.util.find_spec("pipls.path") is None
    assert not hasattr(pipls, "PiPLSPathCV")


def test_fixed_regression_and_path_configuration_have_distinct_ownership() -> None:
    regression = _fixed_estimator()
    path = PiPLSSearchCV()

    assert regression.n_components == 2
    assert regression.predictor_rank == 3
    assert not hasattr(regression, "cv")
    assert not hasattr(regression, "samples_per_predictor_rank")
    assert path.samples_per_predictor_rank == 5.0
    assert path.cv == 5
    assert path.n_components_values == "all"
    assert path.scoring == "neg_response_standardized_mean_squared_error"
    assert path.selection_rule == "best_score"
    assert path.refit is False


def test_path_defaults_have_stable_signature_and_repr() -> None:
    signature = inspect.signature(PiPLSSearchCV)

    assert signature.parameters["scoring"].default == (
        "neg_response_standardized_mean_squared_error"
    )
    assert signature.parameters["refit"].default is False
    assert signature.parameters["selection_rule"].default == "best_score"
    assert "0x" not in str(signature)
    path = PiPLSSearchCV()
    assert repr(path) == "PiPLSSearchCV()"
    assert clone(path).scoring == path.scoring
    assert clone(path).selection_rule == "best_score"
    assert clone(path).refit is False


def test_fixed_regression_constructor_matches_direct_estimator_scope() -> None:
    signature = inspect.signature(PiPLSRegression)
    for name in ("n_components", "predictor_rank"):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty

    assert set(_fixed_estimator().get_params()) == {
        "copy",
        "n_components",
        "predictor_rank",
        "random_state",
        "scale",
        "svd_solver",
    }


def test_path_constructor_has_no_redundant_pipeline_prefix_parameter() -> None:
    assert set(PiPLSSearchCV().get_params(deep=False)) == {
        "cv",
        "estimator",
        "max_predictor_rank",
        "n_components_values",
        "n_jobs",
        "predictor_rank_values",
        "refit",
        "return_oof_predictions",
        "samples_per_predictor_rank",
        "scoring",
        "selection_rule",
        "search_method",
    }


def test_path_output_configuration_belongs_to_estimator_template() -> None:
    path = PiPLSSearchCV()

    assert not hasattr(path, "set_output")
    assert hasattr(_fixed_estimator(), "set_output")


def test_fixed_estimator_interoperates_with_grid_search_for_explicit_pairs() -> None:
    X, Y = _data()
    search = GridSearchCV(
        _fixed_estimator(),
        param_grid=[
            {"n_components": [1], "predictor_rank": [2]},
            {"n_components": [2], "predictor_rank": [3]},
        ],
        cv=3,
    ).fit(X, Y)

    assert isinstance(search.best_estimator_, PiPLSRegression)
    assert search.best_estimator_.predictor_rank in (2, 3)
    assert not hasattr(search.best_estimator_, "cv_results_")


def test_pls_style_method_signatures_include_copy_controls() -> None:
    assert tuple(inspect.signature(PiPLSRegression.predict).parameters) == (
        "self",
        "X",
        "copy",
    )
    assert tuple(inspect.signature(PiPLSRegression.transform).parameters) == (
        "self",
        "X",
        "y",
        "copy",
    )
    assert tuple(inspect.signature(PiPLSRegression.fit_transform).parameters) == (
        "self",
        "X",
        "y",
    )


def test_fit_transform_matches_pls_style_pair_of_scores() -> None:
    X, Y = _data()
    model = _fixed_estimator()

    x_scores, y_scores = model.fit_transform(X, Y)
    transformed_x, transformed_y = model.transform(X, Y)

    np.testing.assert_allclose(x_scores, transformed_x)
    np.testing.assert_allclose(y_scores, transformed_y)
    np.testing.assert_allclose(x_scores, model.x_scores_)
    np.testing.assert_allclose(y_scores, model.y_scores_)


def test_standard_pls_style_attributes_have_documented_meaning() -> None:
    X, Y = _data()
    model = _fixed_estimator().fit(X, Y)
    X_cs = (X - model.x_mean_) / model.x_scale_
    Y_cs = (Y - model.y_mean_) / model.y_scale_

    np.testing.assert_allclose(model.x_scores_, X_cs @ model.x_rotations_)
    np.testing.assert_allclose(model.y_scores_, Y_cs @ model.y_rotations_)
    assert model.x_loadings_.shape == (X.shape[1], model.n_components)
    assert model.y_loadings_.shape == (Y.shape[1], model.n_components)

    x_residual = X_cs - model.x_scores_ @ model.x_loadings_.T
    y_residual = Y_cs - model.y_scores_ @ model.y_loadings_.T
    np.testing.assert_allclose(model.x_scores_.T @ x_residual, 0.0, atol=1e-10)
    np.testing.assert_allclose(model.y_scores_.T @ y_residual, 0.0, atol=1e-10)


def test_public_decomposition_is_read_only_and_replaces_symbolic_aliases() -> None:
    X, Y = _data()
    model = _fixed_estimator().fit(X, Y)
    decomposition = model.decomposition_

    assert isinstance(decomposition, PiPLSDecomposition)
    for matrix in (
        decomposition.predictor_rotations,
        decomposition.dilation,
        decomposition.response_rotations,
    ):
        assert not matrix.flags.writeable
    standardized_map = decomposition.standardized_regression_map
    assert not standardized_map.flags.writeable
    np.testing.assert_allclose(
        standardized_map,
        (decomposition.predictor_rotations * decomposition.dilation[None, :])
        @ decomposition.response_rotations.T,
    )
    assert model.x_rotations_ is decomposition.predictor_rotations
    assert model.y_rotations_ is decomposition.response_rotations

    for removed_alias in (
        "Pi_",
        "C_",
        "W_",
        "P_",
        "D_",
        "Q_",
        "dilation_",
        "coef_matrix_",
        "x_rank_",
        "x_rank_is_exact_",
        "rank_tolerance_",
        "svd_solver_",
    ):
        assert not hasattr(model, removed_alias)

    for removed_field in (
        "Pi",
        "C",
        "W",
        "P",
        "D",
        "Q",
        "predictor_basis",
        "response_basis",
        "least_squares_map",
        "dilation_matrix",
        "regression_map",
        "x_rank",
        "x_rank_is_exact",
    ):
        assert not hasattr(decomposition, removed_field)

    with pytest.raises(ValueError, match="read-only"):
        decomposition.predictor_rotations[0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        decomposition.predictor_numerical_rank = 0  # type: ignore[misc]


def test_copy_parameter_matches_pls_fit_semantics() -> None:
    X, Y = _data()
    X_original = X.copy()
    Y_original = Y.copy()
    _fixed_estimator().set_params(copy=True).fit(X, Y)
    np.testing.assert_array_equal(X, X_original)
    np.testing.assert_array_equal(Y, Y_original)

    X_mutable = X_original.copy()
    Y_mutable = Y_original.copy()
    _fixed_estimator().set_params(copy=False).fit(X_mutable, Y_mutable)
    assert not np.array_equal(X_mutable, X_original)
    assert not np.array_equal(Y_mutable, Y_original)
    np.testing.assert_allclose(np.mean(X_mutable, axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.mean(Y_mutable, axis=0), 0.0, atol=1e-14)


def test_feature_names_and_set_output_match_sklearn_transformers() -> None:
    X, Y = _data()
    columns = [f"feature_{index}" for index in range(X.shape[1])]
    X_frame = pd.DataFrame(X, columns=columns)

    model = _fixed_estimator().fit(X_frame, Y)
    np.testing.assert_array_equal(model.feature_names_in_, columns)
    np.testing.assert_array_equal(
        model.get_feature_names_out(),
        ["piplsregression0", "piplsregression1"],
    )
    transformed = model.set_output(transform="pandas").transform(X_frame)
    assert list(transformed.columns) == ["piplsregression0", "piplsregression1"]


def test_path_and_regression_selected_outputs_are_easy_to_switch() -> None:
    X, Y = _data()
    base = _fixed_estimator()
    direct = clone(base).fit(X, Y)
    path = PiPLSSearchCV(
        estimator=base,
        n_components_values=[2],
        predictor_rank_values=[3],
        max_predictor_rank=3,
        search_method="optimal",
        cv=3,
        refit=True,
        n_jobs=1,
    ).fit(X, Y)

    assert path.best_n_components_ == 2
    assert path.best_predictor_rank_ == 3
    assert path.selected_result_.n_components == 2
    assert path.selected_result_.predictor_rank == 3
    assert path.selected_estimator_ is path.best_estimator_
    assert path.selected_pipls_ is path.best_pipls_
    assert path.best_pipls_ is path.best_estimator_
    np.testing.assert_allclose(path.predict(X), direct.predict(X))
    np.testing.assert_allclose(path.best_pipls_.coef_, direct.coef_)
    np.testing.assert_allclose(
        path.best_pipls_.decomposition_.dilation,
        direct.decomposition_.dilation,
    )
    x_path, y_path = path.transform(X, Y)
    x_direct, y_direct = direct.transform(X, Y)
    np.testing.assert_allclose(x_path, x_direct)
    np.testing.assert_allclose(y_path, y_direct)


def test_path_exposes_nested_pipls_for_pipeline_without_flattening_coefficients() -> None:
    X, Y = _data()
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("regression", _fixed_estimator()),
        ]
    )
    path = PiPLSSearchCV(
        estimator=pipeline,
        n_components_values=[2],
        predictor_rank_values=[3],
        max_predictor_rank=3,
        cv=3,
        refit=True,
        n_jobs=1,
    ).fit(X, Y)

    assert isinstance(path.best_estimator_, Pipeline)
    assert path.selected_estimator_ is path.best_estimator_
    assert path.selected_pipls_ is path.best_pipls_
    assert path.best_pipls_ is path.best_estimator_.named_steps["regression"]
    assert not hasattr(path, "coef_")


def test_one_standard_error_refit_preserves_pipeline_composition() -> None:
    X, Y = _data()
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("regression", _fixed_estimator()),
        ]
    )
    path = PiPLSSearchCV(
        estimator=pipeline,
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="optimal",
        selection_rule="one_standard_error",
        cv=3,
        refit=True,
        n_jobs=1,
    ).fit(X, Y)

    assert isinstance(path.selected_estimator_, Pipeline)
    assert path.selected_pipls_ is path.selected_estimator_.named_steps["regression"]
    assert path.selected_pipls_.n_components == path.selected_result_.n_components
    assert path.selected_pipls_.predictor_rank == path.selected_result_.predictor_rank
    assert not hasattr(path, "best_estimator_")
    assert path.predict(X).shape == Y.shape


def test_path_score_accepts_sample_weight_like_regression() -> None:
    X, Y = _data()
    path = PiPLSSearchCV(
        estimator=_fixed_estimator(),
        n_components_values=[2],
        predictor_rank_values=[3],
        max_predictor_rank=3,
        cv=3,
        refit=True,
        n_jobs=1,
    ).fit(X, Y)
    weights = np.linspace(1.0, 2.0, X.shape[0])

    assert path.score(X, Y, sample_weight=weights) == pytest.approx(
        path.best_pipls_.score(X, Y, sample_weight=weights)
    )


def test_path_preserves_refitted_estimator_output_configuration() -> None:
    X, Y = _data()
    columns = [f"feature_{index}" for index in range(X.shape[1])]
    X_frame = pd.DataFrame(X, columns=columns)
    template = _fixed_estimator().set_output(transform="pandas")
    path = PiPLSSearchCV(
        estimator=template,
        n_components_values=[2],
        predictor_rank_values=[3],
        max_predictor_rank=3,
        cv=3,
        refit=True,
        n_jobs=1,
    ).fit(X_frame, Y)

    np.testing.assert_array_equal(path.feature_names_in_, columns)
    np.testing.assert_array_equal(path.best_pipls_.feature_names_in_, columns)
    transformed = path.transform(X_frame)
    assert list(transformed.columns) == ["piplsregression0", "piplsregression1"]


def test_path_preserves_dataframe_columns_inside_pipeline_folds() -> None:
    X, Y = _data()
    columns = [f"feature_{index}" for index in range(X.shape[1])]
    X_frame = pd.DataFrame(X, columns=columns)
    selected = columns[:5]
    pipeline = Pipeline(
        [
            (
                "columns",
                ColumnTransformer(
                    [("selected", StandardScaler(), selected)],
                    remainder="drop",
                ),
            ),
            (
                "regression",
                PiPLSRegression(
                    n_components=2,
                    predictor_rank=3,
                    scale=False,
                    svd_solver="full",
                    random_state=None,
                ),
            ),
        ]
    )
    path = PiPLSSearchCV(
        estimator=pipeline,
        n_components_values=[2],
        predictor_rank_values=[3],
        max_predictor_rank=3,
        cv=3,
        refit=True,
        n_jobs=1,
    ).fit(X_frame, Y)

    assert path.best_pipls_.n_features_in_ == len(selected)
    assert path.predict(X_frame).shape == Y.shape
    assert np.isfinite(path.scorer_(path.best_estimator_, X_frame, Y))


@pytest.mark.parametrize(
    "estimator",
    [
        PiPLSRegression(
            n_components=1,
            predictor_rank=1,
            scale=False,
            svd_solver="full",
            random_state=None,
        ),
        PiPLSSearchCV(
            estimator=PiPLSRegression(
                n_components=1,
                predictor_rank=1,
                scale=False,
                svd_solver="full",
                random_state=None,
            ),
            n_components_values=[1],
            predictor_rank_values=[1],
            max_predictor_rank=1,
            cv=2,
            refit=True,
            n_jobs=1,
        ),
    ],
)
def test_sklearn_common_checks_except_cross_decomposition_tuple_contract(
    estimator: Any,
) -> None:
    parameters = inspect.signature(check_estimator).parameters
    if "expected_failed_checks" not in parameters:
        pytest.skip("This scikit-learn release cannot declare expected cross-decomposition checks.")
    reason = (
        "Pi-PLS mirrors PLSRegression: fit_transform(X, y) and transform(X, y) "
        "return predictor and response scores as a tuple. Generic transformer checks "
        "special-case sklearn cross-decomposition class names only."
    )
    check_estimator(
        estimator,
        expected_failed_checks={
            "check_transformer_data_not_an_array": reason,
            "check_transformer_general": reason,
        },
        on_skip=None,
    )


def test_inverse_transform_matches_documented_least_squares_reconstruction() -> None:
    X, Y = _data()
    model = _fixed_estimator().fit(X, Y)

    x_scores, y_scores = model.transform(X, Y)
    X_reconstructed, Y_reconstructed = model.inverse_transform(x_scores, y_scores)

    expected_X = (x_scores @ model.x_loadings_.T) * model.x_scale_ + model.x_mean_
    expected_Y = (y_scores @ model.y_loadings_.T) * model.y_scale_ + model.y_mean_
    np.testing.assert_allclose(X_reconstructed, expected_X)
    np.testing.assert_allclose(Y_reconstructed, expected_Y)


def test_decomposition_is_single_source_of_truth_for_pipls_specific_arrays() -> None:
    X, Y = _data()
    model = _fixed_estimator().fit(X, Y)

    assert model.x_rotations_ is model.decomposition_.predictor_rotations
    assert model.y_rotations_ is model.decomposition_.response_rotations
    assert not model.x_rotations_.flags.writeable


def test_path_restricts_estimator_scope_to_direct_or_final_pipeline_pipls() -> None:
    X, Y = _data()
    unsupported = TransformedTargetRegressor(regressor=_fixed_estimator())

    with pytest.raises(ValueError, match="PiPLSRegression or a sklearn Pipeline"):
        PiPLSSearchCV(estimator=unsupported).fit(X, Y)

    invalid_pipeline = Pipeline([("pipls", _fixed_estimator()), ("scale", StandardScaler())])
    with pytest.raises(ValueError, match="pipelines must end"):
        PiPLSSearchCV(estimator=invalid_pipeline).fit(X, Y)


def test_path_search_diagnostics_and_inverse_transform_are_sklearn_like() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        estimator=_fixed_estimator(),
        n_components_values=[2],
        predictor_rank_values=[3],
        max_predictor_rank=3,
        cv=None,
        scoring=None,
        refit=True,
        n_jobs=1,
    ).fit(X, Y)

    assert search.n_splits_ == 5
    assert callable(search.scorer_)
    assert search.refit_time_ >= 0.0
    assert {
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    } <= search.cv_results_.keys()
    assert search.component_path_.predictor_rank_policy == "fixed"
    assert isinstance(search.component_path_, PiPLSComponentPath)
    assert search.component_path_.for_n_components(2).predictor_rank == 3
    assert not hasattr(search, "component_path_results_")
    assert not hasattr(search, "best_predictor_rank_by_n_components_")
    assert not hasattr(search, "best_score_by_n_components_")
    x_scores, y_scores = search.transform(X, Y)
    X_reconstructed, Y_reconstructed = search.inverse_transform(x_scores, y_scores)
    direct_X, direct_Y = search.best_pipls_.inverse_transform(x_scores, y_scores)
    np.testing.assert_allclose(X_reconstructed, direct_X)
    np.testing.assert_allclose(Y_reconstructed, direct_Y)


def test_path_inverse_transform_is_conditionally_available_for_pipeline() -> None:
    pipeline = Pipeline(
        [
            (
                "columns",
                ColumnTransformer(
                    [("selected", StandardScaler(), [0, 1, 2, 3])],
                    remainder="drop",
                ),
            ),
            ("regression", _fixed_estimator()),
        ]
    )
    search = PiPLSSearchCV(estimator=pipeline, refit=True)

    assert hasattr(search, "transform")
    assert not hasattr(search, "inverse_transform")
