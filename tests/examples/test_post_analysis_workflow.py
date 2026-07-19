from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from pipls import PiPLSRegression
from pipls.inspection import (
    pipls_display_factors,
    pls_latent_structure,
    pls_observation_diagnostics,
    prediction_diagnostics,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_example_module(filename: str, module_name: str) -> ModuleType:
    path = _repository_root() / "examples" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OOF = _load_example_module("fixed_model_oof.py", "fixed_model_oof_example")
ARTIFACTS = _load_example_module(
    "post_analysis_artifacts.py",
    "post_analysis_artifacts_example",
)


def _example_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(914)
    X = pd.DataFrame(
        rng.normal(size=(30, 5)),
        columns=[f"Predictor {index}" for index in range(1, 6)],
    )
    coefficient = np.array(
        [
            [1.0, -0.3],
            [0.2, 0.7],
            [-0.5, 0.1],
            [0.3, -0.4],
            [0.0, 0.2],
        ]
    )
    Y = pd.DataFrame(
        X.to_numpy() @ coefficient + 0.1 * rng.normal(size=(30, 2)),
        columns=["Response A", "Response B"],
    )
    return X, Y


def test_fixed_model_oof_matches_an_explicit_manual_loop() -> None:
    X, Y = _example_data()
    splitter = KFold(n_splits=5, shuffle=False)
    estimator = PLSRegression(n_components=2, scale=True)

    result = OOF.fixed_model_oof_predictions(estimator, X, Y, splitter=splitter)

    expected = np.empty_like(Y.to_numpy())
    expected_folds = np.zeros(len(X), dtype=np.int64)
    for fold, (train, validation) in enumerate(splitter.split(X, Y), start=1):
        model = PLSRegression(n_components=2, scale=True).fit(X.iloc[train], Y.iloc[train])
        expected[validation] = model.predict(X.iloc[validation])
        expected_folds[validation] = fold

    np.testing.assert_allclose(result.predictions, expected)
    np.testing.assert_array_equal(result.fold_index, expected_folds)
    assert result.n_splits == 5
    assert not result.predictions.flags.writeable
    assert not result.fold_index.flags.writeable


def test_fixed_model_oof_requires_exactly_one_validation_assignment() -> None:
    X, Y = _example_data()

    class IncompleteSplitter:
        def split(self, X_values: object, Y_values: object) -> list[tuple[np.ndarray, np.ndarray]]:
            del X_values, Y_values
            return [(np.arange(15, 30), np.arange(0, 15))]

    with pytest.raises(ValueError, match="exactly one validation fold"):
        OOF.fixed_model_oof_predictions(
            PLSRegression(n_components=2),
            X,
            Y,
            splitter=IncompleteSplitter(),
        )


def test_post_analysis_tables_and_report_round_trip_through_csv(tmp_path: Path) -> None:
    X, Y = _example_data()
    pipls_model = PiPLSRegression(n_components=2, predictor_rank=4).fit(X, Y)
    pls_model = PLSRegression(n_components=2, scale=True).fit(X, Y)
    pipls_diagnostics = prediction_diagnostics(
        Y,
        pipls_model.predict(X),
        prediction_kind="fixed-parameter OOF predictions",
    )
    pls_diagnostics = prediction_diagnostics(
        Y,
        pls_model.predict(X),
        prediction_kind="fixed-parameter OOF predictions",
    )

    tables = ARTIFACTS.build_post_analysis_tables(
        factors=pipls_display_factors(pipls_model.decomposition_),
        diagnostics_by_model={"Pi-PLS": pipls_diagnostics, "PLS": pls_diagnostics},
        pls_structure=pls_latent_structure(pls_model),
        predictor_names=X.columns.tolist(),
        response_names=Y.columns.tolist(),
        sample_names=[str(index) for index in range(1, len(X) + 1)],
        fold_index=np.repeat(np.arange(1, 6), 6),
    )

    assert set(tables) == set(ARTIFACTS.REQUIRED_TABLE_NAMES)
    for name, table in tables.items():
        assert tuple(table.columns) == ARTIFACTS.TABLE_COLUMNS[name]
        assert not table.empty

    stale_optional = tmp_path / ARTIFACTS.TABLE_FILENAMES[
        "pls_observation_diagnostics"
    ]
    stale_optional.write_text("stale\n", encoding="utf-8")
    paths = ARTIFACTS.write_post_analysis_tables(tmp_path, tables)
    assert set(paths) == set(ARTIFACTS.REQUIRED_TABLE_NAMES)
    assert not stale_optional.exists()
    for path in paths.values():
        assert path.is_file()

    predictions = pd.read_csv(paths["predictions"])
    np.testing.assert_allclose(
        predictions["residual"],
        predictions["observed"] - predictions["predicted"],
    )
    assert set(predictions["prediction_kind"]) == {"fixed-parameter OOF predictions"}

    pdf_path = tmp_path / "post_analysis.pdf"
    ARTIFACTS.render_post_analysis_report(
        tmp_path,
        pdf_path,
        dataset_name="Synthetic",
        pls_score_components=(1, 2),
        pls_biplot_components=(1, 2),
        pls_loading_components=(1, 2),
        coefficient_responses=("Response A",),
    )
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert pdf_path.stat().st_size > 5_000


def test_pulp_example_contains_complete_three_stage_post_analysis() -> None:
    text = (_repository_root() / "examples" / "10_pulp_real_data.py").read_text(
        encoding="utf-8"
    )

    assert "CHOSEN_N_COMPONENTS = 3" in text
    assert "CHOSEN_PLS_N_COMPONENTS = 8" in text
    assert "PLS_SCORE_COMPONENTS = (1, 2)" in text
    assert "PLS_BIPLOT_COMPONENTS = (1, 2)" in text
    assert "PLS_LOADING_COMPONENTS = (1, 2, 3)" in text
    assert "predictor_names = X.columns.astype(str).tolist()" in text
    assert "response_names = Y.columns.astype(str).tolist()" in text
    assert "fixed_model_oof_predictions(" in text
    assert 'prediction_kind = "selection-conditioned OOF predictions"' in text
    assert "build_post_analysis_tables(" in text
    assert "write_post_analysis_tables(" in text
    assert "render_post_analysis_report(" in text
    assert "pls_biplot_components=PLS_BIPLOT_COMPONENTS" in text
    assert "POST_ANALYSIS_DIR" in text
    assert "POST_ANALYSIS_PDF" in text
    assert 'prediction_kind="fitted values"' not in text
    assert "subprocess" not in text


def test_post_analysis_report_supports_an_explicit_physical_predictor_axis(
    tmp_path: Path,
) -> None:
    X, Y = _example_data()
    pipls_model = PiPLSRegression(n_components=2, predictor_rank=4).fit(X, Y)
    pls_model = PLSRegression(n_components=2, scale=True).fit(X, Y)
    diagnostics = prediction_diagnostics(
        Y,
        pls_model.predict(X),
        prediction_kind="fixed-parameter OOF predictions",
    )
    wavelength_labels = [str(value) for value in np.linspace(780, 2500, X.shape[1])]
    tables = ARTIFACTS.build_post_analysis_tables(
        factors=pipls_display_factors(pipls_model.decomposition_),
        diagnostics_by_model={"PLS": diagnostics},
        pls_structure=pls_latent_structure(pls_model),
        predictor_names=wavelength_labels,
        response_names=Y.columns.tolist(),
        sample_names=[str(index) for index in range(1, len(X) + 1)],
        fold_index=np.repeat(np.arange(1, 6), 6),
    )
    ARTIFACTS.write_post_analysis_tables(tmp_path, tables)

    pdf_path = tmp_path / "spectral_post_analysis.pdf"
    ARTIFACTS.render_post_analysis_report(
        tmp_path,
        pdf_path,
        dataset_name="Synthetic spectra",
        predictor_style="line",
        predictor_axis=np.linspace(780.0, 2500.0, X.shape[1]),
        predictor_axis_label="Wavelength (nm)",
        pls_score_components=(1, 2),
        pls_loading_components=(1, 2),
        coefficient_responses=("Response A", "Response B"),
    )

    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert pdf_path.stat().st_size > 5_000


def test_post_analysis_report_supports_response_pages_and_observation_diagnostics(
    tmp_path: Path,
) -> None:
    X, Y = _example_data()
    pipls_model = PiPLSRegression(n_components=2, predictor_rank=4).fit(X, Y)
    pls_model = PLSRegression(n_components=2, scale=True).fit(X, Y)
    diagnostics = prediction_diagnostics(
        Y,
        pls_model.predict(X),
        prediction_kind="fixed-parameter OOF predictions",
    )
    tables = ARTIFACTS.build_post_analysis_tables(
        factors=pipls_display_factors(pipls_model.decomposition_),
        diagnostics_by_model={"PLS": diagnostics},
        pls_structure=pls_latent_structure(pls_model),
        predictor_names=X.columns.tolist(),
        response_names=Y.columns.tolist(),
        sample_names=[str(index) for index in range(1, len(X) + 1)],
        fold_index=np.repeat(np.arange(1, 6), 6),
        pls_observation_diagnostics_result=pls_observation_diagnostics(pls_model, X),
    )
    paths = ARTIFACTS.write_post_analysis_tables(tmp_path, tables)

    assert "pls_observation_diagnostics" in paths
    observation_table = pd.read_csv(paths["pls_observation_diagnostics"])
    assert tuple(observation_table.columns) == ARTIFACTS.TABLE_COLUMNS[
        "pls_observation_diagnostics"
    ]
    assert len(observation_table) == len(X)

    pdf_path = tmp_path / "paginated_post_analysis.pdf"
    ARTIFACTS.render_post_analysis_report(
        tmp_path,
        pdf_path,
        dataset_name="Synthetic",
        pls_score_components=(1, 2),
        pls_loading_components=(1, 2),
        response_pages=(("Response A",), ("Response B",)),
    )

    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert pdf_path.stat().st_size > 5_000


def test_response_pages_must_partition_source_order(tmp_path: Path) -> None:
    X, Y = _example_data()
    pipls_model = PiPLSRegression(n_components=2, predictor_rank=4).fit(X, Y)
    pls_model = PLSRegression(n_components=2, scale=True).fit(X, Y)
    diagnostics = prediction_diagnostics(
        Y,
        pls_model.predict(X),
        prediction_kind="fixed-parameter OOF predictions",
    )
    tables = ARTIFACTS.build_post_analysis_tables(
        factors=pipls_display_factors(pipls_model.decomposition_),
        diagnostics_by_model={"PLS": diagnostics},
        pls_structure=pls_latent_structure(pls_model),
        predictor_names=X.columns.tolist(),
        response_names=Y.columns.tolist(),
        sample_names=[str(index) for index in range(1, len(X) + 1)],
        fold_index=np.repeat(np.arange(1, 6), 6),
    )
    ARTIFACTS.write_post_analysis_tables(tmp_path, tables)

    with pytest.raises(ValueError, match="partition response_names"):
        ARTIFACTS.render_post_analysis_report(
            tmp_path,
            tmp_path / "invalid.pdf",
            dataset_name="Synthetic",
            pls_score_components=(1, 2),
            pls_loading_components=(1, 2),
            response_pages=(("Response B",), ("Response A",)),
        )


def test_sugarcane_example_contains_complete_spectral_post_analysis() -> None:
    text = (_repository_root() / "examples" / "11_sugarcane_real_data.py").read_text(
        encoding="utf-8"
    )

    assert "CHOSEN_N_COMPONENTS = 2" in text
    assert "CHOSEN_PLS_N_COMPONENTS = 4" in text
    assert "PLS_SCORE_COMPONENTS = (1, 2)" in text
    assert "PLS_LOADING_COMPONENTS = (1, 2, 3, 4)" in text
    assert 'COEFFICIENT_RESPONSES = ("TS", "CP", "ADF", "IVOMD")' in text
    assert "predictor_names = X.columns.astype(str).tolist()" in text
    assert "response_names = Y.columns.astype(str).tolist()" in text
    assert "wavelength_nm = X.columns.to_numpy(dtype=np.float64)" in text
    assert 'predictor_style="line"' in text
    assert 'predictor_axis_label="Wavelength (nm)"' in text
    assert "fixed_model_oof_predictions(" in text
    assert 'prediction_kind = "selection-conditioned OOF predictions"' in text
    assert "build_post_analysis_tables(" in text
    assert "write_post_analysis_tables(" in text
    assert "render_post_analysis_report(" in text
    assert "POST_ANALYSIS_DIR" in text
    assert "POST_ANALYSIS_PDF" in text
    assert "pls_biplot_components" not in text
    assert "subprocess" not in text


def test_tobacco_example_contains_paginated_spectral_post_analysis() -> None:
    text = (_repository_root() / "examples" / "12_tobacco_real_data.py").read_text(
        encoding="utf-8"
    )

    assert "CHOSEN_N_COMPONENTS = 8" in text
    assert "CHOSEN_PLS_N_COMPONENTS = 13" in text
    assert "PLS_SCORE_COMPONENTS = (1, 2)" in text
    assert "PLS_LOADING_COMPONENTS = (1, 2, 3, 4)" in text
    assert "RESPONSE_PAGE_SIZE = 5" in text
    assert "predictor_names = X.columns.astype(str).tolist()" in text
    assert "response_names = Y.columns.astype(str).tolist()" in text
    assert "wavenumber_cm_inverse = X.columns.to_numpy(dtype=np.float64)" in text
    assert "np.all(np.diff(wavenumber_cm_inverse) < 0.0)" in text
    assert 'predictor_style="line"' in text
    assert 'predictor_axis_label="Wavenumber (cm$^{-1}$)"' in text
    assert "response_pages=response_pages" in text
    assert "pls_observation_diagnostics(pls_model, X)" in text
    assert "fixed_model_oof_predictions(" in text
    assert 'prediction_kind = "selection-conditioned OOF predictions"' in text
    assert "build_post_analysis_tables(" in text
    assert "write_post_analysis_tables(" in text
    assert "render_post_analysis_report(" in text
    assert "POST_ANALYSIS_DIR" in text
    assert "POST_ANALYSIS_PDF" in text
    assert "pls_biplot_components" not in text
    assert "subprocess" not in text
