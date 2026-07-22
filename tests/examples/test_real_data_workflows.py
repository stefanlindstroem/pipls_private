from __future__ import annotations

from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_pulp_example_uses_direct_in_memory_results() -> None:
    text = (_repository_root() / "examples" / "10_pulp_real_data.py").read_text(
        encoding="utf-8"
    )

    assert "PiPLSPathCV(refit=False).fit(X, Y)" in text
    assert "path.for_n_components(CHOSEN_N_COMPONENTS)" in text
    assert "PiPLSRegression(" in text
    assert "cross_val_predict(" in text
    assert "pipls_display_factors(model.decomposition_)" in text
    assert "latent_structure(model)" in text
    assert "prediction_diagnostics(" in text
    assert 'prediction_kind="selection-conditioned OOF predictions"' in text
    assert 'ANALYSIS_DIR / "predictor_rank_profile.pdf"' in text
    assert 'ANALYSIS_DIR / "pipls_factors.pdf"' in text
    assert 'ANALYSIS_DIR / "latent_structure.pdf"' in text
    assert 'ANALYSIS_DIR / "coefficients.pdf"' in text
    assert 'ANALYSIS_DIR / "prediction_diagnostics.pdf"' in text
    assert "build_post_analysis_tables(" not in text
    assert "write_post_analysis_tables(" not in text
    assert "render_post_analysis_report(" not in text
    assert "fixed_model_oof_predictions" not in text
    assert "run_pulp_workflow(" not in text
    assert ".to_csv(" not in text
    assert "subprocess" not in text


def test_sugarcane_example_is_a_direct_in_memory_workflow() -> None:
    text = (_repository_root() / "examples" / "11_sugarcane_real_data.py").read_text(
        encoding="utf-8"
    )

    assert "CHOSEN_N_COMPONENTS = 2" in text
    assert "from sklearn.cross_decomposition import PLSRegression" not in text
    assert "X.columns.to_numpy(dtype=float)" in text
    assert "Y.columns.tolist()" in text
    assert "path_search.component_path_" in text
    assert "path.for_n_components(CHOSEN_N_COMPONENTS)" in text
    assert "cross_val_predict(" in text
    assert "cv=KFold(n_splits=5, shuffle=False)" in text
    assert "pipls_display_factors(model.decomposition_)" in text
    assert "latent_structure(model)" in text
    assert 'prediction_kind="selection-conditioned OOF predictions"' in text
    assert 'predictor_style="line"' in text
    assert 'predictor_axis_label="Wavelength (nm)"' in text
    assert "include_prediction_kind=False" in text
    assert "build_post_analysis_tables(" not in text
    assert "write_post_analysis_tables(" not in text
    assert "render_post_analysis_report(" not in text
    assert "fixed_model_oof_predictions(" not in text
    assert "plot_pipls_component_path(" not in text
    assert ".to_csv(" not in text
    assert text.count("pd.read_csv") == 2
    assert "subprocess" not in text

    expected_pdfs = {
        "component_path.pdf",
        "pipls_factors.pdf",
        "prediction_diagnostics.pdf",
        "latent_structure.pdf",
        "coefficients.pdf",
    }
    assert {
        filename
        for filename in expected_pdfs
        if f'ANALYSIS_DIR / "{filename}"' in text
    } == expected_pdfs
    assert text.count("figure.savefig(") == len(expected_pdfs)
    assert text.count("plt.close(figure)") == len(expected_pdfs)


def test_tobacco_example_is_a_direct_paginated_spectral_workflow() -> None:
    root = _repository_root()
    text = (root / "examples" / "12_tobacco_real_data.py").read_text(encoding="utf-8")

    assert "CHOSEN_N_COMPONENTS = 8" in text
    assert "DISPLAY_COMPONENTS = (0, 1, 2, 3)" in text
    assert "RESPONSES_PER_PAGE = 5" in text
    assert "from sklearn.cross_decomposition import PLSRegression" not in text
    assert "X.columns.to_numpy(dtype=float)" in text
    assert "Y.columns.tolist()" in text
    assert "range(0, len(response_names), RESPONSES_PER_PAGE)" in text
    assert "min(start + RESPONSES_PER_PAGE, len(response_names))" in text
    assert 'estimator=PiPLSRegression(svd_solver="full")' in text
    assert 'search_method="auto"' in text
    assert "path_search.component_path_" in text
    assert "path.for_n_components(CHOSEN_N_COMPONENTS)" in text
    assert "cross_val_predict(" in text
    assert "cv=KFold(n_splits=5, shuffle=False)" in text
    assert "pipls_display_factors(model.decomposition_)" in text
    assert "latent_structure(model)" in text
    assert "observation_diagnostics(model, X)" in text
    assert 'prediction_kind="selection-conditioned OOF predictions"' in text
    assert 'predictor_style="line"' in text
    assert 'predictor_axis_label="Wavenumber (cm$^{-1}$)"' in text
    assert "components=DISPLAY_COMPONENTS" in text
    assert text.count("for page_number, responses in enumerate(response_pages, start=1):") == 2
    assert text.count("include_prediction_kind=False") == 3
    assert text.count("pd.read_csv") == 2
    assert ".to_csv(" not in text
    assert "build_post_analysis_tables(" not in text
    assert "write_post_analysis_tables(" not in text
    assert "render_post_analysis_report(" not in text
    assert "fixed_model_oof_predictions(" not in text
    assert "plot_pipls_component_path(" not in text
    assert "subprocess" not in text

    expected_pdfs = {
        "component_path.pdf",
        "pipls_factors.pdf",
        "prediction_diagnostics.pdf",
        "latent_structure.pdf",
        "coefficients.pdf",
    }
    assert {
        filename
        for filename in expected_pdfs
        if f'ANALYSIS_DIR / "{filename}"' in text
    } == expected_pdfs
    assert text.count("PdfPages(") == 2
    assert text.count("figure.savefig(") == 3
    assert text.count("report.savefig(figure)") == 2
    assert text.count("plt.close(figure)") == 5
    assert not (root / "examples" / "_support" / "fixed_model_oof.py").exists()
    assert not (root / "examples" / "_support" / "post_analysis_artifacts.py").exists()
