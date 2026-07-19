from __future__ import annotations

from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_model_inspection_example_reads_scientific_labels_from_csv_headers() -> None:
    text = (_repository_root() / "examples" / "09_model_inspection.py").read_text(encoding="utf-8")

    assert 'DATA_DIR = REPOSITORY_ROOT / "datasets" / "pulp"' in text
    assert 'X = pd.read_csv(DATA_DIR / "X.csv")' in text
    assert 'Y = pd.read_csv(DATA_DIR / "Y.csv")' in text
    assert "predictor_names = X.columns.astype(str).tolist()" in text
    assert "response_names = Y.columns.astype(str).tolist()" in text
    assert "predictor_names=predictor_names" in text
    assert "response_names=response_names" in text
    assert "pipls_display_factors(model.decomposition_)" in text
    assert 'prediction_kind="fitted values"' in text
    assert "plot_pipls_decomposition(" in text
    assert "plot_prediction_diagnostics(" in text
    assert "model.predict(X)" in text
    assert "PLSRegression(n_components=N_COMPONENTS, scale=True)" in text
    assert "pls_latent_structure(pls_model)" in text
    assert "plot_pls_scores(" in text
    assert "plot_pls_x_loadings(" in text
    assert "plot_pls_y_loadings(" in text
    assert "plot_pls_coefficients(" in text
    assert "Process variable A" not in text
    assert "Quality response A" not in text
    assert "make_pipls_train_test(" not in text
    assert "subprocess" not in text
