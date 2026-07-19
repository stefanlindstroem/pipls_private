from __future__ import annotations

from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_synthetic_model_inspection_example_uses_external_test_predictions() -> None:
    text = (_repository_root() / "examples" / "09_model_inspection.py").read_text(encoding="utf-8")

    assert "make_pipls_train_test(" in text
    assert "pipls_display_factors(model.decomposition_)" in text
    assert 'prediction_kind="external test predictions"' in text
    assert "plot_pipls_decomposition(" in text
    assert "plot_prediction_diagnostics(" in text
    assert "model.predict(test.X)" in text
    assert "PLSRegression(n_components=2, scale=True)" in text
    assert "pls_latent_structure(pls_model)" in text
    assert "plot_pls_scores(" in text
    assert "plot_pls_x_loadings(" in text
    assert "plot_pls_y_loadings(" in text
    assert "plot_pls_coefficients(" in text
    assert "subprocess" not in text
