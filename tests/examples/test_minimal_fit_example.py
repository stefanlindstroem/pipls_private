from __future__ import annotations

from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_minimal_example_is_a_literal_fixed_model_workflow() -> None:
    path = _repository_root() / "examples" / "01_minimal_fit_and_plot.py"
    text = path.read_text(encoding="utf-8")

    assert "X = np.array(" in text
    assert "Y = np.array(" in text
    assert "PiPLSRegression(n_components=1, predictor_rank=2).fit(X, Y)" in text
    assert "model.predict(X)" in text
    assert "pipls_display_factors(model.decomposition_)" in text
    assert "plot_pipls_decomposition(" in text
    assert 'predictor_names = ["Temperature", "Pressure", "Flow rate"]' in text
    assert 'response_names = ["Yield", "Purity"]' in text
    assert "PiPLSPathCV" not in text
    assert "KFold" not in text
    assert "pandas" not in text
    assert "_support" not in text
