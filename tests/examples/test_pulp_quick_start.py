from __future__ import annotations

from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_pulp_quick_start_is_the_automatic_search_refit_workflow() -> None:
    path = _repository_root() / "examples" / "01_pulp_quick_start.py"
    text = path.read_text(encoding="utf-8")

    assert path.is_file()
    assert "from pipls import PiPLSSearchCV" in text
    assert "from pipls.datasets import load_pulp" in text
    assert "data = load_pulp()" in text
    assert "X, Y = data.data, data.target" in text
    assert (
        'model = PiPLSSearchCV().fit(X, Y).refit(X, Y, rule="one_standard_error")'
        in text
    )
    assert "model.predict(X)" in text
    assert 'prediction_kind="fitted values"' in text
    assert "diagnostics.observed_standardized.ravel()" in text
    assert "diagnostics.predicted_standardized.ravel()" in text
    assert "diagnostics.standardized_rmse.mean()" in text
    assert "plt.subplots(" in text
    assert "axis.scatter(observed, fitted)" in text
    assert 'axis.plot(limits, limits, "--"' in text
    assert '"pulp_quick_start.pdf"' in text
    assert 'print(f"Wrote PDF figure to {output_path}")' in text

    for section in (
        "load-pulp-data",
        "fit-selected-pulp-model",
        "plot-standardized-fitted-values",
    ):
        assert f"# --8<-- [start:{section}]" in text
        assert f"# --8<-- [end:{section}]" in text

    assert "PiPLSRegression" not in text
    assert "validation_report" not in text
    assert "KFold" not in text
    assert "np.array" not in text
    assert "pandas" not in text
    assert "_support" not in text
