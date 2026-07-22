from __future__ import annotations

import re
import runpy
from pathlib import Path

import pytest
from matplotlib.figure import Figure


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_synthetic_example_explains_selection_and_output(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Figure, "savefig", lambda self, *args, **kwargs: None)
    runpy.run_path(
        str(_repository_root() / "examples" / "02_synthetic_path_selection.py"),
        run_name="__main__",
    )
    output = capsys.readouterr().out

    assert "Synthetic Pi-PLS path-selection example" in output
    assert "Training data: X(120, 8), Y(120, 3)" in output
    assert "Independent test data: X(60, 8), Y(60, 3)" in output
    assert "2 shared directions and 2 predictor-specific directions" in output
    assert "Selected fixed model: n_components=2, predictor_rank=4" in output
    assert re.search(r"External-test R\^2: -?\d+\.\d{3}", output)
    assert "Wrote PDF figures to" in output


def test_synthetic_example_owns_the_short_selection_workflow() -> None:
    source = (
        _repository_root() / "examples" / "02_synthetic_path_selection.py"
    ).read_text(encoding="utf-8")

    assert "make_pipls_train_test(" in source
    assert "PiPLSPathCV(refit=False).fit(train.X, train.Y)" in source
    assert "path.for_n_components(CHOSEN_N_COMPONENTS)" in source
    assert "search.predictor_rank_profile(selected.n_components)" in source
    assert "model = PiPLSRegression(" in source
    assert "model.predict(test.X)" in source
    assert 'prediction_kind="external test predictions"' in source
    assert "plot_observed_vs_predicted(" in source
    assert source.count("figure.savefig(") == 3
    assert ".to_csv(" not in source
    assert "pandas" not in source


def test_numbered_examples_are_independent_of_publication_context() -> None:
    examples_dir = _repository_root() / "examples"
    publication_terms = re.compile(r"\b(paper|manuscript)\b", re.IGNORECASE)

    for path in sorted(examples_dir.glob("[0-9][0-9]_*.py")):
        assert publication_terms.search(path.read_text(encoding="utf-8")) is None, path
