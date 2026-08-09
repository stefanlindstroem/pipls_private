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

    assert "Synthetic Π-PLS path-selection example" in output
    assert "Training data: X(120, 8), Y(120, 3)" in output
    assert "Independent test data: X(60, 8), Y(60, 3)" in output
    assert "2 shared directions and 2 predictor-specific directions" in output
    assert "Selected fixed model: n_components=2, predictor_rank=4" in output
    assert re.search(r"External-test R\^2: -?\d+\.\d{3}", output)
    assert "Wrote PDF figures to" in output
