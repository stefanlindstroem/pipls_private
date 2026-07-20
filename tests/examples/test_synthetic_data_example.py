from __future__ import annotations

import re
import runpy
from pathlib import Path

import pytest


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_synthetic_example_explains_problem_and_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runpy.run_path(
        str(_repository_root() / "examples" / "08_synthetic_data.py"),
        run_name="__main__",
    )
    output = capsys.readouterr().out

    assert "Synthetic Pi-PLS train/test example" in output
    assert "Training data: X(120, 20), Y(120, 5)" in output
    assert "Independent test data: X(40, 20), Y(40, 5)" in output
    assert "Shared directions affecting X and Y: 2" in output
    assert "Predictor-specific directions affecting only X: 2" in output
    assert "Response-specific directions affecting only Y: 1" in output
    assert "Components: 2" in output
    assert "Predictor rank: 4" in output
    assert "Predictions: (40, 5)" in output
    assert re.search(r"Test R\^2 \(coefficient of determination\): -?\d+\.\d{3}", output)


def test_numbered_examples_are_independent_of_publication_context() -> None:
    examples_dir = _repository_root() / "examples"
    publication_terms = re.compile(r"\b(paper|manuscript)\b", re.IGNORECASE)

    for path in sorted(examples_dir.glob("[0-9][0-9]_*.py")):
        assert publication_terms.search(path.read_text(encoding="utf-8")) is None, path
