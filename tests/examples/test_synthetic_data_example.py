from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_synthetic_example_produces_finite_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Figure, "savefig", lambda self, *args, **kwargs: None)
    namespace = runpy.run_path(
        str(_repository_root() / "examples" / "02_synthetic_path_selection.py"),
        run_name="__main__",
    )

    train = namespace["train"]
    test = namespace["test"]
    selection = namespace["selection"]
    model = namespace["model"]
    diagnostics = namespace["diagnostics"]

    assert selection.n_components == 2
    assert selection.predictor_rank == 4
    assert model.selection_ == selection
    assert diagnostics.prediction_kind == "external test predictions"
    assert diagnostics.observed.shape == test.Y.shape
    assert diagnostics.predicted.shape == test.Y.shape
    assert train.X.shape[0] == 120
    assert test.X.shape[0] == 60
    assert np.isfinite(diagnostics.response_r2).all()
