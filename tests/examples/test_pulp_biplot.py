from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pipls.inspection import BiplotCoordinates


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_pulp_biplot() -> ModuleType:
    path = _repository_root() / "examples" / "_support" / "pulp_biplot.py"
    spec = importlib.util.spec_from_file_location("pulp_biplot_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PULP_BIPLOT = _load_pulp_biplot()


@pytest.fixture
def coordinates() -> BiplotCoordinates:
    return BiplotCoordinates(
        sample_coordinates=np.array([[0.1, 0.2], [-0.3, 0.4], [0.2, -0.1]]),
        predictor_coordinates=np.array([[1.5, -2.0], [-0.25, 4.0]]),
        component_indices=np.array([0, 1]),
        scaling_factors=np.array([1.0, 1.0]),
    )


def test_simple_biplot_draws_geometry_and_endpoint_labels(
    monkeypatch: pytest.MonkeyPatch,
    coordinates: BiplotCoordinates,
) -> None:
    monkeypatch.setattr(PULP_BIPLOT, "_textalloc", None)
    figure, axis = plt.subplots()
    try:
        labels = PULP_BIPLOT.plot_pulp_biplot(
            axis,
            coordinates,
            predictor_names=("A", "B"),
        )

        assert len(labels) == 2
        assert [label.get_text() for label in labels] == ["A", "B"]
        assert all(label.get_fontsize() == 9 for label in labels)
        np.testing.assert_allclose(labels[0].get_position(), (1.5, -2.0))
        np.testing.assert_allclose(labels[1].get_position(), (-0.25, 4.0))
        assert len(axis.collections) == 1
        assert len(axis.patches) == 2
        assert axis.get_xlabel() == "Balanced component 1"
        assert axis.get_ylabel() == "Balanced component 2"
    finally:
        plt.close(figure)


def test_textalloc_biplot_uses_only_predictor_arrow_line_obstacles(
    monkeypatch: pytest.MonkeyPatch,
    coordinates: BiplotCoordinates,
) -> None:
    captured: dict[str, object] = {}
    text_objects = [object(), object()]

    def allocate(*args: object, **kwargs: object) -> tuple[object, object, list[object], object]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object(), object(), text_objects, object()

    monkeypatch.setattr(PULP_BIPLOT, "_textalloc", SimpleNamespace(allocate=allocate))
    figure, axis = plt.subplots()
    try:
        result = PULP_BIPLOT.plot_pulp_biplot(
            axis,
            coordinates,
            predictor_names=("A", "B"),
            title="Example",
        )

        assert result == tuple(text_objects)
        assert len(axis.collections) == 1
        assert len(axis.patches) == 2
        assert axis.get_title() == "Example"
        args = captured["args"]
        kwargs = captured["kwargs"]
        assert args[0] is axis
        np.testing.assert_array_equal(args[1], coordinates.predictor_coordinates[:, 0])
        np.testing.assert_array_equal(args[2], coordinates.predictor_coordinates[:, 1])
        assert args[3] == ("A", "B")
        assert set(kwargs) == {
            "x_lines",
            "y_lines",
            "textsize",
            "min_distance",
            "max_distance",
            "draw_all",
            "draw_lines",
        }
        assert "x_scatter" not in kwargs
        assert "y_scatter" not in kwargs
        assert kwargs["textsize"] == 9
        assert kwargs["min_distance"] == pytest.approx(0.01125)
        assert kwargs["max_distance"] == pytest.approx(0.15)
        assert kwargs["draw_all"] is True
        assert kwargs["draw_lines"] is False
        x_lines = kwargs["x_lines"]
        y_lines = kwargs["y_lines"]
        assert len(x_lines) == len(y_lines) == 2
        np.testing.assert_array_equal(x_lines[0], np.array([0.0, 1.5]))
        np.testing.assert_array_equal(y_lines[0], np.array([0.0, -2.0]))
        np.testing.assert_array_equal(x_lines[1], np.array([0.0, -0.25]))
        np.testing.assert_array_equal(y_lines[1], np.array([0.0, 4.0]))
    finally:
        plt.close(figure)


def test_explicit_textalloc_renderer_requires_textalloc(
    monkeypatch: pytest.MonkeyPatch,
    coordinates: BiplotCoordinates,
) -> None:
    monkeypatch.setattr(PULP_BIPLOT, "_textalloc", None)
    figure, axis = plt.subplots()
    try:
        with pytest.raises(RuntimeError, match="textalloc is unavailable"):
            PULP_BIPLOT.plot_pulp_biplot_textalloc(
                axis,
                coordinates,
                predictor_names=("A", "B"),
            )
    finally:
        plt.close(figure)


def test_biplot_validates_predictor_label_alignment(
    monkeypatch: pytest.MonkeyPatch,
    coordinates: BiplotCoordinates,
) -> None:
    monkeypatch.setattr(PULP_BIPLOT, "_textalloc", None)
    figure, axis = plt.subplots()
    try:
        with pytest.raises(ValueError, match="one label"):
            PULP_BIPLOT.plot_pulp_biplot(
                axis,
                coordinates,
                predictor_names=("A",),
            )
    finally:
        plt.close(figure)
