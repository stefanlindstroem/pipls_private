from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_annotation_layout() -> ModuleType:
    path = _repository_root() / "examples" / "_support" / "annotation_layout.py"
    spec = importlib.util.spec_from_file_location("annotation_layout_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANNOTATION_LAYOUT = _load_annotation_layout()


class _FakeAxis:
    def __init__(self) -> None:
        self.text_calls: list[tuple[float, float, str, dict[str, object]]] = []

    def text(self, x: float, y: float, label: str, **kwargs: object) -> object:
        self.text_calls.append((x, y, label, kwargs))
        return object()


def test_textalloc_receives_only_predictor_arrow_line_obstacles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    text_objects = [object(), object()]

    def allocate(*args: object, **kwargs: object) -> tuple[object, object, list[object], object]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object(), object(), text_objects, object()

    monkeypatch.setattr(ANNOTATION_LAYOUT, "_textalloc", SimpleNamespace(allocate=allocate))
    endpoints = np.array([[1.5, -2.0], [-0.25, 4.0]])
    axis = _FakeAxis()

    result = ANNOTATION_LAYOUT.allocate_predictor_labels(
        axis,
        endpoints,
        ("A", "B"),
        textsize=9,
    )

    assert result == tuple(text_objects)
    assert axis.text_calls == []
    args = captured["args"]
    kwargs = captured["kwargs"]
    assert args[0] is axis
    np.testing.assert_array_equal(args[1], endpoints[:, 0])
    np.testing.assert_array_equal(args[2], endpoints[:, 1])
    assert args[3] == ("A", "B")
    assert set(kwargs) == {
        "x_lines",
        "y_lines",
        "textsize",
        "draw_all",
        "draw_lines",
    }
    assert "x_scatter" not in kwargs
    assert "y_scatter" not in kwargs
    assert kwargs["textsize"] == 9
    assert kwargs["draw_all"] is True
    assert kwargs["draw_lines"] is False
    x_lines = kwargs["x_lines"]
    y_lines = kwargs["y_lines"]
    assert len(x_lines) == len(y_lines) == 2
    np.testing.assert_array_equal(x_lines[0], np.array([0.0, 1.5]))
    np.testing.assert_array_equal(y_lines[0], np.array([0.0, -2.0]))
    np.testing.assert_array_equal(x_lines[1], np.array([0.0, -0.25]))
    np.testing.assert_array_equal(y_lines[1], np.array([0.0, 4.0]))


def test_missing_textalloc_falls_back_to_endpoint_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ANNOTATION_LAYOUT, "_textalloc", None)
    axis = _FakeAxis()

    labels = ANNOTATION_LAYOUT.allocate_predictor_labels(
        axis,
        np.array([[1.0, 2.0], [-3.0, 4.0]]),
        ("A", "B"),
        textsize=7,
    )

    assert len(labels) == 2
    assert axis.text_calls == [
        (1.0, 2.0, "A", {"fontsize": 7}),
        (-3.0, 4.0, "B", {"fontsize": 7}),
    ]


def test_annotation_layout_validates_endpoint_and_label_alignment() -> None:
    axis = _FakeAxis()
    with pytest.raises(ValueError, match="shape"):
        ANNOTATION_LAYOUT.allocate_predictor_labels(axis, np.array([1.0, 2.0]), ("A",))
    with pytest.raises(ValueError, match="one label"):
        ANNOTATION_LAYOUT.allocate_predictor_labels(
            axis,
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            ("A",),
        )
