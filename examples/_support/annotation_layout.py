"""Example-local text allocation for annotated biplots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    import textalloc as _textalloc
except ModuleNotFoundError as exc:
    if exc.name != "textalloc":
        raise
    _textalloc = None

FloatArray = NDArray[np.float64]


def _as_endpoints(values: ArrayLike) -> FloatArray:
    endpoints = np.asarray(values, dtype=np.float64)
    if endpoints.ndim != 2 or endpoints.shape[1] != 2:
        raise ValueError("predictor_endpoints must have shape (n_predictors, 2).")
    if not np.isfinite(endpoints).all():
        raise ValueError("predictor_endpoints must contain only finite values.")
    return endpoints


def _arrow_segments(endpoints: FloatArray) -> tuple[list[FloatArray], list[FloatArray]]:
    x_lines = [np.array([0.0, endpoint[0]], dtype=np.float64) for endpoint in endpoints]
    y_lines = [np.array([0.0, endpoint[1]], dtype=np.float64) for endpoint in endpoints]
    return x_lines, y_lines


def allocate_predictor_labels(
    axis: Any,
    predictor_endpoints: ArrayLike,
    predictor_names: Sequence[str],
    *,
    textsize: int = 8,
) -> tuple[Any, ...]:
    """Place predictor labels while treating predictor arrows as line obstacles.

    When :mod:`textalloc` is available, the allocator avoids already allocated labels and the
    predictor-arrow shafts. Sample-score points are intentionally not supplied as obstacles. When
    :mod:`textalloc` is unavailable, labels are drawn directly at their predictor endpoints.
    """

    endpoints = _as_endpoints(predictor_endpoints)
    names = tuple(str(name) for name in predictor_names)
    if len(names) != endpoints.shape[0]:
        raise ValueError(
            "predictor_names must contain one label for each predictor endpoint."
        )
    if isinstance(textsize, bool) or not isinstance(textsize, (int, np.integer)):
        raise TypeError("textsize must be an integer.")
    textsize = int(textsize)
    if textsize <= 0:
        raise ValueError("textsize must be positive.")

    if _textalloc is None:
        return tuple(
            axis.text(float(endpoint[0]), float(endpoint[1]), name, fontsize=textsize)
            for endpoint, name in zip(endpoints, names, strict=True)
        )

    x_lines, y_lines = _arrow_segments(endpoints)
    result = _textalloc.allocate(
        axis,
        endpoints[:, 0],
        endpoints[:, 1],
        names,
        x_lines=x_lines,
        y_lines=y_lines,
        textsize=textsize,
        draw_all=True,
        draw_lines=False,
    )
    return tuple(result[2])
