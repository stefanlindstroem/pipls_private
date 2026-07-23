"""Optional Matplotlib figures for fitted-model inspection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .inspection import PiPLSDisplayFactors

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure

FloatArray = NDArray[np.float64]
PredictorStyle: TypeAlias = Literal["bar", "line"]
"""Supported predictor-axis rendering styles."""

__all__ = [
    "PredictorStyle",
    "plot_pipls_dilation",
    "plot_pipls_predictor_directions",
    "plot_pipls_response_directions",
    "plot_pipls_weighted_response_directions",
]



def plot_pipls_predictor_directions(
    factors: PiPLSDisplayFactors,
    *,
    predictor_style: PredictorStyle,
    predictor_names: Sequence[object] | None = None,
    predictor_axis: ArrayLike | None = None,
    predictor_axis_label: str | None = None,
    components: Sequence[int] | None = None,
    title: str = "Pi-PLS predictor directions",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    r"""Plot selected columns of the Pi-PLS predictor-direction matrix $P$.

    Bar rendering groups components for named predictors. Line rendering
    overlays predictor directions on a caller-supplied physical coordinate.
    The function labels component artists but does not create a legend.

    Parameters
    ----------
    factors : pipls.inspection.PiPLSDisplayFactors
        Display-oriented Pi-PLS factors.
    predictor_style : {"bar", "line"}
        Rendering mode for predictor directions.
    predictor_names : sequence of object or None, default=None
        Required predictor labels for bar rendering.
    predictor_axis : array-like of shape (n_features,) or None, default=None
        Required physical coordinate for line rendering.
    predictor_axis_label : str or None, default=None
        Required coordinate label for line rendering.
    components : sequence of int or None, default=None
        Zero-based components to display; ``None`` displays all components.
    title : str, default="Pi-PLS predictor directions"
        Axis title.
    figsize : tuple of float or None, default=None
        Figure size in inches when the function creates the axis. ``None`` uses
        the chart default. It cannot be supplied together with ``ax``.
    ax : matplotlib.axes.Axes or None, default=None
        Existing axis on which to draw. ``None`` creates one figure with one
        axis.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Created figure, or the supplied axis container when ``ax`` is supplied.
    axis : matplotlib.axes.Axes
        Axis containing the chart.
    """

    _require_pipls_display_factors(factors)
    selected = _indices(components, size=factors.n_components, name="components")
    style = _predictor_style(predictor_style)
    feature_labels = _categorical_labels(
        predictor_names,
        size=factors.n_features,
        argument_name="predictor_names",
        required=style == "bar",
    )
    coordinate = _predictor_coordinate(
        predictor_axis,
        size=factors.n_features,
        style=style,
        axis_label=predictor_axis_label,
    )
    figure, axis = _resolve_axis(
        ax,
        figsize=figsize,
        default_figsize=_single_axis_figsize(factors.n_features, style=style),
    )

    selected_array = np.array(selected, dtype=np.int64)
    values = factors.predictor_directions[:, selected_array]
    component_labels = _component_labels(selected)
    if style == "bar":
        assert feature_labels is not None
        _grouped_bars(
            axis,
            values,
            category_labels=feature_labels,
            series_labels=component_labels,
        )
        axis.set_xlabel("Predictor")
    else:
        assert coordinate is not None
        _overlay_lines(
            axis,
            coordinate,
            values,
            series_labels=component_labels,
        )
        axis.set_xlabel(str(predictor_axis_label))
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_ylabel("Predictor direction $P_{:k}$")
    axis.set_title(title)
    return figure, axis


def plot_pipls_dilation(
    factors: PiPLSDisplayFactors,
    *,
    components: Sequence[int] | None = None,
    title: str = "Pi-PLS dilation",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    r"""Plot selected diagonal entries $d_k=D_{kk}$ of the Pi-PLS matrix $D$.

    Parameters
    ----------
    factors : pipls.inspection.PiPLSDisplayFactors
        Display-oriented Pi-PLS factors.
    components : sequence of int or None, default=None
        Zero-based components to display; ``None`` displays all components.
    title : str, default="Pi-PLS dilation"
        Axis title.
    figsize : tuple of float or None, default=None
        Figure size in inches when the function creates the axis. ``None`` uses
        the chart default. It cannot be supplied together with ``ax``.
    ax : matplotlib.axes.Axes or None, default=None
        Existing axis on which to draw. ``None`` creates one figure with one
        axis.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Created figure, or the supplied axis container when ``ax`` is supplied.
    axis : matplotlib.axes.Axes
        Axis containing the chart.
    """

    _require_pipls_display_factors(factors)
    selected = _indices(components, size=factors.n_components, name="components")
    figure, axis = _resolve_axis(
        ax,
        figsize=figsize,
        default_figsize=(6.4, 4.8),
    )

    selected_array = np.array(selected, dtype=np.int64)
    positions = np.arange(len(selected), dtype=np.int64)
    axis.bar(positions, factors.dilation[selected_array])
    axis.set_xticks(positions)
    axis.set_xticklabels(_component_labels(selected))
    axis.set_xlabel("Component")
    axis.set_ylabel("Dilation $d_k$")
    axis.set_title(title)
    return figure, axis


def plot_pipls_response_directions(
    factors: PiPLSDisplayFactors,
    *,
    response_names: Sequence[object] | None = None,
    components: Sequence[int] | None = None,
    title: str = "Pi-PLS response directions",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    r"""Plot selected columns of the Pi-PLS response-direction matrix $Q$.

    Component artists are labeled, but the function does not create a legend.

    Parameters
    ----------
    factors : pipls.inspection.PiPLSDisplayFactors
        Display-oriented Pi-PLS factors.
    response_names : sequence of object
        Required labels for all responses.
    components : sequence of int or None, default=None
        Zero-based components to display; ``None`` displays all components.
    title : str, default="Pi-PLS response directions"
        Axis title.
    figsize : tuple of float or None, default=None
        Figure size in inches when the function creates the axis. ``None`` uses
        the chart default. It cannot be supplied together with ``ax``.
    ax : matplotlib.axes.Axes or None, default=None
        Existing axis on which to draw. ``None`` creates one figure with one
        axis.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Created figure, or the supplied axis container when ``ax`` is supplied.
    axis : matplotlib.axes.Axes
        Axis containing the chart.
    """

    _require_pipls_display_factors(factors)
    return _plot_pipls_response_factor(
        factors,
        values=factors.response_directions,
        response_names=response_names,
        components=components,
        title=title,
        ylabel="Response direction $q_{:k}$",
        figsize=figsize,
        ax=ax,
    )


def plot_pipls_weighted_response_directions(
    factors: PiPLSDisplayFactors,
    *,
    response_names: Sequence[object] | None = None,
    components: Sequence[int] | None = None,
    title: str = "Pi-PLS weighted response directions",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    r"""Plot selected columns of $QD$, namely $d_kq_{:k}$.

    Component artists are labeled, but the function does not create a legend.

    Parameters
    ----------
    factors : pipls.inspection.PiPLSDisplayFactors
        Display-oriented Pi-PLS factors.
    response_names : sequence of object
        Required labels for all responses.
    components : sequence of int or None, default=None
        Zero-based components to display; ``None`` displays all components.
    title : str, default="Pi-PLS weighted response directions"
        Axis title.
    figsize : tuple of float or None, default=None
        Figure size in inches when the function creates the axis. ``None`` uses
        the chart default. It cannot be supplied together with ``ax``.
    ax : matplotlib.axes.Axes or None, default=None
        Existing axis on which to draw. ``None`` creates one figure with one
        axis.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Created figure, or the supplied axis container when ``ax`` is supplied.
    axis : matplotlib.axes.Axes
        Axis containing the chart.
    """

    _require_pipls_display_factors(factors)
    return _plot_pipls_response_factor(
        factors,
        values=factors.weighted_response_directions,
        response_names=response_names,
        components=components,
        title=title,
        ylabel="Weighted response direction $d_k q_{:k}$",
        figsize=figsize,
        ax=ax,
    )


def _plot_pipls_response_factor(
    factors: PiPLSDisplayFactors,
    *,
    values: FloatArray,
    response_names: Sequence[object] | None,
    components: Sequence[int] | None,
    title: str,
    ylabel: str,
    figsize: tuple[float, float] | None,
    ax: Axes | None,
) -> tuple[Figure | SubFigure, Axes]:
    """Plot one response-side Pi-PLS factor matrix."""

    _require_pipls_display_factors(factors)
    selected = _indices(components, size=factors.n_components, name="components")
    target_labels = _categorical_labels(
        response_names,
        size=factors.n_targets,
        argument_name="response_names",
        required=True,
    )
    assert target_labels is not None
    figure, axis = _resolve_axis(
        ax,
        figsize=figsize,
        default_figsize=_single_axis_figsize(factors.n_targets, style="bar"),
    )

    selected_array = np.array(selected, dtype=np.int64)
    _grouped_bars(
        axis,
        values[:, selected_array],
        category_labels=target_labels,
        series_labels=_component_labels(selected),
    )
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_xlabel("Response")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    return figure, axis


def _require_pipls_display_factors(factors: object) -> None:
    """Validate one Pi-PLS display-factor input."""

    if not isinstance(factors, PiPLSDisplayFactors):
        raise TypeError("factors must be a PiPLSDisplayFactors instance.")


def _resolve_axis(
    ax: Axes | None,
    *,
    figsize: tuple[float, float] | None,
    default_figsize: tuple[float, float],
) -> tuple[Figure | SubFigure, Axes]:
    """Return one caller-owned or newly created Matplotlib axis."""

    plt = _pyplot()
    if ax is None:
        size = default_figsize if figsize is None else figsize
        figure: Figure
        axis: Axes
        figure, axis = plt.subplots(figsize=size, layout="constrained")
        return figure, axis
    if figsize is not None:
        raise ValueError("figsize cannot be supplied when ax is provided.")

    from matplotlib.axes import Axes as MatplotlibAxes

    if not isinstance(ax, MatplotlibAxes):
        raise TypeError("ax must be a matplotlib.axes.Axes instance or None.")
    return ax.figure, ax


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised without plot extra
        raise ImportError(
            'Matplotlib is required for pipls.plotting; install it with "pip install pipls[plot]".'
        ) from exc
    return plt


def _predictor_style(value: object) -> PredictorStyle:
    if value not in ("bar", "line"):
        raise ValueError(f'predictor_style must be "bar" or "line"; got {value!r}.')
    return value


def _predictor_coordinate(
    values: ArrayLike | None,
    *,
    size: int,
    style: PredictorStyle,
    axis_label: str | None,
) -> FloatArray | None:
    if style == "bar":
        if values is not None:
            raise ValueError('predictor_axis is only valid when predictor_style="line".')
        if axis_label is not None:
            raise ValueError('predictor_axis_label is only valid when predictor_style="line".')
        return None
    if values is None:
        raise ValueError('predictor_axis is required when predictor_style="line".')
    if not isinstance(axis_label, str) or not axis_label.strip():
        raise ValueError('predictor_axis_label is required when predictor_style="line".')
    coordinate = np.array(values, dtype=np.float64, copy=True)
    if coordinate.ndim != 1 or coordinate.shape[0] != size:
        raise ValueError(
            "predictor_axis must be one-dimensional with one value per predictor: "
            f"expected {(size,)}, got {coordinate.shape}."
        )
    if not np.all(np.isfinite(coordinate)):
        raise ValueError("predictor_axis must contain only finite values.")
    coordinate.setflags(write=False)
    return coordinate


def _optional_labels(
    values: Sequence[object] | None,
    *,
    size: int,
    argument_name: str,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    labels = tuple(str(value) for value in values)
    if len(labels) != size:
        raise ValueError(f"Expected {size} labels in {argument_name}, got {len(labels)}.")
    if any(not label.strip() for label in labels):
        raise ValueError(f"{argument_name} must contain only nonempty labels.")
    return labels


def _categorical_labels(
    values: Sequence[object] | None,
    *,
    size: int,
    argument_name: str,
    required: bool,
) -> tuple[str, ...] | None:
    if values is None:
        if required:
            raise ValueError(
                f"{argument_name} is required for categorical plots so the displayed variables "
                "remain scientifically identifiable."
            )
        return None
    labels = tuple(str(value) for value in values)
    if len(labels) != size:
        raise ValueError(f"Expected {size} labels in {argument_name}, got {len(labels)}.")
    if any(not label.strip() for label in labels):
        raise ValueError(f"{argument_name} must contain only nonempty labels.")
    return labels


def _component_labels(components: Sequence[int]) -> tuple[str, ...]:
    return tuple(f"Component {component + 1}" for component in components)


def _grouped_bars(
    axis: Axes,
    values: FloatArray,
    *,
    category_labels: Sequence[str],
    series_labels: Sequence[str],
) -> None:
    if values.ndim != 2:
        raise ValueError(f"Grouped-bar values must be two-dimensional; got {values.shape}.")
    n_categories, n_series = values.shape
    if len(category_labels) != n_categories or len(series_labels) != n_series:
        raise ValueError("Grouped-bar labels must match the value matrix dimensions.")
    positions = np.arange(n_categories, dtype=np.float64)
    width = 0.8 / n_series
    offsets = (np.arange(n_series, dtype=np.float64) - (n_series - 1) / 2.0) * width
    for column, (offset, label) in enumerate(zip(offsets, series_labels, strict=True)):
        axis.bar(positions + offset, values[:, column], width=width, label=label)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        category_labels,
        rotation=45 if n_categories > 8 else 0,
        ha="right" if n_categories > 8 else "center",
    )


def _overlay_lines(
    axis: Axes,
    coordinate: FloatArray,
    values: FloatArray,
    *,
    series_labels: Sequence[str],
) -> None:
    if values.ndim != 2 or values.shape[0] != coordinate.shape[0]:
        raise ValueError("Line values must have one row per predictor-axis coordinate.")
    if values.shape[1] != len(series_labels):
        raise ValueError("Line labels must match the number of displayed series.")
    for column, label in enumerate(series_labels):
        axis.plot(coordinate, values[:, column], label=label)
    axis.set_xlim(float(coordinate[0]), float(coordinate[-1]))


def _single_axis_figsize(n_categories: int, *, style: PredictorStyle) -> tuple[float, float]:
    if style == "line":
        return 8.0, 4.8
    return max(6.4, min(14.0, 0.55 * n_categories + 3.5)), 4.8


def _indices(values: Sequence[int] | None, *, size: int, name: str) -> tuple[int, ...]:
    if values is None:
        return tuple(range(size))
    indices: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must contain integer indices; got {value!r}.")
        index = int(value)
        if index < 0 or index >= size:
            raise ValueError(f"{name} indices must satisfy 0 <= index < {size}; got {index}.")
        indices.append(index)
    if not indices:
        raise ValueError(f"{name} must contain at least one index.")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} must not contain duplicate indices.")
    return tuple(indices)
