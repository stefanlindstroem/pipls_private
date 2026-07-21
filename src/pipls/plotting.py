"""Optional Matplotlib figures for fitted-model inspection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .inspection import (
    BiplotCoordinates,
    LatentStructure,
    ObservationDiagnostics,
    PiPLSDisplayFactors,
    PredictionDiagnostics,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure

FloatArray = NDArray[np.float64]
PredictorStyle: TypeAlias = Literal["bar", "line"]
"""Supported predictor-axis rendering styles."""

__all__ = [
    "PredictorStyle",
    "plot_biplot",
    "plot_coefficients",
    "plot_observation_diagnostics",
    "plot_scores",
    "plot_x_loadings",
    "plot_y_loadings",
    "plot_pipls_dilation",
    "plot_pipls_predictor_directions",
    "plot_pipls_response_directions",
    "plot_pipls_weighted_response_directions",
    "plot_observed_vs_predicted",
    "plot_residuals_vs_predicted",
    "plot_standardized_rmse",
]


def plot_biplot(
    coordinates: BiplotCoordinates,
    *,
    predictor_names: Sequence[object],
    sample_names: Sequence[object] | None = None,
    title: str = "PLS-family score-loading biplot",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    r"""Plot balanced sample scores and predictor-loading arrows on one axis.

    The coordinates are calculated by
    :func:`pipls.inspection.biplot_coordinates`. The chart adds no response
    arrows, confidence regions, grouping, or automatic scientific labels. It
    labels the sample artist but does not create a legend; callers may compose
    legends and replace labels or titles through the returned axis.

    Parameters
    ----------
    coordinates : pipls.inspection.BiplotCoordinates
        Balanced coordinates for two components.
    predictor_names : sequence of object
        Required labels for all predictor arrows.
    sample_names : sequence of object or None, default=None
        Optional labels for all sample points.
    title : str, default="PLS-family score-loading biplot"
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

    if not isinstance(coordinates, BiplotCoordinates):
        raise TypeError("coordinates must be a BiplotCoordinates instance.")
    predictor_labels = _categorical_labels(
        predictor_names,
        size=coordinates.n_features,
        argument_name="predictor_names",
        required=True,
    )
    assert predictor_labels is not None
    sample_labels = _optional_labels(
        sample_names,
        size=coordinates.n_samples,
        argument_name="sample_names",
    )

    figure, axis = _resolve_axis(
        ax,
        figsize=figsize,
        default_figsize=(8.0, 6.5),
    )
    axis.scatter(
        coordinates.sample_coordinates[:, 0],
        coordinates.sample_coordinates[:, 1],
        alpha=0.75,
        label="Samples",
    )
    if sample_labels is not None:
        for row, label in enumerate(sample_labels):
            axis.annotate(
                label,
                (
                    coordinates.sample_coordinates[row, 0],
                    coordinates.sample_coordinates[row, 1],
                ),
            )

    from matplotlib.patches import FancyArrowPatch

    for row, label in enumerate(predictor_labels):
        endpoint = coordinates.predictor_coordinates[row, :]
        axis.add_patch(
            FancyArrowPatch(
                (0.0, 0.0),
                (float(endpoint[0]), float(endpoint[1])),
                arrowstyle="->",
                mutation_scale=10.0,
                linewidth=1.0,
            )
        )
        axis.annotate(
            label,
            (endpoint[0], endpoint[1]),
            xytext=(3.0 if endpoint[0] >= 0.0 else -3.0, 3.0 if endpoint[1] >= 0.0 else -3.0),
            textcoords="offset points",
            horizontalalignment="left" if endpoint[0] >= 0.0 else "right",
            verticalalignment="bottom" if endpoint[1] >= 0.0 else "top",
            fontsize="small",
        )

    first, second = (int(value) for value in coordinates.component_indices)
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.axvline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_xlabel(f"Balanced component {first + 1}")
    axis.set_ylabel(f"Balanced component {second + 1}")
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="datalim")
    return figure, axis


def plot_observation_diagnostics(
    diagnostics: ObservationDiagnostics,
    *,
    title: str = "PLS-family observation diagnostics",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    r"""Plot raw score distance against squared X-reconstruction residual.

    No theoretical limits or automatic observation labels are added. Callers
    may replace the semantic axis labels or title through the returned axis.

    Parameters
    ----------
    diagnostics : pipls.inspection.ObservationDiagnostics
        Raw observation diagnostics.
    title : str, default="PLS-family observation diagnostics"
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

    if not isinstance(diagnostics, ObservationDiagnostics):
        raise TypeError("diagnostics must be an ObservationDiagnostics instance.")

    figure, axis = _resolve_axis(
        ax,
        figsize=figsize,
        default_figsize=(6.5, 5.0),
    )
    axis.scatter(
        diagnostics.score_distance,
        diagnostics.x_reconstruction_residual,
        alpha=0.75,
    )
    axis.set_xlabel("Score distance")
    axis.set_ylabel("Squared X-reconstruction residual")
    axis.set_title(title)
    return figure, axis


def plot_scores(
    structure: LatentStructure,
    *,
    components: Sequence[int] = (0, 1),
    sample_names: Sequence[object] | None = None,
    title: str = "PLS-family X scores",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    r"""Plot one pair of PLS-family X-score columns on one axis.

    Parameters
    ----------
    structure : pipls.inspection.LatentStructure
        Extracted fitted PLS-family quantities.
    components : sequence of int, default=(0, 1)
        Exactly two distinct zero-based component indices.
    sample_names : sequence of object or None, default=None
        Optional labels for all sample points.
    title : str, default="PLS-family X scores"
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

    if not isinstance(structure, LatentStructure):
        raise TypeError("structure must be a LatentStructure instance.")
    selected = _indices(components, size=structure.n_components, name="components")
    if len(selected) != 2:
        raise ValueError("components must contain exactly two indices for a score plot.")
    labels = None if sample_names is None else tuple(str(value) for value in sample_names)
    if labels is not None:
        if len(labels) != structure.n_samples:
            raise ValueError(
                f"Expected {structure.n_samples} labels in sample_names, got {len(labels)}."
            )
        if any(not label.strip() for label in labels):
            raise ValueError("sample_names must contain only nonempty labels.")

    first, second = selected
    figure, axis = _resolve_axis(
        ax,
        figsize=figsize,
        default_figsize=(6.4, 5.2),
    )
    axis.scatter(structure.x_scores[:, first], structure.x_scores[:, second], alpha=0.75)
    if labels is not None:
        for row, label in enumerate(labels):
            axis.annotate(
                label,
                (structure.x_scores[row, first], structure.x_scores[row, second]),
            )
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.axvline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_xlabel(f"X score component {first + 1}")
    axis.set_ylabel(f"X score component {second + 1}")
    axis.set_title(title)
    return figure, axis


def plot_x_loadings(
    structure: LatentStructure,
    *,
    predictor_style: PredictorStyle,
    predictor_names: Sequence[object] | None = None,
    predictor_axis: ArrayLike | None = None,
    predictor_axis_label: str | None = None,
    components: Sequence[int] | None = None,
    title: str = "PLS-family X loadings",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    r"""Plot selected PLS-family X loadings on one axis.

    Bar rendering groups component bars by named predictor. Line rendering
    overlays components on a caller-supplied physical predictor coordinate.
    Artists receive component labels, but the function does not create a
    legend; callers own legend placement and styling.

    Parameters
    ----------
    structure : pipls.inspection.LatentStructure
        Extracted fitted PLS-family quantities.
    predictor_style : {"bar", "line"}
        Rendering mode for the predictor axis.
    predictor_names : sequence of object or None, default=None
        Required labels for bar rendering.
    predictor_axis : array-like of shape (n_features,) or None, default=None
        Required physical coordinate for line rendering.
    predictor_axis_label : str or None, default=None
        Required coordinate label for line rendering.
    components : sequence of int or None, default=None
        Zero-based components to display; ``None`` displays all components.
    title : str, default="PLS-family X loadings"
        Axis title.
    figsize : tuple of float or None, default=None
        Figure size in inches when the function creates the axis. ``None``
        chooses a size from the predictor count. It cannot be supplied together
        with ``ax``.
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

    if not isinstance(structure, LatentStructure):
        raise TypeError("structure must be a LatentStructure instance.")
    selected = _indices(components, size=structure.n_components, name="components")
    style = _predictor_style(predictor_style)
    feature_labels = _categorical_labels(
        predictor_names,
        size=structure.n_features,
        argument_name="predictor_names",
        required=style == "bar",
    )
    coordinate = _predictor_coordinate(
        predictor_axis,
        size=structure.n_features,
        style=style,
        axis_label=predictor_axis_label,
    )

    figure, axis = _resolve_axis(
        ax,
        figsize=figsize,
        default_figsize=_single_axis_figsize(structure.n_features, style=style),
    )
    selected_array = np.array(selected, dtype=np.int64)
    component_labels = _component_labels(selected)
    values = structure.x_loadings[:, selected_array]
    if style == "bar":
        assert feature_labels is not None
        _grouped_bars(axis, values, category_labels=feature_labels, series_labels=component_labels)
        axis.set_xlabel("Predictor")
    else:
        assert coordinate is not None
        _overlay_lines(axis, coordinate, values, series_labels=component_labels)
        axis.set_xlabel(str(predictor_axis_label))
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_ylabel("X loading")
    axis.set_title(title)
    return figure, axis


def plot_y_loadings(
    structure: LatentStructure,
    *,
    response_names: Sequence[object] | None = None,
    components: Sequence[int] | None = None,
    title: str = "PLS-family Y loadings",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    r"""Plot selected PLS-family Y loadings as grouped bars on one axis.

    Artists receive component labels, but the function does not create a
    legend; callers own legend placement and styling.

    Parameters
    ----------
    structure : pipls.inspection.LatentStructure
        Extracted fitted PLS-family quantities.
    response_names : sequence of object
        Required labels for all responses.
    components : sequence of int or None, default=None
        Zero-based components to display; ``None`` displays all components.
    title : str, default="PLS-family Y loadings"
        Axis title.
    figsize : tuple of float or None, default=None
        Figure size in inches when the function creates the axis. ``None``
        chooses a size from the response count. It cannot be supplied together
        with ``ax``.
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

    if not isinstance(structure, LatentStructure):
        raise TypeError("structure must be a LatentStructure instance.")
    selected = _indices(components, size=structure.n_components, name="components")
    response_labels = _categorical_labels(
        response_names,
        size=structure.n_targets,
        argument_name="response_names",
        required=True,
    )
    assert response_labels is not None

    figure, axis = _resolve_axis(
        ax,
        figsize=figsize,
        default_figsize=_single_axis_figsize(structure.n_targets, style="bar"),
    )
    selected_array = np.array(selected, dtype=np.int64)
    _grouped_bars(
        axis,
        structure.y_loadings[:, selected_array],
        category_labels=response_labels,
        series_labels=_component_labels(selected),
    )
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_xlabel("Response")
    axis.set_ylabel("Y loading")
    axis.set_title(title)
    return figure, axis


def plot_coefficients(
    structure: LatentStructure,
    *,
    predictor_style: PredictorStyle,
    predictor_names: Sequence[object] | None = None,
    response_names: Sequence[object] | None = None,
    predictor_axis: ArrayLike | None = None,
    predictor_axis_label: str | None = None,
    responses: Sequence[int] | None = None,
    title: str = "PLS-family regression coefficients",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    r"""Plot selected response-specific PLS coefficients on one axis.

    Artists receive response labels, but the function does not create a legend;
    callers own legend placement and styling.

    Parameters
    ----------
    structure : pipls.inspection.LatentStructure
        Extracted fitted PLS-family quantities.
    predictor_style : {"bar", "line"}
        Rendering mode for the predictor axis.
    predictor_names : sequence of object or None, default=None
        Required labels for bar rendering.
    response_names : sequence of object
        Required labels for all responses.
    predictor_axis : array-like of shape (n_features,) or None, default=None
        Required physical coordinate for line rendering.
    predictor_axis_label : str or None, default=None
        Required coordinate label for line rendering.
    responses : sequence of int or None, default=None
        Zero-based responses to display; ``None`` displays all responses.
    title : str, default="PLS-family regression coefficients"
        Axis title.
    figsize : tuple of float or None, default=None
        Figure size in inches when the function creates the axis. ``None``
        chooses a size from the predictor count. It cannot be supplied together
        with ``ax``.
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

    if not isinstance(structure, LatentStructure):
        raise TypeError("structure must be a LatentStructure instance.")
    selected = _indices(responses, size=structure.n_targets, name="responses")
    style = _predictor_style(predictor_style)
    feature_labels = _categorical_labels(
        predictor_names,
        size=structure.n_features,
        argument_name="predictor_names",
        required=style == "bar",
    )
    response_labels = _categorical_labels(
        response_names,
        size=structure.n_targets,
        argument_name="response_names",
        required=True,
    )
    assert response_labels is not None
    coordinate = _predictor_coordinate(
        predictor_axis,
        size=structure.n_features,
        style=style,
        axis_label=predictor_axis_label,
    )

    figure, axis = _resolve_axis(
        ax,
        figsize=figsize,
        default_figsize=_single_axis_figsize(structure.n_features, style=style),
    )
    selected_array = np.array(selected, dtype=np.int64)
    selected_response_labels = tuple(response_labels[index] for index in selected)
    values = structure.coefficients[selected_array, :].T
    if style == "bar":
        assert feature_labels is not None
        _grouped_bars(
            axis,
            values,
            category_labels=feature_labels,
            series_labels=selected_response_labels,
        )
        axis.set_xlabel("Predictor")
    else:
        assert coordinate is not None
        _overlay_lines(
            axis,
            coordinate,
            values,
            series_labels=selected_response_labels,
        )
        axis.set_xlabel(str(predictor_axis_label))
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_ylabel("Regression coefficient")
    axis.set_title(title)
    return figure, axis


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


def plot_observed_vs_predicted(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: Sequence[object] | None = None,
    responses: Sequence[int] | None = None,
    title: str = "Observed versus predicted",
    include_prediction_kind: bool = True,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    r"""Plot standardized observed responses against standardized predictions.

    The dashed identity line indicates exact agreement. Response series are
    labelled, but the function does not create a legend. By default the axis
    title includes ``diagnostics.prediction_kind`` so that prediction
    provenance remains visible in a standalone chart.

    Parameters
    ----------
    diagnostics : pipls.inspection.PredictionDiagnostics
        Standardized prediction diagnostics.
    response_names : sequence of object
        Required labels for all responses.
    responses : sequence of int or None, default=None
        Zero-based responses to display; ``None`` displays all responses.
    title : str, default="Observed versus predicted"
        Axis title before optional prediction provenance.
    include_prediction_kind : bool, default=True
        Whether to append prediction provenance to the axis title. Set this to
        ``False`` when a caller-owned panel reports provenance at figure level.
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

    selected, labels, selected_array = _prediction_plot_inputs(
        diagnostics,
        response_names=response_names,
        responses=responses,
    )
    axis_title = _prediction_title(title, diagnostics, include_prediction_kind)
    figure, axis = _resolve_axis(
        ax,
        figsize=figsize,
        default_figsize=(5.4, 4.8),
    )
    for response in selected:
        axis.scatter(
            diagnostics.observed_standardized[:, response],
            diagnostics.predicted_standardized[:, response],
            label=labels[response],
            alpha=0.75,
        )

    identity_limits = _plot_limits(
        diagnostics.observed_standardized[:, selected_array],
        diagnostics.predicted_standardized[:, selected_array],
    )
    axis.plot(identity_limits, identity_limits, linewidth=1.0, linestyle="--", color="0.35")
    axis.set_xlim(identity_limits)
    axis.set_ylim(identity_limits)
    axis.set_xlabel("Observed response (standardized)")
    axis.set_ylabel("Predicted response (standardized)")
    axis.set_title(axis_title)
    return figure, axis


def plot_residuals_vs_predicted(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: Sequence[object] | None = None,
    responses: Sequence[int] | None = None,
    title: str = "Residual versus predicted",
    include_prediction_kind: bool = True,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    r"""Plot standardized residuals against standardized predictions.

    The dashed horizontal line marks zero residual. Response series are
    labelled, but the function does not create a legend. By default the axis
    title includes ``diagnostics.prediction_kind``.

    Parameters
    ----------
    diagnostics : pipls.inspection.PredictionDiagnostics
        Standardized prediction diagnostics.
    response_names : sequence of object
        Required labels for all responses.
    responses : sequence of int or None, default=None
        Zero-based responses to display; ``None`` displays all responses.
    title : str, default="Residual versus predicted"
        Axis title before optional prediction provenance.
    include_prediction_kind : bool, default=True
        Whether to append prediction provenance to the axis title. Set this to
        ``False`` when a caller-owned panel reports provenance at figure level.
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

    selected, labels, selected_array = _prediction_plot_inputs(
        diagnostics,
        response_names=response_names,
        responses=responses,
    )
    axis_title = _prediction_title(title, diagnostics, include_prediction_kind)
    figure, axis = _resolve_axis(
        ax,
        figsize=figsize,
        default_figsize=(5.4, 4.8),
    )
    for response in selected:
        axis.scatter(
            diagnostics.predicted_standardized[:, response],
            diagnostics.residual_standardized[:, response],
            label=labels[response],
            alpha=0.75,
        )

    prediction_limits = _plot_limits(
        diagnostics.predicted_standardized[:, selected_array],
        diagnostics.predicted_standardized[:, selected_array],
    )
    residual_limits = _plot_limits(
        diagnostics.residual_standardized[:, selected_array],
        diagnostics.residual_standardized[:, selected_array],
    )
    axis.axhline(0.0, linewidth=1.0, linestyle="--", color="0.35")
    axis.set_xlim(prediction_limits)
    axis.set_ylim(residual_limits)
    axis.set_xlabel("Predicted response (standardized)")
    axis.set_ylabel("Residual $y-\\hat y$ (standardized)")
    axis.set_title(axis_title)
    return figure, axis


def plot_standardized_rmse(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: Sequence[object] | None = None,
    responses: Sequence[int] | None = None,
    title: str = "Response-wise standardized RMSE",
    include_prediction_kind: bool = True,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    """Plot response-wise root mean squared standardized residuals.

    Parameters
    ----------
    diagnostics : pipls.inspection.PredictionDiagnostics
        Standardized prediction diagnostics.
    response_names : sequence of object
        Required labels for all responses.
    responses : sequence of int or None, default=None
        Zero-based responses to display; ``None`` displays all responses.
    title : str, default="Response-wise standardized RMSE"
        Axis title before optional prediction provenance.
    include_prediction_kind : bool, default=True
        Whether to append prediction provenance to the axis title. Set this to
        ``False`` when a caller-owned panel reports provenance at figure level.
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

    selected, labels, selected_array = _prediction_plot_inputs(
        diagnostics,
        response_names=response_names,
        responses=responses,
    )
    axis_title = _prediction_title(title, diagnostics, include_prediction_kind)
    figure, axis = _resolve_axis(
        ax,
        figsize=figsize,
        default_figsize=_single_axis_figsize(len(selected), style="bar"),
    )
    selected_labels = [labels[index] for index in selected]
    positions = np.arange(len(selected))
    axis.bar(positions, diagnostics.standardized_rmse[selected_array])
    axis.set_xticks(positions)
    axis.set_xticklabels(
        selected_labels,
        rotation=45 if len(selected) > 6 else 0,
        ha="right" if len(selected) > 6 else "center",
    )
    axis.set_xlabel("Response")
    axis.set_ylabel("Standardized RMSE")
    axis.set_title(axis_title)
    return figure, axis


def _prediction_plot_inputs(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: Sequence[object] | None,
    responses: Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[str, ...], NDArray[np.int64]]:
    """Validate shared prediction-plot inputs."""

    if not isinstance(diagnostics, PredictionDiagnostics):
        raise TypeError("diagnostics must be a PredictionDiagnostics instance.")
    selected = _indices(responses, size=diagnostics.n_targets, name="responses")
    labels = _categorical_labels(
        response_names,
        size=diagnostics.n_targets,
        argument_name="response_names",
        required=True,
    )
    assert labels is not None
    return selected, labels, np.array(selected, dtype=np.int64)


def _prediction_title(
    title: str,
    diagnostics: PredictionDiagnostics,
    include_prediction_kind: bool,
) -> str:
    """Return one axis title with optional prediction provenance."""

    if not isinstance(include_prediction_kind, (bool, np.bool_)):
        raise TypeError("include_prediction_kind must be a boolean.")
    if include_prediction_kind:
        return f"{title}\n{diagnostics.prediction_kind}"
    return title


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


def _plot_limits(first: FloatArray, second: FloatArray) -> tuple[float, float]:
    lower = float(min(np.min(first), np.min(second)))
    upper = float(max(np.max(first), np.max(second)))
    span = upper - lower
    margin = 0.05 * span if span > 0.0 else 1.0
    return lower - margin, upper + margin
