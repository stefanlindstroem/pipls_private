"""Optional Matplotlib figures for fitted-model inspection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .inspection import (
    PiPLSDisplayFactors,
    PLSLatentStructure,
    PLSObservationDiagnostics,
    PredictionDiagnostics,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

FloatArray = NDArray[np.float64]
PredictorStyle: TypeAlias = Literal["bar", "line"]

__all__ = [
    "PredictorStyle",
    "plot_pls_coefficients",
    "plot_pls_observation_diagnostics",
    "plot_pls_scores",
    "plot_pls_x_loadings",
    "plot_pls_y_loadings",
    "plot_pipls_decomposition",
    "plot_prediction_diagnostics",
]


def plot_pls_observation_diagnostics(
    diagnostics: PLSObservationDiagnostics,
    *,
    title: str = "PLS observation diagnostics",
    figsize: tuple[float, float] = (6.5, 5.0),
) -> tuple[Figure, dict[str, Axes]]:
    """Plot raw score distance against squared X-reconstruction residual.

    No theoretical limits or automatic observation labels are added. The raw
    numerical values remain available in ``diagnostics`` for application-level
    interpretation.
    """

    if not isinstance(diagnostics, PLSObservationDiagnostics):
        raise TypeError("diagnostics must be a PLSObservationDiagnostics instance.")

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=figsize, layout="constrained")
    axis.scatter(
        diagnostics.score_distance,
        diagnostics.x_reconstruction_residual,
        alpha=0.75,
    )
    axis.set_xlabel("Score distance")
    axis.set_ylabel("Squared X-reconstruction residual")
    axis.set_title(title)
    return figure, {"observation_diagnostics": axis}


def plot_pls_scores(
    structure: PLSLatentStructure,
    *,
    components: Sequence[int] = (0, 1),
    sample_names: Sequence[object] | None = None,
    title: str = "PLS X scores",
    figsize: tuple[float, float] = (6.4, 5.2),
) -> tuple[Figure, dict[str, Axes]]:
    """Plot one pair of ordinary-PLS X score columns.

    ``components`` contains exactly two distinct zero-based component indices.
    Optional sample labels annotate the points but do not define groups or
    confidence regions.
    """

    if not isinstance(structure, PLSLatentStructure):
        raise TypeError("structure must be a PLSLatentStructure instance.")
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
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=figsize, layout="constrained")
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
    return figure, {"scores": axis}


def plot_pls_x_loadings(
    structure: PLSLatentStructure,
    *,
    predictor_style: PredictorStyle,
    predictor_names: Sequence[object] | None = None,
    predictor_axis: ArrayLike | None = None,
    predictor_axis_label: str | None = None,
    components: Sequence[int] | None = None,
    title: str = "PLS X loadings",
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Plot selected ordinary-PLS X loadings together on one axis.

    Bar rendering groups component bars side by side for each named predictor.
    Line rendering overlays components on the supplied physical coordinate.
    """

    if not isinstance(structure, PLSLatentStructure):
        raise TypeError("structure must be a PLSLatentStructure instance.")
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

    plt = _pyplot()
    if figsize is None:
        figsize = _single_axis_figsize(structure.n_features, style=style)
    figure, axis = plt.subplots(figsize=figsize, layout="constrained")
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
    axis.legend(title="Component")
    return figure, {"x_loadings": axis}


def plot_pls_y_loadings(
    structure: PLSLatentStructure,
    *,
    response_names: Sequence[object] | None = None,
    components: Sequence[int] | None = None,
    title: str = "PLS Y loadings",
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Plot selected ordinary-PLS Y loadings as grouped bars on one axis."""

    if not isinstance(structure, PLSLatentStructure):
        raise TypeError("structure must be a PLSLatentStructure instance.")
    selected = _indices(components, size=structure.n_components, name="components")
    response_labels = _categorical_labels(
        response_names,
        size=structure.n_targets,
        argument_name="response_names",
        required=True,
    )
    assert response_labels is not None

    plt = _pyplot()
    if figsize is None:
        figsize = _single_axis_figsize(structure.n_targets, style="bar")
    figure, axis = plt.subplots(figsize=figsize, layout="constrained")
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
    axis.legend(title="Component")
    return figure, {"y_loadings": axis}


def plot_pls_coefficients(
    structure: PLSLatentStructure,
    *,
    predictor_style: PredictorStyle,
    predictor_names: Sequence[object] | None = None,
    response_names: Sequence[object] | None = None,
    predictor_axis: ArrayLike | None = None,
    predictor_axis_label: str | None = None,
    responses: Sequence[int] | None = None,
    title: str = "PLS regression coefficients",
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Plot selected response-specific PLS coefficients together on one axis."""

    if not isinstance(structure, PLSLatentStructure):
        raise TypeError("structure must be a PLSLatentStructure instance.")
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

    plt = _pyplot()
    if figsize is None:
        figsize = _single_axis_figsize(structure.n_features, style=style)
    figure, axis = plt.subplots(figsize=figsize, layout="constrained")
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
    axis.legend(title="Response")
    return figure, {"coefficients": axis}


def plot_pipls_decomposition(
    factors: PiPLSDisplayFactors,
    *,
    predictor_style: PredictorStyle,
    predictor_names: Sequence[object] | None = None,
    response_names: Sequence[object] | None = None,
    predictor_axis: ArrayLike | None = None,
    predictor_axis_label: str | None = None,
    components: Sequence[int] | None = None,
    title: str = "Pi-PLS decomposition",
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    r"""Plot selected display copies of $P$, $D$, and $QD$ on three shared axes.

    Bar rendering groups components side by side for each named predictor or
    response. Line rendering overlays predictor directions on the supplied
    physical coordinate. The axes dictionary contains ``predictor_directions``,
    ``weighted_response_directions``, and ``dilation``.
    """

    if not isinstance(factors, PiPLSDisplayFactors):
        raise TypeError("factors must be a PiPLSDisplayFactors instance.")
    selected = _indices(
        components,
        size=factors.n_components,
        name="components",
    )
    style = _predictor_style(predictor_style)
    feature_labels = _categorical_labels(
        predictor_names,
        size=factors.n_features,
        argument_name="predictor_names",
        required=style == "bar",
    )
    target_labels = _categorical_labels(
        response_names,
        size=factors.n_targets,
        argument_name="response_names",
        required=True,
    )
    assert target_labels is not None
    coordinate = _predictor_coordinate(
        predictor_axis,
        size=factors.n_features,
        style=style,
        axis_label=predictor_axis_label,
    )

    plt = _pyplot()
    if figsize is None:
        width = max(
            _single_axis_figsize(factors.n_features, style=style)[0],
            _single_axis_figsize(factors.n_targets, style="bar")[0],
        )
        figsize = (width, 10.0)
    figure, axis_array = plt.subplots(3, 1, figsize=figsize, layout="constrained")
    predictor_ax, response_ax, dilation_ax = axis_array
    axes: dict[str, Axes] = {
        "predictor_directions": predictor_ax,
        "weighted_response_directions": response_ax,
        "dilation": dilation_ax,
    }

    selected_array = np.array(selected, dtype=np.int64)
    component_labels = _component_labels(selected)
    predictor_values = factors.predictor_directions[:, selected_array]
    if style == "bar":
        assert feature_labels is not None
        _grouped_bars(
            predictor_ax,
            predictor_values,
            category_labels=feature_labels,
            series_labels=component_labels,
        )
        predictor_ax.set_xlabel("Predictor")
    else:
        assert coordinate is not None
        _overlay_lines(
            predictor_ax,
            coordinate,
            predictor_values,
            series_labels=component_labels,
        )
        predictor_ax.set_xlabel(str(predictor_axis_label))
    predictor_ax.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    predictor_ax.set_ylabel("Predictor direction $P$")
    predictor_ax.set_title("Predictor directions")
    predictor_ax.legend(title="Component")

    _grouped_bars(
        response_ax,
        factors.weighted_response_directions[:, selected_array],
        category_labels=target_labels,
        series_labels=component_labels,
    )
    response_ax.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    response_ax.set_xlabel("Response")
    response_ax.set_ylabel("Weighted response direction $d_k q_{:k}$")
    response_ax.set_title("Weighted response directions")
    response_ax.legend(title="Component")

    component_numbers = selected_array + 1
    dilation_ax.bar(component_numbers, factors.dilation[selected_array])
    dilation_ax.set_xticks(component_numbers)
    dilation_ax.set_xticklabels(component_labels)
    dilation_ax.set_xlabel("Component")
    dilation_ax.set_ylabel("Dilation $d_k$")
    dilation_ax.set_title("Dilation")

    figure.suptitle(title)
    return figure, axes


def plot_prediction_diagnostics(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: Sequence[object] | None = None,
    responses: Sequence[int] | None = None,
    title: str = "Prediction diagnostics",
    figsize: tuple[float, float] = (13.0, 4.2),
) -> tuple[Figure, dict[str, Axes]]:
    """Plot standardized predictions, residuals, and response-wise RMSE.

    Parameters
    ----------
    diagnostics:
        Immutable result returned by :func:`pipls.inspection.prediction_diagnostics`.
    response_names:
        Required scientific labels for all response columns.
    responses:
        Zero-based response indices to display. The default displays all responses.
    title:
        Figure title. The prediction provenance is appended automatically.
    figsize:
        Matplotlib figure size.

    Returns
    -------
    figure, axes:
        The Matplotlib figure and named axes ``observed_vs_predicted``,
        ``residual_vs_predicted``, and ``standardized_rmse``.
    """

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

    plt = _pyplot()
    figure, axis_array = plt.subplots(1, 3, figsize=figsize, layout="constrained")
    observed_ax, residual_ax, rmse_ax = axis_array
    axes: dict[str, Axes] = {
        "observed_vs_predicted": observed_ax,
        "residual_vs_predicted": residual_ax,
        "standardized_rmse": rmse_ax,
    }

    selected_array = np.array(selected, dtype=np.int64)
    for response in selected:
        label = labels[response]
        observed_ax.scatter(
            diagnostics.observed_standardized[:, response],
            diagnostics.predicted_standardized[:, response],
            label=label,
            alpha=0.75,
        )
        residual_ax.scatter(
            diagnostics.predicted_standardized[:, response],
            diagnostics.residual_standardized[:, response],
            label=label,
            alpha=0.75,
        )

    identity_limits = _plot_limits(
        diagnostics.observed_standardized[:, selected_array],
        diagnostics.predicted_standardized[:, selected_array],
    )
    observed_ax.plot(identity_limits, identity_limits, linewidth=1.0, linestyle="--", color="0.35")
    observed_ax.set_xlim(identity_limits)
    observed_ax.set_ylim(identity_limits)
    observed_ax.set_xlabel("Observed response (standardized)")
    observed_ax.set_ylabel("Predicted response (standardized)")
    observed_ax.set_title("Observed versus predicted")

    prediction_limits = _plot_limits(
        diagnostics.predicted_standardized[:, selected_array],
        diagnostics.predicted_standardized[:, selected_array],
    )
    residual_limits = _plot_limits(
        diagnostics.residual_standardized[:, selected_array],
        diagnostics.residual_standardized[:, selected_array],
    )
    residual_ax.axhline(0.0, linewidth=1.0, linestyle="--", color="0.35")
    residual_ax.set_xlim(prediction_limits)
    residual_ax.set_ylim(residual_limits)
    residual_ax.set_xlabel("Predicted response (standardized)")
    residual_ax.set_ylabel("Residual $y-\\hat y$ (standardized)")
    residual_ax.set_title("Residual versus predicted")

    selected_labels = [labels[index] for index in selected]
    positions = np.arange(len(selected))
    rmse_ax.bar(positions, diagnostics.standardized_rmse[selected_array])
    rmse_ax.set_xticks(positions)
    rmse_ax.set_xticklabels(
        selected_labels,
        rotation=45 if len(selected) > 6 else 0,
        ha="right" if len(selected) > 6 else "center",
    )
    rmse_ax.set_xlabel("Response")
    rmse_ax.set_ylabel("Standardized RMSE")
    rmse_ax.set_title("Response-wise error")

    if len(selected) > 1:
        observed_ax.legend()
        residual_ax.legend()

    figure.suptitle(f"{title}\n{diagnostics.prediction_kind}")
    return figure, axes


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
