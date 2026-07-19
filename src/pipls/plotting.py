"""Optional Matplotlib figures for fitted-model inspection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .inspection import PiPLSDisplayFactors, PredictionDiagnostics

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

FloatArray = NDArray[np.float64]
PredictorStyle: TypeAlias = Literal["bar", "line"]

__all__ = [
    "PredictorStyle",
    "plot_pipls_decomposition",
    "plot_prediction_diagnostics",
]


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
    r"""Plot display copies of $P$, $D$, and $QD$.

    Parameters
    ----------
    factors:
        Immutable factors returned by :func:`pipls.inspection.pipls_display_factors`.
    predictor_style:
        ``"bar"`` for a small scalar predictor set or ``"line"`` for an ordered
        physical coordinate.
    predictor_names, response_names:
        Optional labels. Predictor names are used only for bar plots.
    predictor_axis, predictor_axis_label:
        Required for line plots. Values are used in the supplied order.
    components:
        Zero-based component indices to display. The default displays all components.
    title:
        Figure title.
    figsize:
        Optional Matplotlib figure size.

    Returns
    -------
    figure, axes:
        The Matplotlib figure and a dictionary of named axes. Predictor and weighted
        response axes use keys ``predictor_component_<k>`` and
        ``response_component_<k>`` with one-based component numbers. The dilation
        axis uses the key ``dilation``.
    """

    if not isinstance(factors, PiPLSDisplayFactors):
        raise TypeError("factors must be a PiPLSDisplayFactors instance.")
    selected = _indices(
        components,
        size=factors.n_components,
        name="components",
    )
    style = _predictor_style(predictor_style)
    feature_labels = _labels(predictor_names, size=factors.n_features, prefix="Feature")
    target_labels = _labels(response_names, size=factors.n_targets, prefix="Response")
    coordinate = _predictor_coordinate(
        predictor_axis,
        size=factors.n_features,
        style=style,
        axis_label=predictor_axis_label,
    )

    plt = _pyplot()
    n_selected = len(selected)
    if figsize is None:
        figsize = (max(5.0, 4.0 * n_selected), 8.5)
    figure = plt.figure(figsize=figsize, layout="constrained")
    grid = figure.add_gridspec(3, n_selected, height_ratios=(1.0, 1.0, 0.65))
    axes: dict[str, Axes] = {}

    for column, component in enumerate(selected):
        component_number = component + 1
        predictor_ax = figure.add_subplot(grid[0, column])
        response_ax = figure.add_subplot(grid[1, column])
        axes[f"predictor_component_{component_number}"] = predictor_ax
        axes[f"response_component_{component_number}"] = response_ax

        predictor_values = factors.predictor_directions[:, component]
        if style == "bar":
            positions = np.arange(factors.n_features)
            predictor_ax.bar(positions, predictor_values)
            predictor_ax.set_xticks(positions)
            predictor_ax.set_xticklabels(
                feature_labels,
                rotation=45 if factors.n_features > 8 else 0,
                ha="right" if factors.n_features > 8 else "center",
            )
            predictor_ax.set_xlabel("Predictor")
        else:
            assert coordinate is not None
            predictor_ax.plot(coordinate, predictor_values)
            predictor_ax.set_xlabel(str(predictor_axis_label))
        predictor_ax.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
        predictor_ax.set_ylabel("Predictor direction $P$")
        predictor_ax.set_title(f"Component {component_number}")

        positions = np.arange(factors.n_targets)
        response_ax.bar(
            positions,
            factors.weighted_response_directions[:, component],
        )
        response_ax.set_xticks(positions)
        response_ax.set_xticklabels(
            target_labels,
            rotation=45 if factors.n_targets > 6 else 0,
            ha="right" if factors.n_targets > 6 else "center",
        )
        response_ax.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
        response_ax.set_xlabel("Response")
        response_ax.set_ylabel("Weighted response direction $d_k q_{:k}$")
        response_ax.set_title(f"Component {component_number}")

    dilation_ax = figure.add_subplot(grid[2, :])
    axes["dilation"] = dilation_ax
    component_numbers = np.array(selected, dtype=np.int64) + 1
    dilation_ax.bar(component_numbers, factors.dilation[np.array(selected, dtype=np.int64)])
    dilation_ax.set_xticks(component_numbers)
    dilation_ax.set_xlabel("Component")
    dilation_ax.set_ylabel("Dilation $d_k$")

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
        Optional labels for all response columns.
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
    labels = _labels(response_names, size=diagnostics.n_targets, prefix="Response")

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


def _labels(values: Sequence[object] | None, *, size: int, prefix: str) -> tuple[str, ...]:
    if values is None:
        return tuple(f"{prefix} {index + 1}" for index in range(size))
    labels = tuple(str(value) for value in values)
    if len(labels) != size:
        raise ValueError(f"Expected {size} {prefix.lower()} labels, got {len(labels)}.")
    if any(not label.strip() for label in labels):
        raise ValueError(f"{prefix} labels must be nonempty.")
    return labels


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
