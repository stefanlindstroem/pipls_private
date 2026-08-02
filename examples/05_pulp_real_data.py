"""Fit and inspect a Pulp Pi-PLS model directly in memory."""

# --8<-- [start:pulp-tutorial-setup]
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.patches import FancyArrowPatch
from sklearn.model_selection import KFold

from pipls import PiPLSSearchCV
from pipls.datasets import load_pulp
from pipls.inspection import (
    biplot_coordinates,
    latent_structure,
    pipls_display_factors,
    prediction_diagnostics,
)

ANALYSIS_DIR = Path(__file__).resolve().parent / "results" / "pulp_post_analysis"
CHOSEN_N_COMPONENTS = 3
DETAILED_RESPONSE_COUNT = 3
CV = KFold(n_splits=5, shuffle=True, random_state=0)
# --8<-- [end:pulp-tutorial-setup]

# --8<-- [start:load-pulp-data]
data = load_pulp()
X, Y = data.data, data.target
predictor_names = data.feature_names
response_names = data.target_names
# --8<-- [end:load-pulp-data]

# --8<-- [start:evaluate-pulp-component-path]
search = PiPLSSearchCV(cv=CV).fit(X, Y)
path = search.component_path_
# --8<-- [end:evaluate-pulp-component-path]

# --8<-- [start:select-pulp-parameters]
# The stored predictor rank minimizes mean CV-MSE for this paired-mode count.
selected = search.select(n_components=CHOSEN_N_COMPONENTS)
# --8<-- [end:select-pulp-parameters]

# --8<-- [start:plot-pulp-component-path]
figure, axis = plt.subplots(
    figsize=(7.4, 4.8),
    layout="constrained",
)
axis.errorbar(
    path.n_components,
    path.cv_mse_mean,
    yerr=path.cv_mse_standard_error,
    fmt="o-",
    capsize=4,
)
axis.scatter(
    [selected.n_components],
    [selected.cv_mse_mean],
    marker="D",
    s=70,
    label=f"Chosen: {selected.n_components} components",
    zorder=3,
)
axis.set_xlabel("Number of components")
axis.set_ylabel("Mean response-standardized CV-MSE (±1 SE)")
axis.set_title(r"Pulp $\Pi$-PLS component path")
axis.set_xticks(path.n_components)
upper = float(np.max(path.cv_mse_mean + path.cv_mse_standard_error))
axis.set_ylim(0.0, max(1.0, 1.05 * upper))
axis.grid(axis="y", alpha=0.25)
axis.legend()
figure.savefig(ANALYSIS_DIR / "component_path.pdf")
plt.close(figure)
# --8<-- [end:plot-pulp-component-path]

# --8<-- [start:extract-pulp-rank-profile]
# Retrieve every predictor rank evaluated at the chosen component count.
rank_profile = search.predictor_rank_profile(CHOSEN_N_COMPONENTS)
# --8<-- [end:extract-pulp-rank-profile]

# --8<-- [start:plot-pulp-rank-profile]
# Plot the conditional predictor-rank profile at the chosen component count.
figure, axis = plt.subplots(
    figsize=(7.4, 4.8),
    layout="constrained",
)
axis.errorbar(
    rank_profile.predictor_rank,
    rank_profile.cv_mse_mean,
    yerr=rank_profile.cv_mse_standard_error,
    fmt="o-",
    capsize=4,
)
axis.scatter(
    [rank_profile.selected_result.predictor_rank],
    [rank_profile.selected_result.cv_mse_mean],
    marker="D",
    s=70,
    label=f"CV-MSE minimum: rank {rank_profile.selected_result.predictor_rank}",
    zorder=3,
)
axis.set_xlabel("Predictor rank")
axis.set_ylabel("Mean response-standardized CV-MSE (±1 SE)")
axis.set_title(
    rf"Pulp $\Pi$-PLS predictor-rank profile at "
    f"{selected.n_components} components"
)
axis.set_xticks(rank_profile.predictor_rank)
upper = float(
    np.max(rank_profile.cv_mse_mean + rank_profile.cv_mse_standard_error)
)
axis.set_ylim(0.0, max(1.0, 1.05 * upper))
axis.grid(axis="y", alpha=0.25)
axis.legend()
figure.savefig(ANALYSIS_DIR / "predictor_rank_profile.pdf")
plt.close(figure)
# --8<-- [end:plot-pulp-rank-profile]

# Fit the selected fixed model only after inspecting the selection figures.
# --8<-- [start:fit-pulp-model]
model = search.refit(
    X,
    Y,
    n_components=CHOSEN_N_COMPONENTS,
)
# --8<-- [end:fit-pulp-model]

# --8<-- [start:pulp-oof-predictions]
report = search.validation_report(
    X,
    Y,
    n_components=CHOSEN_N_COMPONENTS,
)
if report.oof_predictions is None:
    raise RuntimeError("Validation reporting did not produce OOF predictions.")
oof_predictions = report.oof_predictions
# --8<-- [end:pulp-oof-predictions]

# --8<-- [start:pulp-inspection-results]
factors = pipls_display_factors(
    model.decomposition_,
    response_index=response_names.index("TI"),
    response_sign="positive",
)
structure = latent_structure(model)
diagnostics = prediction_diagnostics(
    Y,
    oof_predictions,
    prediction_kind="selection-conditioned OOF predictions",
)
# --8<-- [end:pulp-inspection-results]

# Every fitted component is displayed, so derive the zero-based indices locally.
display_components = tuple(range(CHOSEN_N_COMPONENTS))

# Plot the Pi-PLS factors directly from their immutable arrays.
figure, axes = plt.subplots(
    2,
    2,
    figsize=(13.0, 9.0),
    layout="constrained",
)
component_indices = np.array(display_components, dtype=np.int64)
component_labels = [f"Component {component + 1}" for component in display_components]
predictor_positions = np.arange(len(predictor_names))
predictor_width = 0.8 / CHOSEN_N_COMPONENTS
for series, component in enumerate(display_components):
    offset = (series - (CHOSEN_N_COMPONENTS - 1) / 2.0) * predictor_width
    axes[0, 0].bar(
        predictor_positions + offset,
        factors.predictor_directions[:, component],
        width=predictor_width,
        label=component_labels[series],
    )
axes[0, 0].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axes[0, 0].set_xticks(predictor_positions)
axes[0, 0].set_xticklabels(predictor_names, rotation=45, ha="right")
axes[0, 0].set_xlabel("Predictor")
axes[0, 0].set_ylabel(r"Predictor direction $P_{:k}$")
axes[0, 0].legend()

dilation_positions = np.arange(CHOSEN_N_COMPONENTS)
axes[0, 1].bar(dilation_positions, factors.dilation[component_indices])
axes[0, 1].set_xticks(dilation_positions)
axes[0, 1].set_xticklabels(np.arange(1, CHOSEN_N_COMPONENTS + 1))
axes[0, 1].set_xlabel("Component")
axes[0, 1].set_ylabel(r"Dilation $d_k$")

response_positions = np.arange(len(response_names))
response_width = 0.8 / CHOSEN_N_COMPONENTS
for series, component in enumerate(display_components):
    offset = (series - (CHOSEN_N_COMPONENTS - 1) / 2.0) * response_width
    axes[1, 0].bar(
        response_positions + offset,
        factors.response_directions[:, component],
        width=response_width,
        label=component_labels[series],
    )
    axes[1, 1].bar(
        response_positions + offset,
        factors.weighted_response_directions[:, component],
        width=response_width,
        label=component_labels[series],
    )
for axis, ylabel in (
    (axes[1, 0], r"Response direction $Q_{:k}$"),
    (axes[1, 1], r"Weighted response direction $d_kQ_{:k}$"),
):
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_xticks(response_positions)
    axis.set_xticklabels(response_names, rotation=45, ha="right")
    axis.set_xlabel("Response")
    axis.set_ylabel(ylabel)
    axis.legend()
figure.suptitle(
    r"Pulp $\Pi$-PLS factors with TI-positive orientation "
    f"({selected.n_components} components, predictor rank {selected.predictor_rank})"
)
figure.savefig(ANALYSIS_DIR / "pipls_factors.pdf")
plt.close(figure)

# Plot scores, a score-loading biplot, and loadings.
figure, axes = plt.subplots(
    2,
    2,
    figsize=(13.0, 10.0),
    layout="constrained",
)
score_components = (0, 1)
axes[0, 0].scatter(
    structure.x_scores[:, score_components[0]],
    structure.x_scores[:, score_components[1]],
    alpha=0.75,
)
axes[0, 0].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axes[0, 0].axvline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axes[0, 0].set_xlabel(f"X score component {score_components[0] + 1}")
axes[0, 0].set_ylabel(f"X score component {score_components[1] + 1}")
# --8<-- [start:plot-pulp-biplot]
biplot = biplot_coordinates(structure, components=(0, 1))
sample_xy = biplot.sample_coordinates
predictor_xy = biplot.predictor_coordinates
biplot_axis = axes[0, 1]

biplot_axis.scatter(
    sample_xy[:, 0],
    sample_xy[:, 1],
    alpha=0.75,
    label="Samples",
)
for endpoint in predictor_xy:
    biplot_axis.add_patch(
        FancyArrowPatch(
            (0.0, 0.0),
            (float(endpoint[0]), float(endpoint[1])),
            arrowstyle="->",
            mutation_scale=10.0,
            linewidth=1.0,
        )
    )

predictor_labels = [
    biplot_axis.text(
        float(endpoint[0]),
        float(endpoint[1]),
        name,
        fontsize="small",
    )
    for endpoint, name in zip(predictor_xy, predictor_names, strict=True)
]
first, second = (int(value) for value in biplot.component_indices)
biplot_axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
biplot_axis.axvline(0.0, linewidth=0.8, linestyle="--", color="0.45")
biplot_axis.set_xlabel(f"Balanced component {first + 1}")
biplot_axis.set_ylabel(f"Balanced component {second + 1}")
biplot_axis.set_aspect("equal", adjustable="datalim")
biplot_axis.margins(0.1)
biplot_axis.legend()

adjust_text(
    predictor_labels,
    x=sample_xy[:, 0],
    y=sample_xy[:, 1],
    target_x=predictor_xy[:, 0],
    target_y=predictor_xy[:, 1],
    ax=biplot_axis,
    ensure_inside_axes=True,
    prevent_crossings=False,
    iter_lim=200,
    arrowprops={"arrowstyle": "-", "linewidth": 0.6},
)
# --8<-- [end:plot-pulp-biplot]
predictor_positions = np.arange(len(predictor_names))
component_width = 0.8 / CHOSEN_N_COMPONENTS
for series, component in enumerate(display_components):
    offset = (series - (CHOSEN_N_COMPONENTS - 1) / 2) * component_width
    axes[1, 0].bar(
        predictor_positions + offset,
        structure.x_loadings[:, component],
        width=component_width,
        label=f"Component {component + 1}",
    )
axes[1, 0].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axes[1, 0].set_xticks(predictor_positions)
axes[1, 0].set_xticklabels(predictor_names)
axes[1, 0].set_xlabel("Predictor")
axes[1, 0].set_ylabel("X loading")
axes[1, 0].legend()

response_positions = np.arange(len(response_names))
for series, component in enumerate(display_components):
    offset = (series - (CHOSEN_N_COMPONENTS - 1) / 2) * component_width
    axes[1, 1].bar(
        response_positions + offset,
        structure.y_loadings[:, component],
        width=component_width,
        label=f"Component {component + 1}",
    )
axes[1, 1].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axes[1, 1].set_xticks(response_positions)
axes[1, 1].set_xticklabels(response_names)
axes[1, 1].set_xlabel("Response")
axes[1, 1].set_ylabel("Y loading")
axes[1, 1].legend()
for axis in (axes[1, 0], axes[1, 1]):
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
figure.suptitle(r"Pulp $\Pi$-PLS latent structure")
figure.savefig(ANALYSIS_DIR / "latent_structure.pdf")
plt.close(figure)

# --8<-- [start:plot-pulp-prediction-diagnostics]
# Plot selection-conditioned prediction diagnostics.
figure, axes = plt.subplots(
    1,
    3,
    figsize=(14.0, 4.6),
    layout="constrained",
)
detailed_response_indices = tuple(range(DETAILED_RESPONSE_COUNT))
detailed_array = np.array(detailed_response_indices, dtype=np.int64)
for response in detailed_response_indices:
    axes[0].scatter(
        diagnostics.observed_standardized[:, response],
        diagnostics.predicted_standardized[:, response],
        label=response_names[response],
        alpha=0.75,
    )
    axes[1].scatter(
        diagnostics.predicted_standardized[:, response],
        diagnostics.residual_standardized[:, response],
        label=response_names[response],
        alpha=0.75,
    )
identity_values = np.concatenate(
    [
        diagnostics.observed_standardized[:, detailed_array].ravel(),
        diagnostics.predicted_standardized[:, detailed_array].ravel(),
    ]
)
identity_lower = float(identity_values.min())
identity_upper = float(identity_values.max())
identity_margin = (
    0.05 * (identity_upper - identity_lower)
    if identity_upper > identity_lower
    else 1.0
)
identity_limits = (
    identity_lower - identity_margin,
    identity_upper + identity_margin,
)
axes[0].plot(
    identity_limits,
    identity_limits,
    linewidth=1.0,
    linestyle="--",
    color="0.35",
)
axes[0].set_xlim(identity_limits)
axes[0].set_ylim(identity_limits)
axes[0].set_xlabel("Observed response (standardized)")
axes[0].set_ylabel("Predicted response (standardized)")

predicted = diagnostics.predicted_standardized[:, detailed_array]
residual = diagnostics.residual_standardized[:, detailed_array]
predicted_span = float(predicted.max() - predicted.min())
residual_span = float(residual.max() - residual.min())
axes[1].set_xlim(
    float(predicted.min()) - (0.05 * predicted_span if predicted_span > 0.0 else 1.0),
    float(predicted.max()) + (0.05 * predicted_span if predicted_span > 0.0 else 1.0),
)
axes[1].set_ylim(
    float(residual.min()) - (0.05 * residual_span if residual_span > 0.0 else 1.0),
    float(residual.max()) + (0.05 * residual_span if residual_span > 0.0 else 1.0),
)
axes[1].axhline(0.0, linewidth=1.0, linestyle="--", color="0.35")
axes[1].set_xlabel("Predicted response (standardized)")
axes[1].set_ylabel("Standardized residual")

positions = np.arange(len(response_names))
axes[2].bar(positions, diagnostics.standardized_rmse)
axes[2].set_xticks(positions)
axes[2].set_xticklabels(response_names)
axes[2].set_xlabel("Response")
axes[2].set_ylabel("Standardized RMSE")
axes[0].legend(title="Response")
axes[1].legend(title="Response")
axes[2].tick_params(axis="x", labelrotation=45)
for label in axes[2].get_xticklabels():
    label.set_horizontalalignment("right")
figure.suptitle(
    rf"Pulp $\Pi$-PLS prediction diagnostics — {diagnostics.prediction_kind}"
)
figure.savefig(ANALYSIS_DIR / "prediction_diagnostics.pdf")
plt.close(figure)
# --8<-- [end:plot-pulp-prediction-diagnostics]

# Plot response-specific regression coefficients.
figure, axis = plt.subplots(
    figsize=(10.0, 5.4),
    layout="constrained",
)
response_width = 0.8 / len(detailed_response_indices)
for series, response in enumerate(detailed_response_indices):
    offset = (series - (len(detailed_response_indices) - 1) / 2) * response_width
    axis.bar(
        predictor_positions + offset,
        structure.coefficients[response],
        width=response_width,
        label=response_names[response],
    )
axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axis.set_xticks(predictor_positions)
axis.set_xticklabels(predictor_names)
axis.set_xlabel("Predictor")
axis.set_ylabel("Regression coefficient")
axis.legend(title="Response")
axis.tick_params(axis="x", labelrotation=45)
for label in axis.get_xticklabels():
    label.set_horizontalalignment("right")
figure.savefig(ANALYSIS_DIR / "coefficients.pdf")
plt.close(figure)

print(f"X shape: {X.shape}; Y shape: {Y.shape}")
print(
    "Selected Pi-PLS: "
    f"n_components={model.n_components}, predictor_rank={model.predictor_rank_}"
)
print(
    "Predictor-rank profile: "
    f"evaluated {rank_profile.predictor_rank[0]} to "
    f"{rank_profile.predictor_rank[-1]}; "
    f"selected rank {selected.predictor_rank}"
)
print(f"Wrote PDF figures to {ANALYSIS_DIR}")
