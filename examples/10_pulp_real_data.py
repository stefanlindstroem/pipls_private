"""Fit and inspect a Pulp Pi-PLS model directly in memory."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from pipls import PiPLSPathCV, PiPLSRegression
from pipls.inspection import (
    biplot_coordinates,
    latent_structure,
    pipls_display_factors,
    prediction_diagnostics,
)
from pipls.plotting import (
    plot_biplot,
    plot_coefficients,
    plot_observed_vs_predicted,
    plot_pipls_dilation,
    plot_pipls_predictor_directions,
    plot_pipls_response_directions,
    plot_pipls_weighted_response_directions,
    plot_residuals_vs_predicted,
    plot_scores,
    plot_standardized_rmse,
    plot_x_loadings,
    plot_y_loadings,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "pulp"
ANALYSIS_DIR = Path(__file__).resolve().parent / "results" / "pulp_post_analysis"
CHOSEN_N_COMPONENTS = 3
DISPLAY_COMPONENTS = (0, 1, 2)
DETAILED_RESPONSES = ("CSF", "Density", "TI")

# --8<-- [start:load-pulp-data]
X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")
predictor_names = X.columns.tolist()
response_names = Y.columns.tolist()
# --8<-- [end:load-pulp-data]

# --8<-- [start:evaluate-pulp-component-path]
path_search = PiPLSPathCV(refit=False).fit(X, Y)
path = path_search.component_path_
# --8<-- [end:evaluate-pulp-component-path]

# --8<-- [start:select-pulp-parameters]
# The stored predictor rank minimizes mean CV-MSE for this component count.
selected = path.for_n_components(CHOSEN_N_COMPONENTS)
# --8<-- [end:select-pulp-parameters]

# --8<-- [start:plot-pulp-component-path]
figure, axis = plt.subplots(
    figsize=(7.4, 4.8),
    layout="constrained",
)
axis.errorbar(
    path.n_components,
    path.cv_mse_mean,
    yerr=path.cv_mse_fold_sd,
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
axis.set_ylabel("Response-standardized CV-MSE")
axis.set_title("Pulp Pi-PLS component path")
axis.set_xticks(path.n_components)
axis.grid(axis="y", alpha=0.25)
axis.legend()
figure.savefig(ANALYSIS_DIR / "component_path.pdf")
plt.close(figure)
# --8<-- [end:plot-pulp-component-path]

# --8<-- [start:extract-pulp-rank-profile]
# Retrieve every predictor rank evaluated at the chosen component count.
rank_profile = path_search.predictor_rank_profile(selected.n_components)
# --8<-- [end:extract-pulp-rank-profile]

# Plot the conditional predictor-rank profile at the chosen component count.
figure, axis = plt.subplots(
    figsize=(7.4, 4.8),
    layout="constrained",
)
axis.errorbar(
    rank_profile.predictor_rank,
    rank_profile.cv_mse_mean,
    yerr=rank_profile.cv_mse_fold_sd,
    fmt="o-",
    capsize=4,
)
axis.scatter(
    [rank_profile.selected.predictor_rank],
    [rank_profile.selected.cv_mse_mean],
    marker="D",
    s=70,
    label=f"CV-MSE minimum: rank {rank_profile.selected.predictor_rank}",
    zorder=3,
)
axis.set_xlabel("Predictor rank")
axis.set_ylabel("Response-standardized CV-MSE")
axis.set_title(f"Pulp predictor-rank profile at {selected.n_components} components")
axis.set_xticks(rank_profile.predictor_rank)
axis.grid(axis="y", alpha=0.25)
axis.legend()
figure.savefig(ANALYSIS_DIR / "predictor_rank_profile.pdf")
plt.close(figure)

# Fit the selected fixed model only after inspecting the selection figures.
# --8<-- [start:fit-pulp-model]
model = PiPLSRegression(
    n_components=selected.n_components,
    predictor_rank=selected.predictor_rank,
).fit(X, Y)
# --8<-- [end:fit-pulp-model]

# --8<-- [start:pulp-oof-predictions]
oof_predictions = cross_val_predict(
    model,
    X,
    Y,
    cv=KFold(n_splits=5, shuffle=False),
)
# --8<-- [end:pulp-oof-predictions]

# --8<-- [start:pulp-inspection-results]
factors = pipls_display_factors(model.decomposition_)
structure = latent_structure(model)
diagnostics = prediction_diagnostics(
    Y,
    oof_predictions,
    prediction_kind="selection-conditioned OOF predictions",
)
# --8<-- [end:pulp-inspection-results]

response_index = {name: index for index, name in enumerate(response_names)}
detailed_response_indices = tuple(response_index[name] for name in DETAILED_RESPONSES)

# Plot the Pi-PLS factors.
figure, axes = plt.subplots(
    2,
    2,
    figsize=(13.0, 9.0),
    layout="constrained",
)
plot_pipls_predictor_directions(
    factors,
    predictor_style="bar",
    predictor_names=predictor_names,
    components=DISPLAY_COMPONENTS,
    title=r"Predictor directions $P$",
    ax=axes[0, 0],
)
plot_pipls_dilation(
    factors,
    components=DISPLAY_COMPONENTS,
    title=r"Dilation $D$",
    ax=axes[0, 1],
)
plot_pipls_response_directions(
    factors,
    response_names=response_names,
    components=DISPLAY_COMPONENTS,
    title=r"Response directions $Q$",
    ax=axes[1, 0],
)
plot_pipls_weighted_response_directions(
    factors,
    response_names=response_names,
    components=DISPLAY_COMPONENTS,
    title=r"Weighted response directions $QD$",
    ax=axes[1, 1],
)
axes[0, 0].legend()
axes[1, 0].legend()
axes[1, 1].legend()
for axis in (axes[0, 0], axes[1, 0], axes[1, 1]):
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
figure.suptitle(
    "Pulp Pi-PLS factors "
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
plot_scores(
    structure,
    components=(0, 1),
    title="X scores",
    ax=axes[0, 0],
)
plot_biplot(
    biplot_coordinates(structure, components=(0, 1)),
    predictor_names=predictor_names,
    title="Score-loading biplot",
    ax=axes[0, 1],
)
plot_x_loadings(
    structure,
    predictor_style="bar",
    predictor_names=predictor_names,
    components=DISPLAY_COMPONENTS,
    title="X loadings",
    ax=axes[1, 0],
)
plot_y_loadings(
    structure,
    response_names=response_names,
    components=DISPLAY_COMPONENTS,
    title="Y loadings",
    ax=axes[1, 1],
)
axes[0, 1].legend()
axes[1, 0].legend()
axes[1, 1].legend()
for axis in (axes[1, 0], axes[1, 1]):
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
figure.suptitle("Pulp Pi-PLS latent structure")
figure.savefig(ANALYSIS_DIR / "latent_structure.pdf")
plt.close(figure)

# Plot response-specific regression coefficients.
figure, axis = plt.subplots(
    figsize=(10.0, 5.4),
    layout="constrained",
)
plot_coefficients(
    structure,
    predictor_style="bar",
    predictor_names=predictor_names,
    response_names=response_names,
    responses=detailed_response_indices,
    title="Pulp Pi-PLS regression coefficients",
    ax=axis,
)
axis.legend(title="Response")
axis.tick_params(axis="x", labelrotation=45)
for label in axis.get_xticklabels():
    label.set_horizontalalignment("right")
figure.savefig(ANALYSIS_DIR / "coefficients.pdf")
plt.close(figure)

# Plot selection-conditioned prediction diagnostics.
figure, axes = plt.subplots(
    1,
    3,
    figsize=(14.0, 4.6),
    layout="constrained",
)
plot_observed_vs_predicted(
    diagnostics,
    response_names=response_names,
    responses=detailed_response_indices,
    include_prediction_kind=False,
    ax=axes[0],
)
plot_residuals_vs_predicted(
    diagnostics,
    response_names=response_names,
    responses=detailed_response_indices,
    include_prediction_kind=False,
    ax=axes[1],
)
plot_standardized_rmse(
    diagnostics,
    response_names=response_names,
    include_prediction_kind=False,
    ax=axes[2],
)
axes[0].legend(title="Response")
axes[1].legend(title="Response")
axes[2].tick_params(axis="x", labelrotation=45)
for label in axes[2].get_xticklabels():
    label.set_horizontalalignment("right")
figure.suptitle(f"Pulp Pi-PLS prediction diagnostics\n{diagnostics.prediction_kind}")
figure.savefig(ANALYSIS_DIR / "prediction_diagnostics.pdf")
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
