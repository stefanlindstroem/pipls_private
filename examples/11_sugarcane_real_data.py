"""Fit and inspect a Sugarcane Pi-PLS model directly in memory."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from pipls import PiPLSPathCV, PiPLSRegression
from pipls.inspection import (
    latent_structure,
    pipls_display_factors,
    prediction_diagnostics,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "sugarcane"
ANALYSIS_DIR = Path(__file__).resolve().parent / "results" / "sugarcane_post_analysis"
CHOSEN_N_COMPONENTS = 2

X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")
wavelengths = X.columns.to_numpy(dtype=float)
response_names = Y.columns.tolist()

# Evaluate and plot the Pi-PLS component path.
path_search = PiPLSPathCV(refit=False).fit(X, Y)
path = path_search.component_path_

figure, axis = plt.subplots(
    figsize=(7.0, 4.5),
    layout="constrained",
)
axis.errorbar(
    path.n_components,
    path.cv_mse_mean,
    yerr=path.cv_mse_fold_sd,
    fmt="o-",
    capsize=4,
)
axis.set_xlabel("Number of components")
axis.set_ylabel("Response-standardized CV-MSE")
axis.set_title("Sugarcane Pi-PLS component path")
figure.savefig(ANALYSIS_DIR / "component_path.pdf")
plt.close(figure)

# Fit the selected full-data model.
selected = path.for_n_components(CHOSEN_N_COMPONENTS)
model = PiPLSRegression(
    n_components=selected.n_components,
    predictor_rank=selected.predictor_rank,
).fit(X, Y)

# Calculate selection-conditioned out-of-fold predictions.
oof_predictions = cross_val_predict(
    model,
    X,
    Y,
    cv=KFold(n_splits=5, shuffle=False),
)

# Calculate fitted-model and prediction inspection results.
factors = pipls_display_factors(model.decomposition_)
structure = latent_structure(model)
diagnostics = prediction_diagnostics(
    Y,
    oof_predictions,
    prediction_kind="selection-conditioned OOF predictions",
)

# Plot the Pi-PLS factors directly from their immutable arrays.
figure, axes = plt.subplots(
    2,
    2,
    figsize=(12.0, 9.0),
    layout="constrained",
)
components = tuple(range(factors.n_components))
component_labels = [f"Component {component + 1}" for component in components]
for component, label in zip(components, component_labels, strict=True):
    axes[0, 0].plot(
        wavelengths,
        factors.predictor_directions[:, component],
        label=label,
    )
axes[0, 0].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axes[0, 0].set_xlim(float(wavelengths[0]), float(wavelengths[-1]))
axes[0, 0].set_xlabel("Wavelength (nm)")
axes[0, 0].set_ylabel(r"Predictor direction $P_{:k}$")
axes[0, 0].set_title(r"Predictor directions $P$")
axes[0, 0].legend()

dilation_positions = np.arange(factors.n_components)
axes[0, 1].bar(dilation_positions, factors.dilation)
axes[0, 1].set_xticks(dilation_positions)
axes[0, 1].set_xticklabels(component_labels)
axes[0, 1].set_xlabel("Component")
axes[0, 1].set_ylabel(r"Dilation $d_k$")
axes[0, 1].set_title(r"Dilation $D$")

response_positions = np.arange(len(response_names))
response_width = 0.8 / factors.n_components
for component, label in zip(components, component_labels, strict=True):
    offset = (component - (factors.n_components - 1) / 2.0) * response_width
    axes[1, 0].bar(
        response_positions + offset,
        factors.response_directions[:, component],
        width=response_width,
        label=label,
    )
    axes[1, 1].bar(
        response_positions + offset,
        factors.weighted_response_directions[:, component],
        width=response_width,
        label=label,
    )
for axis, ylabel, title in (
    (axes[1, 0], r"Response direction $q_{:k}$", r"Response directions $Q$"),
    (
        axes[1, 1],
        r"Weighted response direction $d_kq_{:k}$",
        r"Weighted response directions $QD$",
    ),
):
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_xticks(response_positions)
    axis.set_xticklabels(response_names)
    axis.set_xlabel("Response")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
figure.suptitle(
    "Sugarcane Pi-PLS factors "
    f"({selected.n_components} components, predictor rank {selected.predictor_rank})"
)
figure.savefig(ANALYSIS_DIR / "pipls_factors.pdf")
plt.close(figure)

# Plot selection-conditioned prediction diagnostics.
figure, axes = plt.subplots(
    1,
    3,
    figsize=(13.0, 4.2),
    layout="constrained",
)
for response, name in enumerate(response_names):
    axes[0].scatter(
        diagnostics.observed_standardized[:, response],
        diagnostics.predicted_standardized[:, response],
        label=name,
        alpha=0.75,
    )
    axes[1].scatter(
        diagnostics.predicted_standardized[:, response],
        diagnostics.residual_standardized[:, response],
        label=name,
        alpha=0.75,
    )
identity_values = np.concatenate(
    [
        diagnostics.observed_standardized.ravel(),
        diagnostics.predicted_standardized.ravel(),
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
axes[0].plot(identity_limits, identity_limits, linewidth=1.0, linestyle="--", color="0.35")
axes[0].set_xlim(identity_limits)
axes[0].set_ylim(identity_limits)
axes[0].set_xlabel("Observed response (standardized)")
axes[0].set_ylabel("Predicted response (standardized)")
axes[0].set_title("Observed versus predicted")

axes[1].axhline(0.0, linewidth=1.0, linestyle="--", color="0.35")
axes[1].set_xlabel("Predicted response (standardized)")
axes[1].set_ylabel(r"Residual $y-\hat y$ (standardized)")
axes[1].set_title("Residual versus predicted")

positions = np.arange(len(response_names))
axes[2].bar(positions, diagnostics.standardized_rmse)
axes[2].set_xticks(positions)
axes[2].set_xticklabels(response_names)
axes[2].set_xlabel("Response")
axes[2].set_ylabel("Standardized RMSE")
axes[2].set_title("Response-wise standardized RMSE")
axes[0].legend()
axes[1].legend()
figure.suptitle(
    "Sugarcane Pi-PLS prediction diagnostics\n"
    f"{diagnostics.prediction_kind}"
)
figure.savefig(ANALYSIS_DIR / "prediction_diagnostics.pdf")
plt.close(figure)

# Plot scores and loadings from the selected full-data model.
figure, axes = plt.subplots(
    1,
    3,
    figsize=(15.0, 4.5),
    layout="constrained",
    gridspec_kw={"width_ratios": (1.0, 1.6, 1.0)},
)
axes[0].scatter(structure.x_scores[:, 0], structure.x_scores[:, 1], alpha=0.75)
axes[0].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axes[0].axvline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axes[0].set_xlabel("X score component 1")
axes[0].set_ylabel("X score component 2")
axes[0].set_title("X scores")

for component in (0, 1):
    axes[1].plot(
        wavelengths,
        structure.x_loadings[:, component],
        label=f"Component {component + 1}",
    )
axes[1].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axes[1].set_xlabel("Wavelength (nm)")
axes[1].set_ylabel("X loading")
axes[1].set_title("X loadings")
axes[1].legend()

response_positions = np.arange(len(response_names))
component_width = 0.8 / 2
for component in (0, 1):
    offset = (component - 0.5) * component_width
    axes[2].bar(
        response_positions + offset,
        structure.y_loadings[:, component],
        width=component_width,
        label=f"Component {component + 1}",
    )
axes[2].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axes[2].set_xticks(response_positions)
axes[2].set_xticklabels(response_names)
axes[2].set_xlabel("Response")
axes[2].set_ylabel("Y loading")
axes[2].set_title("Y loadings")
axes[2].legend()
figure.suptitle("Sugarcane Pi-PLS latent structure")
figure.savefig(ANALYSIS_DIR / "latent_structure.pdf")
plt.close(figure)

# Plot response-specific regression coefficients.
figure, axis = plt.subplots(
    figsize=(10.0, 5.0),
    layout="constrained",
)
for response, name in enumerate(response_names):
    axis.plot(wavelengths, structure.coefficients[response], label=name)
axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axis.set_xlabel("Wavelength (nm)")
axis.set_ylabel("Regression coefficient")
axis.set_title("Sugarcane Pi-PLS coefficients")
axis.legend(title="Response")
figure.savefig(ANALYSIS_DIR / "coefficients.pdf")
plt.close(figure)

print(f"X shape: {X.shape}; Y shape: {Y.shape}")
print(
    "Selected Pi-PLS: "
    f"n_components={model.n_components}, predictor_rank={model.predictor_rank_}"
)
print(f"Wrote PDF figures to {ANALYSIS_DIR}")
