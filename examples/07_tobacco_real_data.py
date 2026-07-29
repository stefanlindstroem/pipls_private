"""Fit and inspect a Tobacco Pi-PLS model directly in memory."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import KFold, cross_val_predict

from pipls import PiPLSRegression, PiPLSSearchCV
from pipls.inspection import (
    latent_structure,
    observation_diagnostics,
    pipls_display_factors,
    prediction_diagnostics,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "tobacco"
ANALYSIS_DIR = Path(__file__).resolve().parent / "results" / "tobacco_post_analysis"
DISPLAY_COMPONENT_COUNT = 4
RESPONSES_PER_PAGE = 5

X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")
wavenumbers = X.columns.to_numpy(dtype=float)
response_names = Y.columns.tolist()
response_pages = tuple(
    tuple(range(start, min(start + RESPONSES_PER_PAGE, len(response_names))))
    for start in range(0, len(response_names), RESPONSES_PER_PAGE)
)

# Evaluate the Pi-PLS component path over paired-mode counts with a full predictor SVD.
path_search = PiPLSSearchCV(
    estimator=PiPLSRegression(
        n_components=1,
        predictor_rank=1,
        svd_solver="full",
    ),
    search_method="auto",
    n_jobs=1,
).fit(X, Y)
path = path_search.component_path_
minimum = path.minimum_cv_mse_result()
selected = path.one_standard_error_result()
one_se_threshold = minimum.cv_mse_mean + minimum.cv_mse_standard_error
display_components = tuple(
    range(min(DISPLAY_COMPONENT_COUNT, selected.n_components))
)

figure, axis = plt.subplots(
    figsize=(7.0, 4.5),
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
    [minimum.n_components],
    [minimum.cv_mse_mean],
    marker="X",
    s=70,
    label=f"Minimum mean CV-MSE: {minimum.n_components} components",
    zorder=3,
)
axis.axhline(
    one_se_threshold,
    linewidth=1.2,
    linestyle="--",
    color="0.35",
    label="1-SE threshold",
)
axis.scatter(
    [selected.n_components],
    [selected.cv_mse_mean],
    marker="D",
    s=70,
    label=(
        f"1-SE recommendation: {selected.n_components} components, "
        f"predictor rank {selected.predictor_rank}"
    ),
    zorder=3,
)
axis.set_xlabel("Number of components")
axis.set_ylabel("Mean response-standardized CV-MSE (±1 SE)")
axis.set_title(r"Tobacco $\Pi$-PLS component path")
axis.set_xticks(path.n_components)
upper = max(
    float(np.max(path.cv_mse_mean + path.cv_mse_standard_error)),
    float(one_se_threshold),
)
axis.set_ylim(0.0, max(1.0, 1.05 * upper))
axis.grid(axis="y", alpha=0.25)
axis.legend()
figure.savefig(ANALYSIS_DIR / "component_path.pdf")
plt.close(figure)

# Fit the selected full-data model and calculate selection-conditioned OOF predictions.
model = PiPLSRegression(
    n_components=selected.n_components,
    predictor_rank=selected.predictor_rank,
    svd_solver="full",
).fit(X, Y)
oof_predictions = cross_val_predict(
    model,
    X,
    Y,
    cv=KFold(n_splits=5, shuffle=False),
)

# Calculate immutable fitted-model and prediction inspection results.
factors = pipls_display_factors(model.decomposition_)
structure = latent_structure(model)
diagnostics = prediction_diagnostics(
    Y,
    oof_predictions,
    prediction_kind="selection-conditioned OOF predictions",
)
observations = observation_diagnostics(model, X)

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
        wavenumbers,
        factors.predictor_directions[:, component],
        label=label,
    )
axes[0, 0].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axes[0, 0].set_xlim(float(wavenumbers[0]), float(wavenumbers[-1]))
axes[0, 0].set_xlabel("Wavenumber (cm$^{-1}$)")
axes[0, 0].set_ylabel(r"Predictor direction $P_{:k}$")
axes[0, 0].legend()

dilation_positions = np.arange(factors.n_components)
axes[0, 1].bar(dilation_positions, factors.dilation)
axes[0, 1].set_xticks(dilation_positions)
axes[0, 1].set_xticklabels(np.arange(1, factors.n_components + 1))
axes[0, 1].set_xlabel("Component")
axes[0, 1].set_ylabel(r"Dilation $d_k$")

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
    r"Tobacco $\Pi$-PLS factors "
    f"({selected.n_components} components, predictor rank {selected.predictor_rank})"
)
figure.savefig(ANALYSIS_DIR / "pipls_factors.pdf")
plt.close(figure)

# Plot selection-conditioned diagnostics in deterministic source-order response pages.
with PdfPages(ANALYSIS_DIR / "prediction_diagnostics.pdf") as report:
    for page_number, responses in enumerate(response_pages, start=1):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(13.0, 4.2),
            layout="constrained",
        )
        response_array = np.array(responses, dtype=np.int64)
        for response in responses:
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
                diagnostics.observed_standardized[:, response_array].ravel(),
                diagnostics.predicted_standardized[:, response_array].ravel(),
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
        axes[1].axhline(0.0, linewidth=1.0, linestyle="--", color="0.35")
        axes[1].set_xlabel("Predicted response (standardized)")
        axes[1].set_ylabel("Standardized residual")
        positions = np.arange(len(responses))
        axes[2].bar(positions, diagnostics.standardized_rmse[response_array])
        axes[2].set_xticks(positions)
        axes[2].set_xticklabels(
            [response_names[index] for index in responses],
            rotation=45,
            ha="right",
        )
        axes[2].set_xlabel("Response")
        axes[2].set_ylabel("Standardized RMSE")
        if len(responses) > 1:
            axes[0].legend()
            axes[1].legend()
        figure.suptitle(
            rf"Tobacco $\Pi$-PLS prediction diagnostics — "
            f"response page {page_number}/{len(response_pages)} — "
            f"{diagnostics.prediction_kind}"
        )
        report.savefig(figure)
        plt.close(figure)

# Plot scores, loadings, and raw observation diagnostics.
figure, axes = plt.subplots(
    2,
    2,
    figsize=(12.0, 9.0),
    layout="constrained",
)
if selected.n_components >= 2:
    axes[0, 0].scatter(
        structure.x_scores[:, 0],
        structure.x_scores[:, 1],
        alpha=0.75,
    )
    axes[0, 0].axvline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axes[0, 0].set_xlabel("X score component 1")
    axes[0, 0].set_ylabel("X score component 2")
else:
    observation_positions = np.arange(structure.x_scores.shape[0])
    axes[0, 0].scatter(
        observation_positions,
        structure.x_scores[:, 0],
        alpha=0.75,
    )
    axes[0, 0].set_xlabel("Observation")
    axes[0, 0].set_ylabel("X score component 1")
axes[0, 0].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")

for component in display_components:
    axes[0, 1].plot(
        wavenumbers,
        structure.x_loadings[:, component],
        label=f"Component {component + 1}",
    )
axes[0, 1].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axes[0, 1].set_xlabel("Wavenumber (cm$^{-1}$)")
axes[0, 1].set_ylabel("X loading")
axes[0, 1].legend()

response_positions = np.arange(len(response_names))
component_width = 0.8 / len(display_components)
for series, component in enumerate(display_components):
    offset = (series - (len(display_components) - 1) / 2) * component_width
    axes[1, 0].bar(
        response_positions + offset,
        structure.y_loadings[:, component],
        width=component_width,
        label=f"Component {component + 1}",
    )
axes[1, 0].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axes[1, 0].set_xticks(response_positions)
axes[1, 0].set_xticklabels(response_names, rotation=45, ha="right")
axes[1, 0].set_xlabel("Response")
axes[1, 0].set_ylabel("Y loading")
axes[1, 0].legend()

axes[1, 1].scatter(
    observations.score_distance,
    observations.x_reconstruction_residual,
    alpha=0.75,
)
axes[1, 1].set_xlabel("Score distance")
axes[1, 1].set_ylabel("Squared X-reconstruction residual")
figure.suptitle(
    r"Tobacco $\Pi$-PLS latent structure and observation diagnostics"
)
figure.savefig(ANALYSIS_DIR / "latent_structure.pdf")
plt.close(figure)

# Plot response-specific coefficients in the same deterministic pages.
with PdfPages(ANALYSIS_DIR / "coefficients.pdf") as report:
    for page_number, responses in enumerate(response_pages, start=1):
        figure, axis = plt.subplots(
            figsize=(10.0, 5.0),
            layout="constrained",
        )
        for response in responses:
            axis.plot(
                wavenumbers,
                structure.coefficients[response],
                label=response_names[response],
            )
        axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
        axis.set_xlabel("Wavenumber (cm$^{-1}$)")
        axis.set_ylabel("Regression coefficient")
        axis.set_title(
            rf"Tobacco $\Pi$-PLS — "
            f"response page {page_number}/{len(response_pages)}"
        )
        axis.legend(title="Response")
        report.savefig(figure)
        plt.close(figure)

print(f"X shape: {X.shape}; Y shape: {Y.shape}")
print(
    "1-SE-recommended Pi-PLS: "
    f"n_components={model.n_components}, predictor_rank={model.predictor_rank_}"
)
print(f"Wrote PDF figures to {ANALYSIS_DIR}")
