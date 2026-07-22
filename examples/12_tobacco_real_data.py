"""Fit and inspect a Tobacco Pi-PLS model directly in memory."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import KFold, cross_val_predict

from pipls import PiPLSPathCV, PiPLSRegression
from pipls.inspection import (
    latent_structure,
    observation_diagnostics,
    pipls_display_factors,
    prediction_diagnostics,
)
from pipls.plotting import (
    plot_coefficients,
    plot_observation_diagnostics,
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

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "tobacco"
ANALYSIS_DIR = Path(__file__).resolve().parent / "results" / "tobacco_post_analysis"
CHOSEN_N_COMPONENTS = 8
DISPLAY_COMPONENTS = (0, 1, 2, 3)
RESPONSES_PER_PAGE = 5

X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")
wavenumbers = X.columns.to_numpy(dtype=float)
response_names = Y.columns.tolist()
response_pages = tuple(
    tuple(range(start, min(start + RESPONSES_PER_PAGE, len(response_names))))
    for start in range(0, len(response_names), RESPONSES_PER_PAGE)
)

# Evaluate and plot the Pi-PLS component path with a full predictor SVD.
path_search = PiPLSPathCV(
    estimator=PiPLSRegression(svd_solver="full"),
    search_method="auto",
    refit=False,
    n_jobs=1,
).fit(X, Y)
path = path_search.component_path_
selected = path.for_n_components(CHOSEN_N_COMPONENTS)

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
axis.scatter(
    [selected.n_components],
    [selected.cv_mse_mean],
    marker="D",
    s=70,
    label=(
        f"Chosen: {selected.n_components} components, "
        f"predictor rank {selected.predictor_rank}"
    ),
    zorder=3,
)
axis.set_xlabel("Number of components")
axis.set_ylabel("Response-standardized CV-MSE")
axis.set_title("Tobacco Pi-PLS component path")
axis.set_xticks(path.n_components)
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

# Plot the Pi-PLS factors.
figure, axes = plt.subplots(
    2,
    2,
    figsize=(12.0, 9.0),
    layout="constrained",
)
plot_pipls_predictor_directions(
    factors,
    predictor_style="line",
    predictor_axis=wavenumbers,
    predictor_axis_label="Wavenumber (cm$^{-1}$)",
    ax=axes[0, 0],
)
plot_pipls_dilation(factors, ax=axes[0, 1])
plot_pipls_response_directions(
    factors,
    response_names=response_names,
    ax=axes[1, 0],
)
plot_pipls_weighted_response_directions(
    factors,
    response_names=response_names,
    ax=axes[1, 1],
)
axes[0, 0].legend(title="Component")
axes[1, 0].legend(title="Component")
axes[1, 1].legend(title="Component")
figure.suptitle(
    "Tobacco Pi-PLS factors "
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
        plot_observed_vs_predicted(
            diagnostics,
            response_names=response_names,
            responses=responses,
            include_prediction_kind=False,
            ax=axes[0],
        )
        plot_residuals_vs_predicted(
            diagnostics,
            response_names=response_names,
            responses=responses,
            include_prediction_kind=False,
            ax=axes[1],
        )
        plot_standardized_rmse(
            diagnostics,
            response_names=response_names,
            responses=responses,
            include_prediction_kind=False,
            ax=axes[2],
        )
        if len(responses) > 1:
            axes[0].legend()
            axes[1].legend()
        figure.suptitle(
            "Tobacco Pi-PLS prediction diagnostics "
            f"— response page {page_number}/{len(response_pages)}\n"
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
plot_scores(
    structure,
    components=(0, 1),
    title="X scores",
    ax=axes[0, 0],
)
plot_x_loadings(
    structure,
    predictor_style="line",
    predictor_axis=wavenumbers,
    predictor_axis_label="Wavenumber (cm$^{-1}$)",
    components=DISPLAY_COMPONENTS,
    title="X loadings",
    ax=axes[0, 1],
)
plot_y_loadings(
    structure,
    response_names=response_names,
    components=DISPLAY_COMPONENTS,
    title="Y loadings",
    ax=axes[1, 0],
)
plot_observation_diagnostics(
    observations,
    title="Observation diagnostics",
    ax=axes[1, 1],
)
axes[0, 1].legend(title="Component")
axes[1, 0].legend(title="Component")
figure.suptitle("Tobacco Pi-PLS latent structure and observation diagnostics")
figure.savefig(ANALYSIS_DIR / "latent_structure.pdf")
plt.close(figure)

# Plot response-specific coefficients in the same deterministic pages.
with PdfPages(ANALYSIS_DIR / "coefficients.pdf") as report:
    for page_number, responses in enumerate(response_pages, start=1):
        figure, axis = plt.subplots(
            figsize=(10.0, 5.0),
            layout="constrained",
        )
        plot_coefficients(
            structure,
            predictor_style="line",
            response_names=response_names,
            predictor_axis=wavenumbers,
            predictor_axis_label="Wavenumber (cm$^{-1}$)",
            responses=responses,
            title=(
                "Tobacco Pi-PLS coefficients "
                f"— response page {page_number}/{len(response_pages)}"
            ),
            ax=axis,
        )
        axis.legend(title="Response")
        report.savefig(figure)
        plt.close(figure)

print(f"X shape: {X.shape}; Y shape: {Y.shape}")
print(
    "Selected Pi-PLS: "
    f"n_components={model.n_components}, predictor_rank={model.predictor_rank_}"
)
print(f"Wrote PDF figures to {ANALYSIS_DIR}")
