"""Fit and inspect a Sugarcane Pi-PLS model directly in memory."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold

from pipls import PiPLSComponentPath, PiPLSPredictorRankProfile, PiPLSSearchCV
from pipls.datasets import load_sugarcane
from pipls.inspection import (
    LatentStructure,
    PiPLSDisplayFactors,
    PredictionDiagnostics,
    latent_structure,
    pipls_display_factors,
    prediction_diagnostics,
)

ANALYSIS_DIR = Path(__file__).resolve().parent / "results" / "sugarcane_post_analysis"
CHOSEN_N_COMPONENTS = 2
CV = KFold(n_splits=5, shuffle=True, random_state=0)


def _plot_component_path(path: PiPLSComponentPath, output_path: Path) -> None:
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
    axis.set_xlabel("Number of components")
    axis.set_ylabel("Mean response-standardized CV-MSE (±1 SE)")
    axis.set_title(r"Sugarcane $\Pi$-PLS component path")
    axis.set_xticks(path.n_components)
    upper = float(np.max(path.cv_mse_mean + path.cv_mse_standard_error))
    axis.set_ylim(0.0, max(1.0, 1.05 * upper))
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_path)
    plt.close(figure)


def _plot_predictor_rank_profile(
    profile: PiPLSPredictorRankProfile,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(7.0, 4.5),
        layout="constrained",
    )
    axis.errorbar(
        profile.predictor_rank,
        profile.cv_mse_mean,
        yerr=profile.cv_mse_standard_error,
        fmt="o-",
        capsize=4,
    )
    axis.scatter(
        [profile.selection.predictor_rank],
        [profile.selection.cv_mse_mean],
        marker="D",
        s=70,
        label=f"CV-MSE minimum: rank {profile.selection.predictor_rank}",
        zorder=3,
    )
    axis.set_xlabel("Predictor rank")
    axis.set_ylabel("Mean response-standardized CV-MSE (±1 SE)")
    axis.set_title(
        rf"Sugarcane $\Pi$-PLS predictor-rank profile at "
        f"{profile.n_components} components"
    )
    axis.set_xticks(profile.predictor_rank)
    upper = float(np.max(profile.cv_mse_mean + profile.cv_mse_standard_error))
    axis.set_ylim(0.0, max(1.0, 1.05 * upper))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output_path)
    plt.close(figure)


def _plot_pipls_factors(
    factors: PiPLSDisplayFactors,
    wavelengths: NDArray[np.float64],
    response_names: list[str],
    *,
    selected_n_components: int,
    selected_predictor_rank: int,
    output_path: Path,
) -> None:
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
        axis.set_xticklabels(response_names)
        axis.set_xlabel("Response")
        axis.set_ylabel(ylabel)
        axis.legend()
    figure.suptitle(
        r"Sugarcane $\Pi$-PLS factors "
        f"({selected_n_components} components, predictor rank {selected_predictor_rank})"
    )
    figure.savefig(output_path)
    plt.close(figure)


def _plot_prediction_diagnostics(
    diagnostics: PredictionDiagnostics,
    response_names: list[str],
    output_path: Path,
) -> None:
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

    positions = np.arange(len(response_names))
    axes[2].bar(positions, diagnostics.standardized_rmse)
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(response_names)
    axes[2].set_xlabel("Response")
    axes[2].set_ylabel("Standardized RMSE")
    axes[0].legend()
    axes[1].legend()
    figure.suptitle(
        rf"Sugarcane $\Pi$-PLS prediction diagnostics — {diagnostics.prediction_kind}"
    )
    figure.savefig(output_path)
    plt.close(figure)


def _plot_latent_structure(
    structure: LatentStructure,
    wavelengths: NDArray[np.float64],
    response_names: list[str],
    output_path: Path,
) -> None:
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

    for component in (0, 1):
        axes[1].plot(
            wavelengths,
            structure.x_loadings[:, component],
            label=f"Component {component + 1}",
        )
    axes[1].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("X loading")
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
    axes[2].legend()
    figure.suptitle(r"Sugarcane $\Pi$-PLS latent structure")
    figure.savefig(output_path)
    plt.close(figure)


def _plot_coefficients(
    structure: LatentStructure,
    wavelengths: NDArray[np.float64],
    response_names: list[str],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(10.0, 5.0),
        layout="constrained",
    )
    for response, name in enumerate(response_names):
        axis.plot(wavelengths, structure.coefficients[response], label=name)
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Regression coefficient")
    axis.legend(title="Response")
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    data = load_sugarcane()
    X = data.data
    Y = data.target
    wavelengths = np.asarray(data.feature_names, dtype=np.float64)
    response_names = list(data.target_names)

    # Complete search and full-data modeling before numerical analysis.
    search = PiPLSSearchCV(cv=CV).fit(X, Y)
    model = search.refit(
        X,
        Y,
        n_components=CHOSEN_N_COMPONENTS,
    )

    # Retrieve selection, path, rank-profile, and OOF evidence after modeling.
    selection = model.selection_
    path = search.component_path_
    rank_profile = search.predictor_rank_profile(selection.n_components)
    report = search.oof_report(
        X,
        Y,
        selection=selection,
    )
    oof_predictions = report.oof_predictions

    # Calculate fitted-model and prediction inspection results.
    factors = pipls_display_factors(model.decomposition_)
    structure = latent_structure(model)
    diagnostics = prediction_diagnostics(
        Y,
        oof_predictions,
        prediction_kind="selection-conditioned OOF predictions",
    )

    # Render the final reports from completed public result objects.
    _plot_component_path(path, ANALYSIS_DIR / "component_path.pdf")
    _plot_predictor_rank_profile(
        rank_profile,
        ANALYSIS_DIR / "predictor_rank_profile.pdf",
    )
    _plot_pipls_factors(
        factors,
        wavelengths,
        response_names,
        selected_n_components=selection.n_components,
        selected_predictor_rank=selection.predictor_rank,
        output_path=ANALYSIS_DIR / "pipls_factors.pdf",
    )
    _plot_prediction_diagnostics(
        diagnostics,
        response_names,
        ANALYSIS_DIR / "prediction_diagnostics.pdf",
    )
    _plot_latent_structure(
        structure,
        wavelengths,
        response_names,
        ANALYSIS_DIR / "latent_structure.pdf",
    )
    _plot_coefficients(
        structure,
        wavelengths,
        response_names,
        ANALYSIS_DIR / "coefficients.pdf",
    )

    print(f"X shape: {X.shape}; Y shape: {Y.shape}")
    print(
        "Selected Pi-PLS: "
        f"n_components={model.n_components}, predictor_rank={model.predictor_rank_}"
    )
    print(
        "Predictor-rank profile: "
        f"evaluated {rank_profile.predictor_rank[0]} to "
        f"{rank_profile.predictor_rank[-1]}; "
        f"selected rank {rank_profile.selection.predictor_rank}"
    )
    print(f"Wrote PDF figures to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
