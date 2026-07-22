"""Generate deterministic single-chart figures for the Pulp tutorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
path_entry = str(SRC_DIR)
if path_entry not in sys.path:
    sys.path.insert(0, path_entry)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "pipls-pulp-tutorial"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402

from pipls import PiPLSComponentPath, PiPLSPathCV, PiPLSRegression  # noqa: E402
from pipls.inspection import (  # noqa: E402
    biplot_coordinates,
    latent_structure,
    pipls_display_factors,
    prediction_diagnostics,
)
from pipls.plotting import (  # noqa: E402
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

PULP_DATA_DIR = REPOSITORY_ROOT / "datasets" / "pulp"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "docs" / "assets" / "generated" / "pulp"
CHOSEN_N_COMPONENTS = 3
DISPLAY_COMPONENTS = (0, 1, 2)
DETAILED_RESPONSES = ("CSF", "Density", "TI")
PREDICTION_KIND = "selection-conditioned OOF predictions"
FIGURE_FILENAMES = (
    "component_path.svg",
    "predictor_rank_profile.svg",
    "scores.svg",
    "biplot.svg",
    "x_loadings.svg",
    "y_loadings.svg",
    "predictor_directions.svg",
    "dilation.svg",
    "response_directions.svg",
    "weighted_response_directions.svg",
    "coefficients.svg",
    "observed_vs_predicted.svg",
    "residuals_vs_predicted.svg",
    "standardized_rmse.svg",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _figure(
    *,
    figsize: tuple[float, float],
) -> tuple[Figure, Axes]:
    return plt.subplots(figsize=figsize, layout="constrained")


def _save_svg(figure: Figure, path: Path) -> None:
    figure.savefig(
        path,
        format="svg",
        metadata={"Creator": "Pi-PLS repository", "Date": None},
    )
    plt.close(figure)


def _rotate_category_labels(axis: Axes) -> None:
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")


def _response_indices(response_names: Sequence[str]) -> tuple[int, ...]:
    index_by_name = {name: index for index, name in enumerate(response_names)}
    missing = [name for name in DETAILED_RESPONSES if name not in index_by_name]
    if missing:
        raise RuntimeError(f"Pulp response table is missing tutorial responses: {missing!r}.")
    return tuple(index_by_name[name] for name in DETAILED_RESPONSES)


def _render_component_path(
    component_path: PiPLSComponentPath,
    *,
    chosen_n_components: int,
    chosen_predictor_rank: int,
    output_path: Path,
) -> None:
    selected = component_path.for_n_components(chosen_n_components)
    if selected.predictor_rank != chosen_predictor_rank:
        raise RuntimeError("The tutorial component path and selected predictor rank disagree.")

    figure, axis = _figure(figsize=(7.4, 4.8))
    axis.errorbar(
        component_path.n_components,
        component_path.cv_mse_mean,
        yerr=component_path.cv_mse_fold_sd,
        fmt="o-",
        capsize=4,
    )
    axis.scatter(
        [chosen_n_components],
        [selected.cv_mse_mean],
        marker="D",
        s=70,
        label=(f"Chosen: {chosen_n_components} components, predictor rank {chosen_predictor_rank}"),
        zorder=3,
    )
    for n_components, mean_mse, predictor_rank in zip(
        component_path.n_components,
        component_path.cv_mse_mean,
        component_path.predictor_rank,
        strict=True,
    ):
        axis.annotate(
            rf"$r_\pi={int(predictor_rank)}$",
            (n_components, mean_mse),
            xytext=(0, 8),
            textcoords="offset points",
            horizontalalignment="center",
        )
    axis.set_title("Pulp component path")
    axis.set_xlabel("Number of response components")
    axis.set_ylabel("Response-standardized CV-MSE")
    axis.set_xticks(component_path.n_components)
    axis.set_ylim(
        0.0,
        max(
            1.0,
            1.05 * float(np.max(component_path.cv_mse_mean + component_path.cv_mse_fold_sd)),
        ),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_svg(figure, output_path)


def _render_predictor_rank_profile(
    predictor_ranks: np.ndarray,
    cv_mse_mean: np.ndarray,
    cv_mse_fold_sd: np.ndarray,
    *,
    chosen_n_components: int,
    chosen_predictor_rank: int,
    output_path: Path,
) -> None:
    selected_rows = np.flatnonzero(predictor_ranks == chosen_predictor_rank)
    if selected_rows.size != 1:
        raise RuntimeError("The tutorial rank profile must contain one selected predictor rank.")
    selected = int(selected_rows[0])

    figure, axis = _figure(figsize=(7.4, 4.8))
    axis.errorbar(
        predictor_ranks,
        cv_mse_mean,
        yerr=cv_mse_fold_sd,
        fmt="o-",
        capsize=4,
    )
    axis.scatter(
        [chosen_predictor_rank],
        [cv_mse_mean[selected]],
        marker="D",
        s=70,
        label=f"Selected predictor rank {chosen_predictor_rank}",
        zorder=3,
    )
    axis.set_title(f"Pulp predictor-rank profile at {chosen_n_components} components")
    axis.set_xlabel("Predictor rank")
    axis.set_ylabel("Response-standardized CV-MSE")
    axis.set_xticks(predictor_ranks)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_svg(figure, output_path)


def render_pulp_tutorial_assets(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Generate all Pulp tutorial figures and return the manifest path."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    X = pd.read_csv(PULP_DATA_DIR / "X.csv")
    Y = pd.read_csv(PULP_DATA_DIR / "Y.csv")
    predictor_names = tuple(str(name) for name in X.columns)
    response_names = tuple(str(name) for name in Y.columns)
    detailed_response_indices = _response_indices(response_names)

    path_search = PiPLSPathCV(refit=False).fit(X, Y)
    component_path = path_search.component_path_
    selected = component_path.for_n_components(CHOSEN_N_COMPONENTS)

    cv_results = path_search.cv_results_
    rank_rows = np.asarray(cv_results["n_components"]) == selected.n_components
    rank_order = np.argsort(np.asarray(cv_results["predictor_rank"])[rank_rows])
    predictor_ranks = np.asarray(cv_results["predictor_rank"])[rank_rows][rank_order]
    rank_cv_mse_mean = np.asarray(cv_results["mean_response_standardized_mse"])[rank_rows][
        rank_order
    ]
    rank_cv_mse_fold_sd = np.asarray(cv_results["std_response_standardized_mse"])[rank_rows][
        rank_order
    ]

    model = PiPLSRegression(
        n_components=selected.n_components,
        predictor_rank=selected.predictor_rank,
    ).fit(X, Y)
    oof_predictions = cross_val_predict(
        model,
        X,
        Y,
        cv=KFold(n_splits=5, shuffle=False),
    )
    factors = pipls_display_factors(model.decomposition_)
    structure = latent_structure(model)
    diagnostics = prediction_diagnostics(
        Y,
        oof_predictions,
        prediction_kind=PREDICTION_KIND,
    )

    _render_component_path(
        component_path,
        chosen_n_components=selected.n_components,
        chosen_predictor_rank=selected.predictor_rank,
        output_path=output_dir / "component_path.svg",
    )
    _render_predictor_rank_profile(
        predictor_ranks,
        rank_cv_mse_mean,
        rank_cv_mse_fold_sd,
        chosen_n_components=selected.n_components,
        chosen_predictor_rank=selected.predictor_rank,
        output_path=output_dir / "predictor_rank_profile.svg",
    )

    figure, axis = _figure(figsize=(6.4, 5.0))
    plot_scores(
        structure,
        components=(0, 1),
        title="Pulp X scores",
        ax=axis,
    )
    _save_svg(figure, output_dir / "scores.svg")

    figure, axis = _figure(figsize=(8.0, 6.2))
    plot_biplot(
        biplot_coordinates(structure, components=(0, 1)),
        predictor_names=predictor_names,
        title="Pulp score-loading biplot",
        ax=axis,
    )
    axis.legend()
    _save_svg(figure, output_dir / "biplot.svg")

    figure, axis = _figure(figsize=(10.0, 5.4))
    plot_x_loadings(
        structure,
        predictor_style="bar",
        predictor_names=predictor_names,
        components=DISPLAY_COMPONENTS,
        title="Pulp X loadings",
        ax=axis,
    )
    axis.legend(title="Component")
    _rotate_category_labels(axis)
    _save_svg(figure, output_dir / "x_loadings.svg")

    figure, axis = _figure(figsize=(8.8, 5.2))
    plot_y_loadings(
        structure,
        response_names=response_names,
        components=DISPLAY_COMPONENTS,
        title="Pulp Y loadings",
        ax=axis,
    )
    axis.legend(title="Component")
    _rotate_category_labels(axis)
    _save_svg(figure, output_dir / "y_loadings.svg")

    figure, axis = _figure(figsize=(10.0, 5.4))
    plot_pipls_predictor_directions(
        factors,
        predictor_style="bar",
        predictor_names=predictor_names,
        components=DISPLAY_COMPONENTS,
        title=r"Pulp predictor directions $P$",
        ax=axis,
    )
    axis.legend(title="Component")
    _rotate_category_labels(axis)
    _save_svg(figure, output_dir / "predictor_directions.svg")

    figure, axis = _figure(figsize=(6.4, 4.6))
    plot_pipls_dilation(
        factors,
        components=DISPLAY_COMPONENTS,
        title=r"Pulp dilation $D$",
        ax=axis,
    )
    _save_svg(figure, output_dir / "dilation.svg")

    figure, axis = _figure(figsize=(8.8, 5.2))
    plot_pipls_response_directions(
        factors,
        response_names=response_names,
        components=DISPLAY_COMPONENTS,
        title=r"Pulp response directions $Q$",
        ax=axis,
    )
    axis.legend(title="Component")
    _rotate_category_labels(axis)
    _save_svg(figure, output_dir / "response_directions.svg")

    figure, axis = _figure(figsize=(8.8, 5.2))
    plot_pipls_weighted_response_directions(
        factors,
        response_names=response_names,
        components=DISPLAY_COMPONENTS,
        title=r"Pulp weighted response directions $QD$",
        ax=axis,
    )
    axis.legend(title="Component")
    _rotate_category_labels(axis)
    _save_svg(figure, output_dir / "weighted_response_directions.svg")

    figure, axis = _figure(figsize=(10.0, 5.4))
    plot_coefficients(
        structure,
        predictor_style="bar",
        predictor_names=predictor_names,
        response_names=response_names,
        responses=detailed_response_indices,
        title="Pulp regression coefficients",
        ax=axis,
    )
    axis.legend(title="Response")
    _rotate_category_labels(axis)
    _save_svg(figure, output_dir / "coefficients.svg")

    figure, axis = _figure(figsize=(6.4, 5.0))
    plot_observed_vs_predicted(
        diagnostics,
        response_names=response_names,
        responses=detailed_response_indices,
        title="Pulp observed versus predicted",
        ax=axis,
    )
    axis.legend(title="Response")
    _save_svg(figure, output_dir / "observed_vs_predicted.svg")

    figure, axis = _figure(figsize=(6.4, 5.0))
    plot_residuals_vs_predicted(
        diagnostics,
        response_names=response_names,
        responses=detailed_response_indices,
        title="Pulp residuals versus predicted",
        ax=axis,
    )
    axis.legend(title="Response")
    _save_svg(figure, output_dir / "residuals_vs_predicted.svg")

    figure, axis = _figure(figsize=(7.4, 5.0))
    plot_standardized_rmse(
        diagnostics,
        response_names=response_names,
        title="Pulp standardized RMSE",
        ax=axis,
    )
    _rotate_category_labels(axis)
    _save_svg(figure, output_dir / "standardized_rmse.svg")

    figures = [
        {
            "filename": filename,
            "sha256": _sha256(output_dir / filename),
        }
        for filename in FIGURE_FILENAMES
    ]
    manifest = {
        "schema_version": 1,
        "dataset": {
            "name": "Pulp",
            "files": {
                "X.csv": _sha256(PULP_DATA_DIR / "X.csv"),
                "Y.csv": _sha256(PULP_DATA_DIR / "Y.csv"),
            },
        },
        "analysis": {
            "chosen_n_components": selected.n_components,
            "chosen_predictor_rank": selected.predictor_rank,
            "evaluated_predictor_ranks": predictor_ranks.tolist(),
            "predictor_rank_at_upper_boundary": bool(
                selected.predictor_rank == int(predictor_ranks[-1])
            ),
            "displayed_components": [component + 1 for component in DISPLAY_COMPONENTS],
            "detailed_responses": list(DETAILED_RESPONSES),
            "prediction_kind": diagnostics.prediction_kind,
        },
        "figures": figures,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory receiving the generated SVG files and manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = render_pulp_tutorial_assets(args.output_dir)
    print(f"Wrote Pulp tutorial figures to {manifest.parent}")


if __name__ == "__main__":
    main()
