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
EXAMPLES_DIR = REPOSITORY_ROOT / "examples"
for path in (SRC_DIR, EXAMPLES_DIR):
    path_entry = str(path)
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "pipls-pulp-tutorial"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _support.pulp_workflow import PULP_DATA_DIR, run_pulp_workflow  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pipls import PiPLSComponentPath  # noqa: E402
from pipls.inspection import biplot_coordinates  # noqa: E402
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

DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "docs" / "assets" / "generated" / "pulp"
DISPLAY_COMPONENTS = (0, 1, 2)
DETAILED_RESPONSES = ("CSF", "Density", "TI")
FIGURE_FILENAMES = (
    "component_path.svg",
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
    x = component_path.n_components
    mean = component_path.cv_mse_mean
    fold_sd = component_path.cv_mse_fold_sd
    ranks = component_path.predictor_rank
    selected_rows = np.flatnonzero(x == chosen_n_components)
    if selected_rows.size != 1:
        raise RuntimeError("The tutorial component path must contain one selected component row.")
    selected = int(selected_rows[0])
    if int(ranks[selected]) != chosen_predictor_rank:
        raise RuntimeError("The tutorial component path and selected predictor rank disagree.")

    figure, axis = _figure(figsize=(7.4, 4.8))
    axis.errorbar(x, mean, yerr=fold_sd, fmt="o-", capsize=4)
    axis.scatter(
        [chosen_n_components],
        [mean[selected]],
        marker="D",
        s=70,
        label=(f"Chosen: {chosen_n_components} components, predictor rank {chosen_predictor_rank}"),
        zorder=3,
    )
    for n_components, mean_mse, predictor_rank in zip(x, mean, ranks, strict=True):
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
    axis.set_xticks(x)
    axis.set_ylim(0.0, max(1.0, 1.05 * float(np.max(mean + fold_sd))))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_svg(figure, output_path)


def render_pulp_tutorial_assets(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Generate all Pulp tutorial figures and return the manifest path."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    workflow = run_pulp_workflow()
    predictor_names = tuple(str(name) for name in workflow.X.columns)
    response_names = tuple(str(name) for name in workflow.Y.columns)
    detailed_response_indices = _response_indices(response_names)

    _render_component_path(
        workflow.component_path,
        chosen_n_components=workflow.chosen_n_components,
        chosen_predictor_rank=workflow.chosen_predictor_rank,
        output_path=output_dir / "component_path.svg",
    )

    figure, axis = _figure(figsize=(6.4, 5.0))
    plot_scores(
        workflow.structure,
        components=(0, 1),
        title="Pulp X scores",
        ax=axis,
    )
    _save_svg(figure, output_dir / "scores.svg")

    figure, axis = _figure(figsize=(8.0, 6.2))
    plot_biplot(
        biplot_coordinates(workflow.structure, components=(0, 1)),
        predictor_names=predictor_names,
        title="Pulp score-loading biplot",
        ax=axis,
    )
    axis.legend()
    _save_svg(figure, output_dir / "biplot.svg")

    figure, axis = _figure(figsize=(10.0, 5.4))
    plot_x_loadings(
        workflow.structure,
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
        workflow.structure,
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
        workflow.factors,
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
        workflow.factors,
        components=DISPLAY_COMPONENTS,
        title=r"Pulp dilation $D$",
        ax=axis,
    )
    _save_svg(figure, output_dir / "dilation.svg")

    figure, axis = _figure(figsize=(8.8, 5.2))
    plot_pipls_response_directions(
        workflow.factors,
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
        workflow.factors,
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
        workflow.structure,
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
        workflow.diagnostics,
        response_names=response_names,
        responses=detailed_response_indices,
        title="Pulp observed versus predicted",
        ax=axis,
    )
    axis.legend(title="Response")
    _save_svg(figure, output_dir / "observed_vs_predicted.svg")

    figure, axis = _figure(figsize=(6.4, 5.0))
    plot_residuals_vs_predicted(
        workflow.diagnostics,
        response_names=response_names,
        responses=detailed_response_indices,
        title="Pulp residuals versus predicted",
        ax=axis,
    )
    axis.legend(title="Response")
    _save_svg(figure, output_dir / "residuals_vs_predicted.svg")

    figure, axis = _figure(figsize=(7.4, 5.0))
    plot_standardized_rmse(
        workflow.diagnostics,
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
            "chosen_n_components": workflow.chosen_n_components,
            "chosen_predictor_rank": workflow.chosen_predictor_rank,
            "displayed_components": [component + 1 for component in DISPLAY_COMPONENTS],
            "detailed_responses": list(DETAILED_RESPONSES),
            "prediction_kind": workflow.diagnostics.prediction_kind,
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
