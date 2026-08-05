"""Generate deterministic single-chart figures for the Pulp tutorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
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
from adjustText import adjust_text  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from sklearn.model_selection import RepeatedKFold  # noqa: E402

from pipls import PiPLSSearchCV  # noqa: E402
from pipls.component_path import (  # noqa: E402
    PiPLSComponentPath,
    PiPLSPredictorRankProfile,
    PiPLSSelection,
)
from pipls.datasets import load_pulp  # noqa: E402
from pipls.inspection import (  # noqa: E402
    biplot_coordinates,
    latent_structure,
    pipls_display_factors,
    prediction_diagnostics,
)

DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "docs" / "assets" / "generated" / "pulp"
CHOSEN_N_COMPONENTS = 3
DETAILED_RESPONSE_COUNT = 3
CV = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
PREDICTION_KIND = "selection-conditioned OOF predictions"
FIGURE_FILENAMES = (
    "component_path.svg",
    "predictor_rank_profile.svg",
    "biplot.svg",
    "predictor_directions.svg",
    "weighted_response_directions.svg",
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


def _render_component_path(
    component_path: PiPLSComponentPath,
    *,
    selected: PiPLSSelection,
    output_path: Path,
) -> None:
    figure, axis = _figure(figsize=(7.4, 4.8))
    axis.errorbar(
        component_path.n_components,
        component_path.cv_mse_mean,
        yerr=component_path.cv_mse_std,
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
    axis.set_title(r"Pulp $\Pi$-PLS component path")
    axis.set_xlabel("Number of components")
    axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
    axis.set_xticks(component_path.n_components)
    upper = float(
        np.max(component_path.cv_mse_mean + component_path.cv_mse_std)
    )
    axis.set_ylim(0.0, max(1.0, 1.05 * upper))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_svg(figure, output_path)


def _render_predictor_rank_profile(
    profile: PiPLSPredictorRankProfile,
    *,
    output_path: Path,
) -> None:
    figure, axis = _figure(figsize=(7.4, 4.8))
    axis.errorbar(
        profile.predictor_rank,
        profile.cv_mse_mean,
        yerr=profile.cv_mse_std,
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
    axis.set_title(
        rf"Pulp $\Pi$-PLS predictor-rank profile at "
        f"{profile.n_components} components"
    )
    axis.set_xlabel("Predictor rank")
    axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
    axis.set_xticks(profile.predictor_rank)
    upper = float(np.max(profile.cv_mse_mean + profile.cv_mse_std))
    axis.set_ylim(0.0, max(1.0, 1.05 * upper))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_svg(figure, output_path)


def render_pulp_tutorial_assets(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Generate the representative Pulp tutorial figures and return the manifest path."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    data = load_pulp()
    X, Y = data.X, data.Y
    predictor_names = data.feature_names
    response_names = data.target_names

    search = PiPLSSearchCV(cv=CV).fit(X, Y)
    component_path = search.component_path_
    selection = search.select(n_components=CHOSEN_N_COMPONENTS)
    rank_profile = search.predictor_rank_profile(selection.n_components)
    report = search.oof_report(
        X,
        Y,
        selection=selection,
    )
    oof_predictions = report.oof_predictions
    if not np.all(report.oof_prediction_counts == 10):
        raise RuntimeError(
            "Repeated Pulp CV must produce ten OOF predictions per observation."
        )
    model = search.refit(X, Y, selection=selection)
    factors = pipls_display_factors(
        model.decomposition_,
        response_index=response_names.index("TI"),
        response_sign="positive",
    )
    structure = latent_structure(model)
    diagnostics = prediction_diagnostics(
        Y,
        oof_predictions,
        prediction_kind=PREDICTION_KIND,
    )
    display_components = tuple(range(selection.n_components))

    _render_component_path(
        component_path,
        selected=selection,
        output_path=output_dir / "component_path.svg",
    )
    _render_predictor_rank_profile(
        rank_profile,
        output_path=output_dir / "predictor_rank_profile.svg",
    )

    # --8<-- [start:render-pulp-biplot]
    figure, axis = plt.subplots(figsize=(9.0, 7.0), layout="constrained")
    biplot = biplot_coordinates(structure, components=(0, 1))
    sample_xy = biplot.sample_coordinates
    predictor_xy = biplot.predictor_coordinates
    axis.scatter(
        sample_xy[:, 0],
        sample_xy[:, 1],
        alpha=0.75,
        label="Samples",
    )
    for endpoint in predictor_xy:
        axis.add_patch(
            FancyArrowPatch(
                (0.0, 0.0),
                (float(endpoint[0]), float(endpoint[1])),
                arrowstyle="->",
                mutation_scale=10.0,
                linewidth=1.0,
            )
        )
    labels = [
        axis.text(
            float(endpoint[0]),
            float(endpoint[1]),
            name,
            fontsize="small",
        )
        for endpoint, name in zip(predictor_xy, predictor_names, strict=True)
    ]
    first, second = (int(value) for value in biplot.component_indices)
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.axvline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_xlabel(f"Balanced component {first + 1}")
    axis.set_ylabel(f"Balanced component {second + 1}")
    axis.set_title(r"Pulp $\Pi$-PLS score-loading biplot")
    axis.set_aspect("equal", adjustable="datalim")
    axis.margins(0.1)
    axis.legend()
    adjust_text(
        labels,
        x=sample_xy[:, 0],
        y=sample_xy[:, 1],
        target_x=predictor_xy[:, 0],
        target_y=predictor_xy[:, 1],
        ax=axis,
        ensure_inside_axes=True,
        prevent_crossings=False,
        iter_lim=200,
        arrowprops={"arrowstyle": "-", "linewidth": 0.6},
    )
    # --8<-- [end:render-pulp-biplot]
    _save_svg(figure, output_dir / "biplot.svg")

    # --8<-- [start:render-pulp-predictor-directions]
    figure, axis = plt.subplots(figsize=(10.0, 5.4), layout="constrained")
    predictor_positions = np.arange(len(predictor_names))
    predictor_width = 0.8 / CHOSEN_N_COMPONENTS
    for series, component in enumerate(display_components):
        offset = (series - (CHOSEN_N_COMPONENTS - 1) / 2.0) * predictor_width
        axis.bar(
            predictor_positions + offset,
            factors.predictor_directions[:, component],
            width=predictor_width,
            label=f"Component {component + 1}",
        )
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_xticks(predictor_positions)
    axis.set_xticklabels(predictor_names)
    axis.set_xlabel("Predictor")
    axis.set_ylabel(r"Predictor direction $P_{:k}$")
    axis.legend()
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    # --8<-- [end:render-pulp-predictor-directions]
    _save_svg(figure, output_dir / "predictor_directions.svg")

    # --8<-- [start:render-pulp-weighted-response-directions]
    figure, axis = plt.subplots(figsize=(8.2, 5.4), layout="constrained")
    response_positions = np.arange(len(response_names))
    response_width = 0.8 / CHOSEN_N_COMPONENTS
    for series, component in enumerate(display_components):
        offset = (series - (CHOSEN_N_COMPONENTS - 1) / 2.0) * response_width
        axis.bar(
            response_positions + offset,
            factors.weighted_response_directions[:, component],
            width=response_width,
            label=f"Component {component + 1}",
        )
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_xticks(response_positions)
    axis.set_xticklabels(response_names)
    axis.set_xlabel("Response")
    axis.set_ylabel(r"Weighted response direction $D_kQ_{:k}$")
    axis.legend()
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    # --8<-- [end:render-pulp-weighted-response-directions]
    _save_svg(figure, output_dir / "weighted_response_directions.svg")

    # --8<-- [start:render-pulp-observed-vs-predicted]
    figure, axis = plt.subplots(figsize=(6.4, 5.0), layout="constrained")
    detailed_response_indices = tuple(range(DETAILED_RESPONSE_COUNT))
    detailed_array = np.array(detailed_response_indices, dtype=np.int64)
    for response in detailed_response_indices:
        axis.scatter(
            diagnostics.observed_standardized[:, response],
            diagnostics.predicted_standardized[:, response],
            label=response_names[response],
            alpha=0.75,
        )
    values = np.concatenate(
        [
            diagnostics.observed_standardized[:, detailed_array].ravel(),
            diagnostics.predicted_standardized[:, detailed_array].ravel(),
        ]
    )
    lower = float(values.min())
    upper = float(values.max())
    margin = 0.05 * (upper - lower) if upper > lower else 1.0
    limits = (lower - margin, upper + margin)
    axis.plot(limits, limits, linewidth=1.0, linestyle="--", color="0.35")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Observed response (standardized)")
    axis.set_ylabel("Predicted response (standardized)")
    axis.set_title(
        rf"Pulp $\Pi$-PLS — {diagnostics.prediction_kind}"
    )
    axis.legend(title="Response")
    # --8<-- [end:render-pulp-observed-vs-predicted]
    _save_svg(figure, output_dir / "observed_vs_predicted.svg")

    # --8<-- [start:render-pulp-residuals-vs-predicted]
    figure, axis = plt.subplots(figsize=(6.4, 5.0), layout="constrained")
    detailed_array = np.array(detailed_response_indices, dtype=np.int64)
    for response in detailed_response_indices:
        axis.scatter(
            diagnostics.predicted_standardized[:, response],
            diagnostics.residual_standardized[:, response],
            label=response_names[response],
            alpha=0.75,
        )
    predicted = diagnostics.predicted_standardized[:, detailed_array]
    residual = diagnostics.residual_standardized[:, detailed_array]
    predicted_span = float(predicted.max() - predicted.min())
    residual_span = float(residual.max() - residual.min())
    axis.set_xlim(
        float(predicted.min())
        - (0.05 * predicted_span if predicted_span > 0.0 else 1.0),
        float(predicted.max())
        + (0.05 * predicted_span if predicted_span > 0.0 else 1.0),
    )
    axis.set_ylim(
        float(residual.min())
        - (0.05 * residual_span if residual_span > 0.0 else 1.0),
        float(residual.max())
        + (0.05 * residual_span if residual_span > 0.0 else 1.0),
    )
    axis.axhline(0.0, linewidth=1.0, linestyle="--", color="0.35")
    axis.set_xlabel("Predicted response (standardized)")
    axis.set_ylabel("Standardized residual")
    axis.set_title(
        rf"Pulp $\Pi$-PLS — {diagnostics.prediction_kind}"
    )
    axis.legend(title="Response")
    # --8<-- [end:render-pulp-residuals-vs-predicted]
    _save_svg(figure, output_dir / "residuals_vs_predicted.svg")

    # --8<-- [start:render-pulp-standardized-rmse]
    figure, axis = plt.subplots(figsize=(7.4, 5.0), layout="constrained")
    positions = np.arange(len(response_names))
    axis.bar(positions, diagnostics.standardized_rmse)
    axis.set_xticks(positions)
    axis.set_xticklabels(response_names)
    axis.set_xlabel("Response")
    axis.set_ylabel("Standardized RMSE")
    axis.set_title(
        rf"Pulp $\Pi$-PLS — {diagnostics.prediction_kind}"
    )
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    # --8<-- [end:render-pulp-standardized-rmse]
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
            "id": data.metadata["dataset"]["id"],
            "version": data.metadata["dataset"]["version"],
            "source_doi": data.metadata["source"]["doi"],
            "license": data.metadata["license"]["identifier"],
            "resource_sha256": dict(data.metadata["integrity"]["resource_sha256"]),
            "array_sha256": dict(data.metadata["integrity"]["array_sha256"]),
        },
        "analysis": {
            "chosen_n_components": selection.n_components,
            "chosen_predictor_rank": selection.predictor_rank,
            "evaluated_predictor_ranks": rank_profile.predictor_rank.tolist(),
            "predictor_rank_at_upper_boundary": bool(
                selection.predictor_rank == int(rank_profile.predictor_rank[-1])
            ),
            "displayed_components": [component + 1 for component in display_components],
            "factor_sign_anchor": {
                "response": "TI",
                "sign": "positive",
            },
            "detailed_responses": [
                response_names[index] for index in detailed_response_indices
            ],
            "prediction_kind": diagnostics.prediction_kind,
            "cross_validation": {
                "splitter": type(CV).__name__,
                "n_splits": 5,
                "n_repeats": 10,
                "random_state": 0,
                "materialized_splits": search.n_splits_,
            },
            "oof_predictions_per_observation": int(
                report.oof_prediction_counts[0]
            ),
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
