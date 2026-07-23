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
from adjustText import adjust_text  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402

from pipls import (  # noqa: E402
    PiPLSComponentPath,
    PiPLSComponentResult,
    PiPLSPathCV,
    PiPLSPredictorRankProfile,
    PiPLSRegression,
)
from pipls.inspection import (  # noqa: E402
    biplot_coordinates,
    latent_structure,
    pipls_display_factors,
    prediction_diagnostics,
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
    "biplot.svg",
    "predictor_directions.svg",
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
    selected: PiPLSComponentResult,
    output_path: Path,
) -> None:
    figure, axis = _figure(figsize=(7.4, 4.8))
    axis.errorbar(
        component_path.n_components,
        component_path.cv_mse_mean,
        yerr=component_path.cv_mse_fold_sd,
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
    axis.set_title("Pulp component path")
    axis.set_xlabel("Number of components")
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
    profile: PiPLSPredictorRankProfile,
    *,
    output_path: Path,
) -> None:
    figure, axis = _figure(figsize=(7.4, 4.8))
    axis.errorbar(
        profile.predictor_rank,
        profile.cv_mse_mean,
        yerr=profile.cv_mse_fold_sd,
        fmt="o-",
        capsize=4,
    )
    axis.scatter(
        [profile.selected.predictor_rank],
        [profile.selected.cv_mse_mean],
        marker="D",
        s=70,
        label=f"CV-MSE minimum: rank {profile.selected.predictor_rank}",
        zorder=3,
    )
    axis.set_title(
        f"Pulp predictor-rank profile at {profile.n_components} components"
    )
    axis.set_xlabel("Predictor rank")
    axis.set_ylabel("Response-standardized CV-MSE")
    axis.set_xticks(profile.predictor_rank)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_svg(figure, output_path)


def render_pulp_tutorial_assets(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Generate the representative Pulp tutorial figures and return the manifest path."""

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

    _render_component_path(
        component_path,
        selected=selected,
        output_path=output_dir / "component_path.svg",
    )
    rank_profile = path_search.predictor_rank_profile(selected.n_components)
    _render_predictor_rank_profile(
        rank_profile,
        output_path=output_dir / "predictor_rank_profile.svg",
    )

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

    figure, axis = _figure(figsize=(8.0, 6.2))
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
    axis.set_title("Pulp score-loading biplot")
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
    _save_svg(figure, output_dir / "biplot.svg")

    figure, axis = _figure(figsize=(10.0, 5.4))
    predictor_positions = np.arange(len(predictor_names))
    predictor_width = 0.8 / len(DISPLAY_COMPONENTS)
    for series, component in enumerate(DISPLAY_COMPONENTS):
        offset = (series - (len(DISPLAY_COMPONENTS) - 1) / 2.0) * predictor_width
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
    axis.set_title(r"Pulp predictor directions $P$")
    axis.legend()
    _rotate_category_labels(axis)
    _save_svg(figure, output_dir / "predictor_directions.svg")

    figure, axis = _figure(figsize=(6.4, 5.0))
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
    axis.set_title(f"Pulp observed versus predicted\n{diagnostics.prediction_kind}")
    axis.legend(title="Response")
    _save_svg(figure, output_dir / "observed_vs_predicted.svg")

    figure, axis = _figure(figsize=(6.4, 5.0))
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
    axis.set_ylabel(r"Residual $y-\hat y$ (standardized)")
    axis.set_title(f"Pulp residual versus predicted\n{diagnostics.prediction_kind}")
    axis.legend(title="Response")
    _save_svg(figure, output_dir / "residuals_vs_predicted.svg")

    figure, axis = _figure(figsize=(7.4, 5.0))
    positions = np.arange(len(response_names))
    axis.bar(positions, diagnostics.standardized_rmse)
    axis.set_xticks(positions)
    axis.set_xticklabels(response_names)
    axis.set_xlabel("Response")
    axis.set_ylabel("Standardized RMSE")
    axis.set_title(f"Pulp standardized RMSE\n{diagnostics.prediction_kind}")
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
            "evaluated_predictor_ranks": rank_profile.predictor_rank.tolist(),
            "predictor_rank_at_upper_boundary": bool(
                selected.predictor_rank == int(rank_profile.predictor_rank[-1])
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
