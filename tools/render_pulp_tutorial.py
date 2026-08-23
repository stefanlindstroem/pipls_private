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
from _support.annotation_layout import allocate_predictor_labels  # noqa: E402
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
from pipls.datasets import PiPLSDataset, load_pulp  # noqa: E402
from pipls.inspection import (  # noqa: E402
    LatentStructure,
    PiPLSDisplayFactors,
    PredictionDiagnostics,
    biplot_coordinates,
    latent_structure,
    pipls_display_factors,
    prediction_diagnostics,
)
from pipls.validation import PiPLSOOFReport  # noqa: E402

DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "docs" / "assets" / "generated" / "pulp"
CV = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
PREDICTION_KIND = "selection-conditioned OOF predictions"
FIGURE_FILENAMES = (
    "component_path.svg",
    "selected_component_path.svg",
    "predictor_rank_profile.svg",
    "biplot.svg",
    "predictor_directions.svg",
    "weighted_response_directions.svg",
    "observed_vs_predicted.svg",
    "residuals_vs_predicted.svg",
    "standardized_rmse.svg",
    "final_fit_observed_vs_predicted.svg",
    "final_fit_r2.svg",
    "final_fit_residual_distribution.svg",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _figure(*, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
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
    selected: PiPLSSelection | None,
    title: str,
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
    if selected is not None:
        axis.scatter(
            [selected.n_components],
            [selected.cv_mse_mean],
            marker="D",
            color="tab:orange",
            s=70,
            label=f"Chosen: {selected.n_components} components",
            zorder=3,
        )
        axis.legend()
    axis.set_title(title)
    axis.set_xlabel("Number of components")
    axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
    axis.set_xticks(component_path.n_components)
    upper = float(np.max(component_path.cv_mse_mean + component_path.cv_mse_std))
    axis.set_ylim(0.0, max(1.0, 1.05 * upper))
    axis.grid(axis="y", alpha=0.25)
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
        color="tab:orange",
        s=70,
        label=f"Selected rank: {profile.selection.predictor_rank}",
        zorder=3,
    )
    axis.set_title(
        "Pulp Π-PLS predictor-rank profile at "
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


def _render_biplot(
    structure: LatentStructure,
    *,
    predictor_names: tuple[str, ...],
    output_path: Path,
) -> None:
    # --8<-- [start:render-pulp-biplot]
    figure, axis = _figure(figsize=(9.0, 7.0))
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
    first, second = (int(value) for value in biplot.component_indices)
    axis.axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.axvline(0.0, linewidth=0.8, linestyle="--", color="0.45")
    axis.set_xlabel(f"Balanced component {first + 1}")
    axis.set_ylabel(f"Balanced component {second + 1}")
    axis.set_title(r"Pulp $\Pi$-PLS score-loading biplot")
    axis.set_aspect("equal", adjustable="datalim")
    axis.margins(0.1)
    axis.legend()
    allocate_predictor_labels(
        axis,
        predictor_xy,
        predictor_names,
        textsize=8,
    )
    # --8<-- [end:render-pulp-biplot]
    _save_svg(figure, output_path)


def _render_predictor_directions(
    factors: PiPLSDisplayFactors,
    *,
    predictor_names: tuple[str, ...],
    n_components: int,
    output_path: Path,
) -> None:
    # --8<-- [start:render-pulp-predictor-directions]
    figure, axis = _figure(figsize=(10.0, 5.4))
    predictor_positions = np.arange(len(predictor_names))
    predictor_width = 0.8 / n_components
    for series, component in enumerate(range(n_components)):
        offset = (series - (n_components - 1) / 2.0) * predictor_width
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
    _save_svg(figure, output_path)


def _render_weighted_response_directions(
    factors: PiPLSDisplayFactors,
    *,
    response_names: tuple[str, ...],
    n_components: int,
    output_path: Path,
) -> None:
    # --8<-- [start:render-pulp-weighted-response-directions]
    figure, axis = _figure(figsize=(8.2, 5.4))
    response_positions = np.arange(len(response_names))
    response_width = 0.8 / n_components
    for series, component in enumerate(range(n_components)):
        offset = (series - (n_components - 1) / 2.0) * response_width
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
    _save_svg(figure, output_path)


def _render_observed_vs_predicted(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: tuple[str, ...],
    output_path: Path,
) -> None:
    # --8<-- [start:render-pulp-observed-vs-predicted]
    figure, axis = _figure(figsize=(6.4, 5.0))
    for response, response_name in enumerate(response_names):
        axis.scatter(
            diagnostics.observed_standardized[:, response],
            diagnostics.predicted_standardized[:, response],
            label=response_name,
            alpha=0.75,
        )
    values = np.concatenate(
        [
            diagnostics.observed_standardized.ravel(),
            diagnostics.predicted_standardized.ravel(),
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
    axis.set_title(rf"Pulp $\Pi$-PLS — {diagnostics.prediction_kind}")
    axis.legend(title="Response", fontsize="small", ncols=2)
    # --8<-- [end:render-pulp-observed-vs-predicted]
    _save_svg(figure, output_path)


def _render_residuals_vs_predicted(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: tuple[str, ...],
    output_path: Path,
) -> None:
    # --8<-- [start:render-pulp-residuals-vs-predicted]
    figure, axis = _figure(figsize=(6.4, 5.0))
    for response, response_name in enumerate(response_names):
        axis.scatter(
            diagnostics.predicted_standardized[:, response],
            diagnostics.residual_standardized[:, response],
            label=response_name,
            alpha=0.75,
        )
    predicted = diagnostics.predicted_standardized
    residual = diagnostics.residual_standardized
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
    axis.set_title(rf"Pulp $\Pi$-PLS — {diagnostics.prediction_kind}")
    axis.legend(title="Response", fontsize="small", ncols=2)
    # --8<-- [end:render-pulp-residuals-vs-predicted]
    _save_svg(figure, output_path)


def _render_standardized_rmse(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: tuple[str, ...],
    output_path: Path,
) -> None:
    # --8<-- [start:render-pulp-standardized-rmse]
    figure, axis = _figure(figsize=(7.4, 5.0))
    positions = np.arange(len(response_names))
    axis.bar(positions, diagnostics.standardized_rmse)
    axis.set_xticks(positions)
    axis.set_xticklabels(response_names)
    axis.set_xlabel("Response")
    axis.set_ylabel("Standardized RMSE")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(rf"Pulp $\Pi$-PLS — {diagnostics.prediction_kind}")
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    # --8<-- [end:render-pulp-standardized-rmse]
    _save_svg(figure, output_path)


def _render_final_fit_observed_vs_predicted(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: tuple[str, ...],
    output_path: Path,
) -> None:
    figure, axis = _figure(figsize=(6.2, 6.2))
    for response, response_name in enumerate(response_names):
        axis.scatter(
            diagnostics.observed_standardized[:, response],
            diagnostics.predicted_standardized[:, response],
            label=response_name,
            alpha=0.7,
        )

    limits_source = np.concatenate(
        [
            diagnostics.observed_standardized.ravel(),
            diagnostics.predicted_standardized.ravel(),
        ]
    )
    lower = float(limits_source.min())
    upper = float(limits_source.max())
    margin = 0.05 * (upper - lower) if upper > lower else 1.0
    limits = (lower - margin, upper + margin)
    axis.plot(limits, limits, linewidth=1.0, linestyle="--", color="0.35")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Observed response (standardized)")
    axis.set_ylabel("Fitted response (standardized)")
    axis.set_title(r"Final $\Pi$-PLS fit: observed versus fitted")
    axis.legend(title="Response", fontsize="small", ncols=2)
    axis.grid(alpha=0.2)
    _save_svg(figure, output_path)


def _render_final_fit_r2(
    diagnostics: PredictionDiagnostics,
    *,
    response_names: tuple[str, ...],
    output_path: Path,
) -> None:
    positions = np.arange(len(response_names))
    figure, axis = _figure(figsize=(8.2, 4.8))
    axis.bar(positions, diagnostics.response_r2)
    axis.axhline(0.0, linewidth=0.8, color="0.35")
    axis.set_xticks(positions)
    axis.set_xticklabels(response_names, rotation=45, ha="right")
    axis.set_xlabel("Response")
    axis.set_ylabel(r"Fitted $R^2$")
    axis.set_title(r"Final $\Pi$-PLS fit: response-wise $R^2$")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.2)
    _save_svg(figure, output_path)


def _render_final_fit_residual_distribution(
    diagnostics: PredictionDiagnostics,
    *,
    output_path: Path,
) -> None:
    residuals = diagnostics.residual_standardized.ravel()
    residual_mean = float(np.mean(residuals))
    residual_std = float(np.std(residuals, ddof=1))

    figure, axis = _figure(figsize=(7.0, 4.8))
    axis.hist(
        residuals,
        bins=18,
        density=True,
        alpha=0.65,
        label="Standardized residuals",
    )
    if residual_std > 0.0:
        x_values = np.linspace(float(residuals.min()), float(residuals.max()), 300)
        normal_density = np.exp(
            -0.5 * ((x_values - residual_mean) / residual_std) ** 2
        ) / (residual_std * np.sqrt(2.0 * np.pi))
        axis.plot(
            x_values,
            normal_density,
            linewidth=1.5,
            label="Matched normal density",
        )
    axis.axvline(0.0, linewidth=0.8, linestyle="--", color="0.35")
    axis.set_xlabel("Standardized residual")
    axis.set_ylabel("Density")
    axis.set_title(r"Final $\Pi$-PLS fit: residual distribution")
    axis.legend()
    _save_svg(figure, output_path)

def _write_manifest(
    output_dir: Path,
    *,
    data: PiPLSDataset,
    search: PiPLSSearchCV,
    selection: PiPLSSelection,
    rank_profile: PiPLSPredictorRankProfile,
    report: PiPLSOOFReport,
    oof_diagnostics: PredictionDiagnostics,
    fitted_diagnostics: PredictionDiagnostics,
    displayed_components: tuple[int, ...],
) -> Path:
    figures = [
        {"filename": filename, "sha256": _sha256(output_dir / filename)}
        for filename in FIGURE_FILENAMES
    ]
    manifest = {
        "schema_version": 2,
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
            "search_method": search.search_method,
            "search_is_exhaustive": search.search_is_exhaustive_,
            "max_predictor_rank": search.max_predictor_rank_,
            "evaluated_predictor_ranks": rank_profile.predictor_rank.tolist(),
            "predictor_rank_at_upper_boundary": bool(
                selection.predictor_rank == int(rank_profile.predictor_rank[-1])
            ),
            "displayed_components": [
                component + 1 for component in displayed_components
            ],
            "factor_sign_anchor": {"response": "TI", "sign": "positive"},
            "detailed_responses": list(data.target_names),
            "prediction_kind": oof_diagnostics.prediction_kind,
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
        "final_fit": {
            "prediction_kind": fitted_diagnostics.prediction_kind,
            "response_r2": [
                {"response": name, "value": float(value)}
                for name, value in zip(
                    data.target_names, fitted_diagnostics.response_r2, strict=True
                )
            ],
            "pooled_standardized_residuals": {
                "count": int(fitted_diagnostics.residual_standardized.size),
                "mean": float(np.mean(fitted_diagnostics.residual_standardized)),
                "sample_sd": float(
                    np.std(fitted_diagnostics.residual_standardized, ddof=1)
                ),
            },
        },
        "figures": figures,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


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
    _render_component_path(
        component_path,
        selected=None,
        title=r"Pulp $\Pi$-PLS component path before selection",
        output_path=output_dir / "component_path.svg",
    )

    selection = search.select(n_components=3)
    rank_profile = search.predictor_rank_profile(selection.n_components)
    report = search.oof_report(X, Y, selection=selection)
    if not np.all(report.oof_prediction_counts == 10):
        raise RuntimeError(
            "Repeated Pulp CV must produce ten OOF predictions per observation."
        )
    oof_diagnostics = prediction_diagnostics(
        Y,
        report.oof_predictions,
        prediction_kind=PREDICTION_KIND,
    )
    _render_component_path(
        component_path,
        selected=selection,
        title=r"Pulp $\Pi$-PLS selected component path",
        output_path=output_dir / "selected_component_path.svg",
    )
    _render_predictor_rank_profile(
        rank_profile,
        output_path=output_dir / "predictor_rank_profile.svg",
    )

    model = search.refit(X, Y, selection=selection)
    fitted_diagnostics = prediction_diagnostics(
        Y,
        model.predict(X),
        prediction_kind="fitted values",
    )
    factors = pipls_display_factors(
        model.decomposition_,
        response_index=response_names.index("TI"),
        response_sign="positive",
    )
    structure = latent_structure(model)
    displayed_components = tuple(range(selection.n_components))

    _render_biplot(
        structure,
        predictor_names=predictor_names,
        output_path=output_dir / "biplot.svg",
    )
    _render_predictor_directions(
        factors,
        predictor_names=predictor_names,
        n_components=selection.n_components,
        output_path=output_dir / "predictor_directions.svg",
    )
    _render_weighted_response_directions(
        factors,
        response_names=response_names,
        n_components=selection.n_components,
        output_path=output_dir / "weighted_response_directions.svg",
    )
    _render_observed_vs_predicted(
        oof_diagnostics,
        response_names=response_names,
        output_path=output_dir / "observed_vs_predicted.svg",
    )
    _render_residuals_vs_predicted(
        oof_diagnostics,
        response_names=response_names,
        output_path=output_dir / "residuals_vs_predicted.svg",
    )
    _render_standardized_rmse(
        oof_diagnostics,
        response_names=response_names,
        output_path=output_dir / "standardized_rmse.svg",
    )
    _render_final_fit_observed_vs_predicted(
        fitted_diagnostics,
        response_names=response_names,
        output_path=output_dir / "final_fit_observed_vs_predicted.svg",
    )
    _render_final_fit_r2(
        fitted_diagnostics,
        response_names=response_names,
        output_path=output_dir / "final_fit_r2.svg",
    )
    _render_final_fit_residual_distribution(
        fitted_diagnostics,
        output_path=output_dir / "final_fit_residual_distribution.svg",
    )

    return _write_manifest(
        output_dir,
        data=data,
        search=search,
        selection=selection,
        rank_profile=rank_profile,
        report=report,
        oof_diagnostics=oof_diagnostics,
        fitted_diagnostics=fitted_diagnostics,
        displayed_components=displayed_components,
    )


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
