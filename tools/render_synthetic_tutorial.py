"""Generate deterministic figures for the synthetic entry tutorial."""

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
matplotlib.rcParams["svg.hashsalt"] = "pipls-synthetic-tutorial"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from pipls import PiPLSSearchCV  # noqa: E402
from pipls.component_path import PiPLSComponentPath, PiPLSSelection  # noqa: E402
from pipls.datasets import make_pipls_train_test  # noqa: E402
from pipls.inspection import prediction_diagnostics  # noqa: E402

DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "docs" / "assets" / "generated" / "synthetic"
)
CV = KFold(n_splits=5, shuffle=True, random_state=0)
FIGURE_FILENAMES = (
    "component_path.svg",
    "selected_component_path.svg",
    "predictor_rank_profile.svg",
    "observed_vs_predicted.svg",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_svg(figure: Figure, path: Path) -> None:
    figure.savefig(
        path,
        format="svg",
        metadata={"Creator": "Pi-PLS repository", "Date": None},
    )
    plt.close(figure)


def _render_component_path(
    path: PiPLSComponentPath,
    *,
    selected: PiPLSSelection | None,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5), layout="constrained")
    axis.errorbar(
        path.n_components,
        path.cv_mse_mean,
        yerr=path.cv_mse_std,
        fmt="o-",
        capsize=4,
    )
    if selected is not None:
        axis.scatter(
            [selected.n_components],
            [selected.cv_mse_mean],
            marker="D",
            s=70,
            label=f"Chosen: {selected.n_components} components",
            zorder=3,
        )
        axis.legend()
    axis.set_xlabel("Number of components")
    axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
    axis.set_title(title)
    axis.set_xticks(path.n_components)
    upper = float(np.max(path.cv_mse_mean + path.cv_mse_std))
    axis.set_ylim(0.0, max(1.0, 1.05 * upper))
    axis.grid(axis="y", alpha=0.25)
    _save_svg(figure, output_path)


def render_synthetic_tutorial_assets(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Generate the four synthetic tutorial figures and return the manifest path."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    train, test = make_pipls_train_test(
        n_train=120,
        n_test=60,
        n_features=8,
        n_targets=3,
        n_shared=2,
        n_predictor_specific=2,
        n_response_specific=1,
        shared_strength=(2.5, 1.5),
        noise=(0.2, 0.25),
        random_state=0,
    )
    search = PiPLSSearchCV(cv=CV).fit(train.X, train.Y)
    path = search.component_path_
    _render_component_path(
        path,
        selected=None,
        title=r"Synthetic $\Pi$-PLS component path before selection",
        output_path=output_dir / "component_path.svg",
    )

    chosen_n_components = 2
    selection = search.select(n_components=chosen_n_components)
    selected_path = search.component_path_
    rank_profile = search.predictor_rank_profile(selection.n_components)
    _render_component_path(
        selected_path,
        selected=selection,
        title=r"Synthetic $\Pi$-PLS selected component path",
        output_path=output_dir / "selected_component_path.svg",
    )

    figure, axis = plt.subplots(figsize=(7.0, 4.5), layout="constrained")
    axis.errorbar(
        rank_profile.predictor_rank,
        rank_profile.cv_mse_mean,
        yerr=rank_profile.cv_mse_std,
        fmt="o-",
        capsize=4,
    )
    axis.scatter(
        [rank_profile.selection.predictor_rank],
        [rank_profile.selection.cv_mse_mean],
        marker="D",
        s=70,
        label=f"CV-MSE minimum: rank {rank_profile.selection.predictor_rank}",
        zorder=3,
    )
    axis.set_xlabel("Predictor rank")
    axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
    axis.set_title(
        rf"Synthetic $\Pi$-PLS predictor-rank profile at "
        f"{selection.n_components} components"
    )
    axis.set_xticks(rank_profile.predictor_rank)
    upper = float(np.max(rank_profile.cv_mse_mean + rank_profile.cv_mse_std))
    axis.set_ylim(0.0, max(1.0, 1.05 * upper))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_svg(figure, output_dir / "predictor_rank_profile.svg")

    model = search.refit(
        train.X,
        train.Y,
        selection=selection,
    )
    predictions = model.predict(test.X)
    diagnostics = prediction_diagnostics(
        test.Y,
        predictions,
        prediction_kind="external test predictions",
    )

    figure, axis = plt.subplots(figsize=(6.2, 5.0), layout="constrained")
    for response, name in enumerate(test.target_names):
        axis.scatter(
            diagnostics.observed_standardized[:, response],
            diagnostics.predicted_standardized[:, response],
            label=name,
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
    axis.set_title(rf"Synthetic $\Pi$-PLS — {diagnostics.prediction_kind}")
    axis.legend(title="Response")
    _save_svg(figure, output_dir / "observed_vs_predicted.svg")

    manifest = {
        "schema_version": 1,
        "generator": {
            "name": "make_pipls_train_test",
            "random_state": 0,
            "n_train": train.n_samples,
            "n_test": test.n_samples,
            "n_features": train.n_features,
            "n_targets": train.n_targets,
            "n_shared": train.truth.n_shared,
            "n_predictor_specific": train.truth.n_predictor_specific,
            "n_response_specific": train.truth.n_response_specific,
        },
        "analysis": {
            "chosen_n_components": selection.n_components,
            "chosen_predictor_rank": selection.predictor_rank,
            "evaluated_component_counts": path.n_components.tolist(),
            "evaluated_predictor_ranks": rank_profile.predictor_rank.tolist(),
            "prediction_kind": diagnostics.prediction_kind,
            "external_test_r2": model.score(test.X, test.Y),
        },
        "figures": [
            {"filename": filename, "sha256": _sha256(output_dir / filename)}
            for filename in FIGURE_FILENAMES
        ],
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
    manifest = render_synthetic_tutorial_assets(args.output_dir)
    print(f"Wrote synthetic tutorial figures to {manifest.parent}")


if __name__ == "__main__":
    main()
