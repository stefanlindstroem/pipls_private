"""Plot Pi-PLS and optional standard-PLS component paths from canonical CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PIPLS_REQUIRED_COLUMNS = (
    "n_components",
    "predictor_rank",
    "predictor_rank_policy",
    "response_standardized_cv_mse_mean",
    "response_standardized_cv_mse_fold_sd",
    "n_splits",
)
PLS_REQUIRED_COLUMNS = (
    "n_components",
    "algorithm",
    "response_standardized_cv_mse_mean",
    "response_standardized_cv_mse_fold_sd",
    "n_splits",
)


def _read_ordered_path(csv_path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    path = pd.read_csv(csv_path)
    missing = [column for column in required_columns if column not in path.columns]
    if missing:
        raise ValueError(f"Component-path CSV is missing columns: {missing!r}.")
    if path.empty:
        raise ValueError("Component-path CSV must contain at least one row.")
    if path["n_components"].duplicated().any():
        raise ValueError("Component-path CSV must contain one row per n_components value.")
    if not path["n_components"].is_monotonic_increasing:
        raise ValueError("Component-path CSV rows must be ordered by n_components.")
    if (path["response_standardized_cv_mse_fold_sd"] < 0).any():
        raise ValueError("Fold standard deviations must be nonnegative.")
    return path


def read_component_path(csv_path: Path) -> pd.DataFrame:
    """Read and validate a Pi-PLS component-path CSV before plotting."""

    return _read_ordered_path(csv_path, PIPLS_REQUIRED_COLUMNS)


def read_pls_component_path(csv_path: Path) -> pd.DataFrame:
    """Read and validate a standard-PLS component-path CSV before plotting."""

    path = _read_ordered_path(csv_path, PLS_REQUIRED_COLUMNS)
    algorithms = path["algorithm"].drop_duplicates().tolist()
    if len(algorithms) != 1 or not str(algorithms[0]).strip():
        raise ValueError("PLS component-path CSV must contain one nonempty algorithm value.")
    return path


def plot_component_path(
    csv_path: Path,
    pdf_path: Path,
    *,
    title: str,
    pls_csv_path: Path | None = None,
) -> None:
    """Read canonical path CSVs and write a compact comparison PDF."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = read_component_path(csv_path)
    x = path["n_components"].to_numpy()
    y = path["response_standardized_cv_mse_mean"].to_numpy()
    yerr = path["response_standardized_cv_mse_fold_sd"].to_numpy()
    ranks = path["predictor_rank"].to_numpy()

    figure, axes = plt.subplots(figsize=(8, 5))
    axes.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        capsize=4,
        label=r"$\Pi$-PLS",
    )
    upper_values = [float(np.max(y + yerr))]

    if pls_csv_path is not None:
        pls_path = read_pls_component_path(pls_csv_path)
        pls_x = pls_path["n_components"].to_numpy()
        if not np.array_equal(pls_x, x):
            raise ValueError("Pi-PLS and PLS paths must contain the same component counts.")
        pls_y = pls_path["response_standardized_cv_mse_mean"].to_numpy()
        pls_yerr = pls_path["response_standardized_cv_mse_fold_sd"].to_numpy()
        algorithm = str(pls_path["algorithm"].iloc[0])
        axes.errorbar(
            pls_x,
            pls_y,
            yerr=pls_yerr,
            fmt="s--",
            capsize=4,
            label=f"PLS ({algorithm})",
        )
        upper_values.append(float(np.max(pls_y + pls_yerr)))

    axes.set_title(title)
    axes.set_xlabel("Number of response components")
    axes.set_ylabel("Response-standardized CV-MSE")
    axes.set_xticks(x)
    axes.margins(x=0.05)
    axes.grid(axis="y", alpha=0.25)
    for n_components, mean_mse, predictor_rank in zip(x, y, ranks, strict=True):
        axes.annotate(
            rf"$r_\pi={int(predictor_rank)}$",
            (n_components, mean_mse),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    axes.set_ylim(0, max(1.0, 1.05 * max(upper_values)))
    axes.legend()
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, format="pdf")
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--title", default="Component-path comparison")
    parser.add_argument("--pls-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    """Plot one Pi-PLS path and an optional standard-PLS comparison."""

    args = _parse_args()
    plot_component_path(
        args.csv_path,
        args.pdf_path,
        title=args.title,
        pls_csv_path=args.pls_csv,
    )


if __name__ == "__main__":
    main()
