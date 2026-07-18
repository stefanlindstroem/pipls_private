"""Plot a Pi-PLS component-path PDF from its canonical CSV table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "n_components",
    "predictor_rank",
    "predictor_rank_policy",
    "response_standardized_cv_mse_mean",
    "response_standardized_cv_mse_fold_sd",
    "n_splits",
)


def read_component_path(csv_path: Path) -> pd.DataFrame:
    """Read and validate a component-path CSV before plotting."""

    path = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in path.columns]
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


def plot_component_path(csv_path: Path, pdf_path: Path, *, title: str) -> None:
    """Read ``csv_path`` and write a compact component-path PDF."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = read_component_path(csv_path)
    x = path["n_components"].to_numpy()
    y = path["response_standardized_cv_mse_mean"].to_numpy()
    yerr = path["response_standardized_cv_mse_fold_sd"].to_numpy()
    ranks = path["predictor_rank"].to_numpy()

    figure, axes = plt.subplots(figsize=(6.4, 4.2))
    axes.errorbar(x, y, yerr=yerr, fmt="o-", capsize=4)
    axes.set_title(title)
    axes.set_xlabel("Number of response components")
    axes.set_ylabel("Response-standardized CV-MSE")
    axes.set_xticks(x)
    axes.margins(x=0.08)
    axes.grid(axis="y", alpha=0.25)
    for n_components, mean_mse, predictor_rank in zip(x, y, ranks, strict=True):
        axes.annotate(
            rf"$r_\pi={int(predictor_rank)}$",
            (n_components, mean_mse),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    figure.text(
        0.5,
        0.01,
        "Error bars show fold-to-fold SD.",
        ha="center",
        fontsize="small",
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, format="pdf")
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--title", default="Pi-PLS component path")
    return parser.parse_args()


def main() -> None:
    """Plot one component-path CSV from the command line."""

    args = _parse_args()
    plot_component_path(args.csv_path, args.pdf_path, title=args.title)


if __name__ == "__main__":
    main()
