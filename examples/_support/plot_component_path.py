"""Plot Pi-PLS paths and Pi-PLS-versus-PLS path comparisons from CSV files."""

from __future__ import annotations

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


def _add_pipls_path(axes: object, path: pd.DataFrame, *, label: str | None = None) -> None:
    x = path["n_components"].to_numpy()
    y = path["response_standardized_cv_mse_mean"].to_numpy()
    yerr = path["response_standardized_cv_mse_fold_sd"].to_numpy()
    axes.errorbar(x, y, yerr=yerr, fmt="o-", capsize=4, label=label)
    for n_components, mean_mse, predictor_rank in zip(
        x,
        y,
        path["predictor_rank"].to_numpy(),
        strict=True,
    ):
        axes.annotate(
            rf"$r_\pi={int(predictor_rank)}$",
            (n_components, mean_mse),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )


def _finish_path_figure(
    figure: object,
    axes: object,
    *,
    title: str,
    component_counts: np.ndarray,
    upper: float,
    show_legend: bool,
) -> None:
    axes.set_title(title)
    axes.set_xlabel("Number of response components")
    axes.set_ylabel("Response-standardized CV-MSE")
    axes.set_xticks(component_counts)
    axes.margins(x=0.05)
    axes.grid(axis="y", alpha=0.25)
    axes.set_ylim(0, max(1.0, 1.05 * upper))
    if show_legend:
        axes.legend()
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))


def plot_pipls_component_path(
    pipls_csv_path: Path,
    pdf_path: Path,
    *,
    title: str,
) -> None:
    """Read one canonical Pi-PLS path CSV and write its PDF."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = read_component_path(pipls_csv_path)
    figure, axes = plt.subplots(figsize=(8, 5))
    _add_pipls_path(axes, path)
    _finish_path_figure(
        figure,
        axes,
        title=title,
        component_counts=path["n_components"].to_numpy(),
        upper=float(
            np.max(
                path["response_standardized_cv_mse_mean"].to_numpy()
                + path["response_standardized_cv_mse_fold_sd"].to_numpy()
            )
        ),
        show_legend=False,
    )
    figure.savefig(pdf_path, format="pdf")
    plt.close(figure)


def plot_component_path_comparison(
    pipls_csv_path: Path,
    pls_csv_path: Path,
    pdf_path: Path,
    *,
    title: str,
) -> None:
    """Read canonical Pi-PLS and PLS path CSVs and write their comparison PDF."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pipls_path = read_component_path(pipls_csv_path)
    pls_path = read_pls_component_path(pls_csv_path)

    component_counts = pipls_path["n_components"].to_numpy()
    pls_component_counts = pls_path["n_components"].to_numpy()
    if not np.array_equal(pls_component_counts, component_counts):
        raise ValueError("Pi-PLS and PLS paths must contain the same component counts.")

    figure, axes = plt.subplots(figsize=(8, 5))
    _add_pipls_path(axes, pipls_path, label=r"$\Pi$-PLS")
    pls_mean = pls_path["response_standardized_cv_mse_mean"].to_numpy()
    pls_sd = pls_path["response_standardized_cv_mse_fold_sd"].to_numpy()
    axes.errorbar(
        pls_component_counts,
        pls_mean,
        yerr=pls_sd,
        fmt="s--",
        capsize=4,
        label=f"PLS ({pls_path['algorithm'].iloc[0]})",
    )
    _finish_path_figure(
        figure,
        axes,
        title=title,
        component_counts=component_counts,
        upper=max(
            float(
                np.max(
                    pipls_path["response_standardized_cv_mse_mean"].to_numpy()
                    + pipls_path["response_standardized_cv_mse_fold_sd"].to_numpy()
                )
            ),
            float(np.max(pls_mean + pls_sd)),
        ),
        show_legend=True,
    )
    figure.savefig(pdf_path, format="pdf")
    plt.close(figure)
