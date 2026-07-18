"""Write the Tobacco randomized-SVD component path as a focused CSV smoke check."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from pipls import PiPLSPathCV, PiPLSRegression

ResultValue = int | float | str

RESULT_COLUMNS = (
    "n_components",
    "predictor_rank",
    "predictor_rank_policy",
    "response_standardized_cv_mse_mean",
    "response_standardized_cv_mse_fold_sd",
    "n_splits",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPOSITORY_ROOT / "datasets" / "tobacco"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "tobacco_path_smoke.csv"
N_COMPONENTS_VALUES = tuple(range(1, 9))


def read_tobacco_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the public predictor and response tables directly with pandas."""

    X = pd.read_csv(DATA_DIR / "X.csv")
    Y = pd.read_csv(DATA_DIR / "Y.csv")
    return X, Y


def fit_search(X: pd.DataFrame, Y: pd.DataFrame) -> PiPLSPathCV:
    """Evaluate the bounded auto path with an explicit randomized predictor SVD."""

    return PiPLSPathCV(
        estimator=PiPLSRegression(
            svd_solver="randomized",
            random_state=0,
        ),
        n_components_values=N_COMPONENTS_VALUES,
        search_method="auto",
        refit=False,
        n_jobs=1,
    ).fit(X, Y)


def _result_rows(search: PiPLSPathCV) -> list[dict[str, ResultValue]]:
    """Return one validated conditional predictor-rank row per component count."""

    results = search.component_path_results_
    rows: list[dict[str, ResultValue]] = []
    for index in range(len(results["n_components"])):
        row = {
            column: (
                results[column][index].item()
                if isinstance(results[column][index], np.generic)
                else results[column][index]
            )
            for column in RESULT_COLUMNS
        }
        if row["predictor_rank_policy"] != "optimized":
            raise RuntimeError("The Tobacco path must conditionally optimize predictor rank.")
        if not np.isfinite(row["response_standardized_cv_mse_mean"]):
            raise RuntimeError("The component-path mean CV-MSE must be finite.")
        if not np.isfinite(row["response_standardized_cv_mse_fold_sd"]):
            raise RuntimeError("The component-path fold SD must be finite.")
        rows.append(row)
    return rows


def run_benchmark() -> list[dict[str, ResultValue]]:
    """Run the Tobacco randomized-SVD component-path smoke check."""

    X, Y = read_tobacco_data()
    return _result_rows(fit_search(X, Y))


def write_results(
    rows: list[dict[str, ResultValue]],
    output: Path = DEFAULT_OUTPUT,
) -> None:
    """Write the component path as UTF-8 comma-separated data."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    """Run the smoke check and write its component-path CSV."""

    args = _parse_args()
    write_results(run_benchmark(), args.output)


if __name__ == "__main__":
    main()
