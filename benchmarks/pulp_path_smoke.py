"""Run the Pulp Pi-PLS path-selection smoke check.

The reported validation diagnostics are selection-conditioned. They describe the
same cross-validation result used for rank selection and are not an unbiased
post-selection performance estimate.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from pipls import PiPLSPathCV

ResultValue = int | float

RESULT_COLUMNS = (
    "selected_n_components",
    "selected_predictor_rank",
    "selection_conditioned_response_standardized_mse",
    "selection_conditioned_pooled_oof_r2",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPOSITORY_ROOT / "datasets" / "pulp"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "pulp_path_smoke.csv"
N_COMPONENTS_VALUES = (1, 2, 3, 4)


def read_pulp_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the public predictor and response tables directly with pandas."""

    X = pd.read_csv(DATA_DIR / "X.csv")
    Y = pd.read_csv(DATA_DIR / "Y.csv")
    return X, Y


def fit_search(X: pd.DataFrame, Y: pd.DataFrame) -> PiPLSPathCV:
    """Fit the public path workflow over the complete Pulp predictor-rank range."""

    return PiPLSPathCV(
        n_components_values=N_COMPONENTS_VALUES,
        max_predictor_rank=X.shape[1],
        search_method="auto",
        cv=5,
        return_oof_predictions=True,
        n_jobs=1,
    ).fit(X, Y)


def _result_row(search: PiPLSPathCV, n_samples: int) -> dict[str, ResultValue]:
    """Return one minimal row after checking the OOF reporting contract."""

    report = search.validation_report_
    if report.estimate_kind != "selection-conditioned":
        raise RuntimeError("The path-search validation report must be selection-conditioned.")
    if report.oof_predictions is None or report.oof_prediction_counts is None:
        raise RuntimeError("The path search did not return OOF predictions and counts.")
    if report.oof_predictions.shape[0] != n_samples:
        raise RuntimeError("The OOF prediction table does not preserve the input row count.")
    if not np.all(report.oof_prediction_counts == 1):
        raise RuntimeError("Five-fold CV must produce exactly one OOF prediction per Pulp row.")
    if report.pooled_oof_r2 is None:
        raise RuntimeError("The path search did not report pooled OOF R2.")

    return {
        "selected_n_components": int(search.best_n_components_),
        "selected_predictor_rank": int(search.best_predictor_rank_),
        "selection_conditioned_response_standardized_mse": float(
            report.mean_response_standardized_mse
        ),
        "selection_conditioned_pooled_oof_r2": float(report.pooled_oof_r2),
    }


def run_benchmark() -> list[dict[str, ResultValue]]:
    """Run the single-dataset smoke check and return its one result row."""

    X, Y = read_pulp_data()
    search = fit_search(X, Y)
    return [_result_row(search, len(X))]


def write_results(
    rows: list[dict[str, ResultValue]],
    output: Path = DEFAULT_OUTPUT,
) -> None:
    """Write the smoke-check row as one minimal UTF-8 comma-separated table."""

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
    """Run the smoke check and write its dedicated CSV output."""

    args = _parse_args()
    write_results(run_benchmark(), args.output)


if __name__ == "__main__":
    main()
