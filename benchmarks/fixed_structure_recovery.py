"""Run the fixed-structure recovery benchmark for Pi-PLS."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pipls import PiPLSRegression
from pipls.datasets import PiPLSDataset, make_pipls_train_test

FloatArray = NDArray[np.float64]
ResultValue = str | int | float

RESULT_COLUMNS = (
    "scenario",
    "seed",
    "test_mse",
    "predictor_shared_capture",
    "predictor_signal_capture",
    "response_shared_capture",
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "fixed_structure_recovery.csv"
SEEDS = (1729, 2718, 3141)

N_TRAIN = 160
N_TEST = 160
N_FEATURES = 24
N_TARGETS = 6
N_SHARED = 2
SHARED_STRENGTH = (2.5, 1.5)
NOISE = (0.2, 0.2)


@dataclass(frozen=True)
class Scenario:
    """One controlled predictor-structure condition."""

    name: str
    n_predictor_specific: int
    predictor_specific_strength: float | tuple[float, ...]


SCENARIOS = (
    Scenario(
        name="shared_only",
        n_predictor_specific=0,
        predictor_specific_strength=1.0,
    ),
    Scenario(
        name="predictor_specific_nuisance",
        n_predictor_specific=4,
        predictor_specific_strength=(3.0, 2.5, 2.0, 1.5),
    ),
)


def generate_problem(scenario: Scenario, seed: int) -> tuple[PiPLSDataset, PiPLSDataset]:
    """Generate one deterministic train/test problem for a benchmark scenario."""

    return make_pipls_train_test(
        n_train=N_TRAIN,
        n_test=N_TEST,
        n_features=N_FEATURES,
        n_targets=N_TARGETS,
        n_shared=N_SHARED,
        n_predictor_specific=scenario.n_predictor_specific,
        n_response_specific=0,
        shared_strength=SHARED_STRENGTH,
        predictor_specific_strength=scenario.predictor_specific_strength,
        noise=NOISE,
        random_state=seed,
    )


def _subspace_capture(true_basis: FloatArray, estimated_basis: FloatArray) -> float:
    """Return the mean squared canonical-correlation capture of a true subspace."""

    if true_basis.ndim != 2 or estimated_basis.ndim != 2:
        raise ValueError("Subspace bases must be two-dimensional.")
    if true_basis.shape[0] != estimated_basis.shape[0]:
        raise ValueError("Subspace bases must use the same ambient dimension.")
    if true_basis.shape[1] == 0:
        raise ValueError("The true subspace must contain at least one direction.")
    if np.linalg.matrix_rank(true_basis) != true_basis.shape[1]:
        raise ValueError("The true subspace basis must have full column rank.")
    if np.linalg.matrix_rank(estimated_basis) != estimated_basis.shape[1]:
        raise ValueError("The estimated subspace basis must have full column rank.")

    true_orthogonal = np.linalg.qr(true_basis, mode="reduced")[0]
    estimated_orthogonal = np.linalg.qr(estimated_basis, mode="reduced")[0]
    squared_cosines = (
        np.linalg.svd(
            true_orthogonal.T @ estimated_orthogonal,
            compute_uv=False,
        )
        ** 2
    )
    capture = float(np.sum(squared_cosines) / true_basis.shape[1])
    return float(np.clip(capture, 0.0, 1.0))


def _scaled_truth_basis(
    loadings: FloatArray,
    observed_scale: FloatArray,
    fitted_scale: FloatArray,
) -> FloatArray:
    """Map generator loadings into the estimator's fitted standardized coordinates."""

    coordinate_scale = observed_scale / fitted_scale
    return np.asarray(coordinate_scale[:, None] * loadings, dtype=np.float64)


def evaluate_scenario(scenario: Scenario, seed: int) -> dict[str, ResultValue]:
    """Fit fixed oracle-rank Pi-PLS and return one benchmark result row."""

    train, test = generate_problem(scenario, seed)
    truth = train.truth
    if truth is None:
        raise RuntimeError("The public synthetic generator did not return latent truth.")

    predictor_signal_rank = truth.n_shared + truth.n_predictor_specific
    model = PiPLSRegression(
        n_components=truth.n_shared,
        predictor_rank=predictor_signal_rank,
        scale=True,
        svd_solver="full",
        random_state=seed,
    ).fit(train.X, train.Y)

    residual = test.Y - model.predict(test.X)
    test_mse = float(np.mean(residual**2))

    predictor_shared_truth = _scaled_truth_basis(
        truth.x_shared_loadings,
        truth.feature_scale,
        model.x_scale_,
    )
    predictor_signal_truth = _scaled_truth_basis(
        np.column_stack((truth.x_shared_loadings, truth.x_predictor_specific_loadings)),
        truth.feature_scale,
        model.x_scale_,
    )
    response_shared_truth = _scaled_truth_basis(
        truth.y_shared_loadings,
        truth.target_scale,
        model.y_scale_,
    )

    return {
        "scenario": scenario.name,
        "seed": seed,
        "test_mse": test_mse,
        "predictor_shared_capture": _subspace_capture(
            predictor_shared_truth,
            model.decomposition_.P,
        ),
        "predictor_signal_capture": _subspace_capture(
            predictor_signal_truth,
            model.decomposition_.Pi,
        ),
        "response_shared_capture": _subspace_capture(
            response_shared_truth,
            model.decomposition_.Q,
        ),
    }


def run_benchmark() -> list[dict[str, ResultValue]]:
    """Return all fixed-structure recovery rows in stable scenario/seed order."""

    return [evaluate_scenario(scenario, seed) for scenario in SCENARIOS for seed in SEEDS]


def write_results(
    rows: list[dict[str, ResultValue]],
    output: Path = DEFAULT_OUTPUT,
) -> None:
    """Write benchmark rows as one minimal UTF-8 comma-separated table."""

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
    """Run the benchmark and write its dedicated CSV output."""

    args = _parse_args()
    write_results(run_benchmark(), args.output)


if __name__ == "__main__":
    main()
