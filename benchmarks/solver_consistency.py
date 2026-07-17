"""Compare full and randomized predictor SVD for fixed Pi-PLS models."""

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
    "prediction_relative_difference",
    "coefficient_relative_difference",
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "solver_consistency.csv"
SEEDS = (1729, 2718, 3141)

N_TEST = 96
N_TARGETS = 8
N_SHARED = 3
N_PREDICTOR_SPECIFIC = 5
SHARED_STRENGTH = (3.0, 2.0, 1.5)
PREDICTOR_SPECIFIC_STRENGTH = (2.5, 2.0, 1.5, 1.0, 0.75)
NOISE = (0.3, 0.2)


@dataclass(frozen=True)
class Scenario:
    """One high-dimensional training-matrix geometry."""

    name: str
    n_train: int
    n_features: int


SCENARIOS = (
    Scenario(name="wide_n96_p384", n_train=96, n_features=384),
    Scenario(name="square_n192_p192", n_train=192, n_features=192),
    Scenario(name="tall_n384_p96", n_train=384, n_features=96),
)


def generate_problem(scenario: Scenario, seed: int) -> tuple[PiPLSDataset, PiPLSDataset]:
    """Generate one deterministic train/test problem for a benchmark scenario."""

    return make_pipls_train_test(
        n_train=scenario.n_train,
        n_test=N_TEST,
        n_features=scenario.n_features,
        n_targets=N_TARGETS,
        n_shared=N_SHARED,
        n_predictor_specific=N_PREDICTOR_SPECIFIC,
        n_response_specific=0,
        shared_strength=SHARED_STRENGTH,
        predictor_specific_strength=PREDICTOR_SPECIFIC_STRENGTH,
        noise=NOISE,
        random_state=seed,
    )


def fit_models(
    train: PiPLSDataset,
    seed: int,
) -> tuple[PiPLSRegression, PiPLSRegression]:
    """Fit paired fixed-rank models differing only in predictor SVD solver."""

    truth = train.truth
    if truth is None:
        raise RuntimeError("The public synthetic generator did not return latent truth.")

    predictor_rank = truth.n_shared + truth.n_predictor_specific
    full = PiPLSRegression(
        n_components=truth.n_shared,
        predictor_rank=predictor_rank,
        scale=True,
        svd_solver="full",
        random_state=seed,
    ).fit(train.X, train.Y)
    randomized = PiPLSRegression(
        n_components=truth.n_shared,
        predictor_rank=predictor_rank,
        scale=True,
        svd_solver="randomized",
        random_state=seed,
    ).fit(train.X, train.Y)
    return full, randomized


def _relative_frobenius_difference(reference: FloatArray, comparison: FloatArray) -> float:
    """Return a Frobenius difference relative to the exact-path reference norm."""

    if reference.shape != comparison.shape:
        raise ValueError("Compared arrays must have identical shapes.")
    numerator = float(np.linalg.norm(comparison - reference, ord="fro"))
    denominator = max(
        float(np.linalg.norm(reference, ord="fro")),
        float(np.finfo(np.float64).eps),
    )
    return numerator / denominator


def evaluate_scenario(scenario: Scenario, seed: int) -> dict[str, ResultValue]:
    """Fit paired solvers and return one independent-test consistency row."""

    train, test = generate_problem(scenario, seed)
    full, randomized = fit_models(train, seed)

    full_prediction = np.asarray(full.predict(test.X), dtype=np.float64)
    randomized_prediction = np.asarray(randomized.predict(test.X), dtype=np.float64)

    return {
        "scenario": scenario.name,
        "seed": seed,
        "prediction_relative_difference": _relative_frobenius_difference(
            full_prediction,
            randomized_prediction,
        ),
        "coefficient_relative_difference": _relative_frobenius_difference(
            np.asarray(full.coef_, dtype=np.float64),
            np.asarray(randomized.coef_, dtype=np.float64),
        ),
    }


def run_benchmark() -> list[dict[str, ResultValue]]:
    """Return all solver-consistency rows in stable scenario/seed order."""

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
