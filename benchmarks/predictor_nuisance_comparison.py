"""Compare fixed Pi-PLS and ordinary PLS under predictor-specific nuisance."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from pipls import PiPLSRegression
from pipls.datasets import PiPLSDataset, make_pipls_train_test

ResultValue = str | int | float

RESULT_COLUMNS = (
    "scenario",
    "seed",
    "pipls_test_mse",
    "pls_test_mse",
    "pipls_minus_pls_mse",
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "predictor_nuisance_comparison.csv"
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
    """One controlled predictor-specific nuisance level."""

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
        name="moderate_predictor_nuisance",
        n_predictor_specific=4,
        predictor_specific_strength=(1.5, 1.25, 1.0, 0.75),
    ),
    Scenario(
        name="strong_predictor_nuisance",
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


def fit_models(
    train: PiPLSDataset,
    seed: int,
) -> tuple[PiPLSRegression, PLSRegression]:
    """Fit paired fixed-component models using generator-declared ranks."""

    truth = train.truth
    if truth is None:
        raise RuntimeError("The public synthetic generator did not return latent truth.")

    pipls = PiPLSRegression(
        n_components=truth.n_shared,
        predictor_rank=truth.n_shared + truth.n_predictor_specific,
        scale=True,
        svd_solver="full",
        random_state=seed,
    ).fit(train.X, train.Y)
    pls = PLSRegression(
        n_components=truth.n_shared,
        scale=True,
    ).fit(train.X, train.Y)
    return pipls, pls


def evaluate_scenario(scenario: Scenario, seed: int) -> dict[str, ResultValue]:
    """Fit both methods and return one paired independent-test result row."""

    train, test = generate_problem(scenario, seed)
    pipls, pls = fit_models(train, seed)

    pipls_residual = test.Y - pipls.predict(test.X)
    pls_residual = test.Y - pls.predict(test.X)
    pipls_test_mse = float(np.mean(pipls_residual**2))
    pls_test_mse = float(np.mean(pls_residual**2))

    return {
        "scenario": scenario.name,
        "seed": seed,
        "pipls_test_mse": pipls_test_mse,
        "pls_test_mse": pls_test_mse,
        "pipls_minus_pls_mse": pipls_test_mse - pls_test_mse,
    }


def run_benchmark() -> list[dict[str, ResultValue]]:
    """Return all paired comparison rows in stable scenario/seed order."""

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
