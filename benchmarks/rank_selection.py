"""Run the adaptive rank-selection benchmark for Pi-PLS."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipls import PiPLSPathCV, PiPLSRegression
from pipls.datasets import PiPLSDataset, make_pipls_train_test

ResultValue = str | int | float

RESULT_COLUMNS = (
    "scenario",
    "seed",
    "true_n_components",
    "selected_n_components",
    "true_predictor_rank",
    "selected_predictor_rank",
    "test_mse",
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "rank_selection.csv"
SEEDS = (1729, 2718, 3141)

N_TRAIN = 160
N_TEST = 160
N_FEATURES = 24
N_TARGETS = 6
NOISE = (0.2, 0.2)


@dataclass(frozen=True)
class Scenario:
    """One controlled latent-rank condition."""

    name: str
    n_shared: int
    n_predictor_specific: int
    shared_strength: tuple[float, ...]
    predictor_specific_strength: float | tuple[float, ...]


SCENARIOS = (
    Scenario(
        name="one_shared",
        n_shared=1,
        n_predictor_specific=0,
        shared_strength=(2.5,),
        predictor_specific_strength=1.0,
    ),
    Scenario(
        name="two_shared",
        n_shared=2,
        n_predictor_specific=0,
        shared_strength=(2.5, 1.5),
        predictor_specific_strength=1.0,
    ),
    Scenario(
        name="two_shared_with_predictor_nuisance",
        n_shared=2,
        n_predictor_specific=4,
        shared_strength=(2.5, 1.5),
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
        n_shared=scenario.n_shared,
        n_predictor_specific=scenario.n_predictor_specific,
        n_response_specific=0,
        shared_strength=scenario.shared_strength,
        predictor_specific_strength=scenario.predictor_specific_strength,
        noise=NOISE,
        random_state=seed,
    )


def fit_search(train: PiPLSDataset, seed: int) -> PiPLSPathCV:
    """Fit the adaptive Pi-PLS path with fold-local model standardization."""

    return PiPLSPathCV(
        estimator=PiPLSRegression(
            scale=True,
            svd_solver="full",
            random_state=seed,
        ),
        search_method="auto",
        samples_per_predictor_rank=10.0,
        cv=5,
        n_jobs=1,
    ).fit(train.X, train.Y)


def evaluate_scenario(scenario: Scenario, seed: int) -> dict[str, ResultValue]:
    """Select Pi-PLS ranks and return one independent-test result row."""

    train, test = generate_problem(scenario, seed)
    truth = train.truth
    if truth is None:
        raise RuntimeError("The public synthetic generator did not return latent truth.")

    search = fit_search(train, seed)
    residual = test.Y - search.predict(test.X)

    return {
        "scenario": scenario.name,
        "seed": seed,
        "true_n_components": truth.n_shared,
        "selected_n_components": search.best_n_components_,
        "true_predictor_rank": truth.n_shared + truth.n_predictor_specific,
        "selected_predictor_rank": search.best_predictor_rank_,
        "test_mse": float(np.mean(residual**2)),
    }


def run_benchmark() -> list[dict[str, ResultValue]]:
    """Return all rank-selection rows in stable scenario/seed order."""

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
