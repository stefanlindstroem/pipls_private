from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from threadpoolctl import threadpool_limits


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_benchmark_module() -> ModuleType:
    path = _repository_root() / "benchmarks" / "solver_consistency.py"
    spec = importlib.util.spec_from_file_location("solver_consistency_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


def test_benchmark_generation_is_deterministic() -> None:
    scenario = BENCHMARK.SCENARIOS[0]
    first_train, first_test = BENCHMARK.generate_problem(scenario, BENCHMARK.SEEDS[0])
    second_train, second_test = BENCHMARK.generate_problem(scenario, BENCHMARK.SEEDS[0])

    assert_array_equal(first_train.X, second_train.X)
    assert_array_equal(first_train.Y, second_train.Y)
    assert_array_equal(first_test.X, second_test.X)
    assert_array_equal(first_test.Y, second_test.Y)


def test_scenarios_change_geometry_at_fixed_training_matrix_size() -> None:
    shapes = {(scenario.n_train, scenario.n_features) for scenario in BENCHMARK.SCENARIOS}
    entry_counts = {scenario.n_train * scenario.n_features for scenario in BENCHMARK.SCENARIOS}

    assert len(shapes) == len(BENCHMARK.SCENARIOS)
    assert entry_counts == {96 * 384}


def test_run_benchmark_uses_stable_scenario_seed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_evaluate(scenario: Any, seed: int) -> dict[str, str | int | float]:
        name = str(scenario.name)
        calls.append((name, seed))
        return {
            "scenario": name,
            "seed": seed,
            "prediction_relative_difference": 0.01,
            "coefficient_relative_difference": 0.02,
        }

    monkeypatch.setattr(BENCHMARK, "evaluate_scenario", fake_evaluate)

    rows = BENCHMARK.run_benchmark()
    expected = [
        (scenario.name, seed) for scenario in BENCHMARK.SCENARIOS for seed in BENCHMARK.SEEDS
    ]

    assert calls == expected
    assert [(row["scenario"], row["seed"]) for row in rows] == expected


def test_models_use_paired_generator_declared_ranks_and_solvers() -> None:
    scenario = BENCHMARK.SCENARIOS[1]
    train, _ = BENCHMARK.generate_problem(scenario, BENCHMARK.SEEDS[0])
    truth = train.truth
    assert truth is not None

    with threadpool_limits(limits=1):
        full, randomized = BENCHMARK.fit_models(train, BENCHMARK.SEEDS[0])
    expected_predictor_rank = truth.n_shared + truth.n_predictor_specific

    assert full.n_components == randomized.n_components == truth.n_shared
    assert full.predictor_rank == randomized.predictor_rank == expected_predictor_rank
    assert full.scale is randomized.scale is True
    assert full.random_state == randomized.random_state == BENCHMARK.SEEDS[0]
    assert full.decomposition_.predictor_svd_solver == "full"
    assert randomized.decomposition_.predictor_svd_solver == "randomized"
    assert_array_equal(full.x_mean_, randomized.x_mean_)
    assert_array_equal(full.x_scale_, randomized.x_scale_)
    assert_array_equal(full.y_mean_, randomized.y_mean_)
    assert_array_equal(full.y_scale_, randomized.y_scale_)


def test_relative_difference_uses_full_reference_norm() -> None:
    reference = np.asarray([[3.0, 4.0]], dtype=np.float64)
    comparison = np.asarray([[0.0, 4.0]], dtype=np.float64)

    observed = BENCHMARK._relative_frobenius_difference(reference, comparison)

    assert observed == pytest.approx(3.0 / 5.0)


def test_scientific_values_are_repeatable_finite_and_nonnegative() -> None:
    scenario = BENCHMARK.SCENARIOS[0]
    seed = BENCHMARK.SEEDS[0]

    with threadpool_limits(limits=1):
        first = BENCHMARK.evaluate_scenario(scenario, seed)
        second = BENCHMARK.evaluate_scenario(scenario, seed)

    assert first == second
    assert set(first) == set(BENCHMARK.RESULT_COLUMNS)
    for name in (
        "prediction_relative_difference",
        "coefficient_relative_difference",
    ):
        assert np.isfinite(first[name])
        assert first[name] >= 0.0


def test_cli_writes_exact_minimal_csv_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "solver_consistency.csv"
    row = {
        "scenario": "example",
        "seed": 1,
        "prediction_relative_difference": 0.01,
        "coefficient_relative_difference": 0.02,
    }
    monkeypatch.setattr(BENCHMARK, "run_benchmark", lambda: [row])
    monkeypatch.setattr(
        sys,
        "argv",
        ["solver_consistency.py", "--output", str(output)],
    )

    BENCHMARK.main()

    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == BENCHMARK.RESULT_COLUMNS
    assert rows == [{name: str(value) for name, value in row.items()}]
