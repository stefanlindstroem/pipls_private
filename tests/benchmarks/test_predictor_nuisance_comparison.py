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


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_benchmark_module() -> ModuleType:
    path = _repository_root() / "benchmarks" / "predictor_nuisance_comparison.py"
    spec = importlib.util.spec_from_file_location(
        "predictor_nuisance_comparison_benchmark",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


def test_benchmark_generation_is_deterministic() -> None:
    scenario = BENCHMARK.SCENARIOS[2]
    first_train, first_test = BENCHMARK.generate_problem(scenario, BENCHMARK.SEEDS[0])
    second_train, second_test = BENCHMARK.generate_problem(scenario, BENCHMARK.SEEDS[0])

    assert_array_equal(first_train.X, second_train.X)
    assert_array_equal(first_train.Y, second_train.Y)
    assert_array_equal(first_test.X, second_test.X)
    assert_array_equal(first_test.Y, second_test.Y)


def test_nuisance_scenarios_change_strengths_without_redrawing_structure() -> None:
    seed = BENCHMARK.SEEDS[0]
    moderate, _ = BENCHMARK.generate_problem(BENCHMARK.SCENARIOS[1], seed)
    strong, _ = BENCHMARK.generate_problem(BENCHMARK.SCENARIOS[2], seed)
    moderate_truth = moderate.truth
    strong_truth = strong.truth
    assert moderate_truth is not None
    assert strong_truth is not None

    assert_array_equal(moderate_truth.x_shared_loadings, strong_truth.x_shared_loadings)
    assert_array_equal(
        moderate_truth.x_predictor_specific_loadings,
        strong_truth.x_predictor_specific_loadings,
    )
    assert_array_equal(moderate_truth.y_shared_loadings, strong_truth.y_shared_loadings)
    assert_array_equal(moderate_truth.shared_scores, strong_truth.shared_scores)
    assert_array_equal(
        moderate_truth.predictor_specific_scores,
        strong_truth.predictor_specific_scores,
    )
    assert_array_equal(moderate_truth.x_noise, strong_truth.x_noise)
    assert_array_equal(moderate_truth.y_noise, strong_truth.y_noise)
    assert_array_equal(moderate.Y, strong.Y)
    assert not np.array_equal(moderate.X, strong.X)


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
            "pipls_test_mse": 0.1,
            "pls_test_mse": 0.2,
            "pipls_minus_pls_mse": -0.1,
        }

    monkeypatch.setattr(BENCHMARK, "evaluate_scenario", fake_evaluate)

    rows = BENCHMARK.run_benchmark()
    expected = [
        (scenario.name, seed) for scenario in BENCHMARK.SCENARIOS for seed in BENCHMARK.SEEDS
    ]

    assert calls == expected
    assert [(row["scenario"], row["seed"]) for row in rows] == expected


def test_models_use_generator_declared_fixed_ranks() -> None:
    scenario = BENCHMARK.SCENARIOS[2]
    train, _ = BENCHMARK.generate_problem(scenario, BENCHMARK.SEEDS[0])
    truth = train.truth
    assert truth is not None

    pipls, pls = BENCHMARK.fit_models(train, BENCHMARK.SEEDS[0])

    assert pipls.n_components == truth.n_shared
    assert pipls.predictor_rank == truth.n_shared + truth.n_predictor_specific
    assert pipls.scale is True
    assert pipls.svd_solver == "full"
    assert pls.n_components == truth.n_shared
    assert pls.scale is True


def test_scientific_values_are_repeatable_finite_and_paired() -> None:
    scenario = BENCHMARK.SCENARIOS[1]
    seed = BENCHMARK.SEEDS[0]

    first = BENCHMARK.evaluate_scenario(scenario, seed)
    second = BENCHMARK.evaluate_scenario(scenario, seed)

    assert first == second
    assert set(first) == set(BENCHMARK.RESULT_COLUMNS)
    assert np.isfinite(first["pipls_test_mse"])
    assert np.isfinite(first["pls_test_mse"])
    assert np.isfinite(first["pipls_minus_pls_mse"])
    assert first["pipls_test_mse"] >= 0.0
    assert first["pls_test_mse"] >= 0.0
    assert first["pipls_minus_pls_mse"] == pytest.approx(
        first["pipls_test_mse"] - first["pls_test_mse"]
    )


def test_cli_writes_exact_minimal_csv_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "predictor_nuisance_comparison.csv"
    row = {
        "scenario": "example",
        "seed": 1,
        "pipls_test_mse": 0.1,
        "pls_test_mse": 0.2,
        "pipls_minus_pls_mse": -0.1,
    }
    monkeypatch.setattr(BENCHMARK, "run_benchmark", lambda: [row])
    monkeypatch.setattr(
        sys,
        "argv",
        ["predictor_nuisance_comparison.py", "--output", str(output)],
    )

    BENCHMARK.main()

    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == BENCHMARK.RESULT_COLUMNS
    assert rows == [{name: str(value) for name, value in row.items()}]
