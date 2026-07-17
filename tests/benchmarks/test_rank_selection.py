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
    path = _repository_root() / "benchmarks" / "rank_selection.py"
    spec = importlib.util.spec_from_file_location("rank_selection_benchmark", path)
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
            "true_n_components": 1,
            "selected_n_components": 1,
            "true_predictor_rank": 1,
            "selected_predictor_rank": 1,
            "test_mse": 0.0,
        }

    monkeypatch.setattr(BENCHMARK, "evaluate_scenario", fake_evaluate)

    rows = BENCHMARK.run_benchmark()
    expected = [
        (scenario.name, seed)
        for scenario in BENCHMARK.SCENARIOS
        for seed in BENCHMARK.SEEDS
    ]

    assert calls == expected
    assert [(row["scenario"], row["seed"]) for row in rows] == expected


def test_benchmark_scientific_values_are_repeatable_and_use_generator_truth() -> None:
    scenario = BENCHMARK.SCENARIOS[2]
    seed = BENCHMARK.SEEDS[0]

    first = BENCHMARK.evaluate_scenario(scenario, seed)
    second = BENCHMARK.evaluate_scenario(scenario, seed)
    train, _ = BENCHMARK.generate_problem(scenario, seed)
    truth = train.truth
    assert truth is not None

    assert first == second
    assert set(first) == set(BENCHMARK.RESULT_COLUMNS)
    assert first["true_n_components"] == truth.n_shared
    assert first["true_predictor_rank"] == truth.n_shared + truth.n_predictor_specific
    assert isinstance(first["selected_n_components"], int)
    assert isinstance(first["selected_predictor_rank"], int)
    assert 1 <= first["selected_n_components"] <= first["selected_predictor_rank"]
    assert np.isfinite(first["test_mse"])
    assert first["test_mse"] >= 0.0


def test_selected_ranks_match_the_refitted_model_and_search_bound() -> None:
    scenario = BENCHMARK.SCENARIOS[2]
    train, _ = BENCHMARK.generate_problem(scenario, BENCHMARK.SEEDS[0])
    search = BENCHMARK.fit_search(train, BENCHMARK.SEEDS[0])

    assert search.best_n_components_ == search.best_estimator_.n_components
    assert search.best_predictor_rank_ == search.best_estimator_.predictor_rank
    assert search.best_n_components_ == search.best_pipls_.n_components
    assert search.best_predictor_rank_ == search.best_pipls_.predictor_rank
    assert 1 <= search.best_n_components_ <= search.best_predictor_rank_
    assert search.best_predictor_rank_ <= search.max_predictor_rank_


def test_cli_writes_exact_minimal_csv_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "rank_selection.csv"
    row = {
        "scenario": "example",
        "seed": 1,
        "true_n_components": 2,
        "selected_n_components": 2,
        "true_predictor_rank": 6,
        "selected_predictor_rank": 5,
        "test_mse": 0.1,
    }
    monkeypatch.setattr(BENCHMARK, "run_benchmark", lambda: [row])
    monkeypatch.setattr(
        sys,
        "argv",
        ["rank_selection.py", "--output", str(output)],
    )

    BENCHMARK.main()

    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == BENCHMARK.RESULT_COLUMNS
    assert rows == [{name: str(value) for name, value in row.items()}]
