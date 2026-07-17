from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from numpy.testing import assert_array_equal


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_benchmark_module() -> ModuleType:
    path = _repository_root() / "benchmarks" / "fixed_structure_recovery.py"
    spec = importlib.util.spec_from_file_location("fixed_structure_recovery_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


def test_benchmark_generation_is_deterministic() -> None:
    scenario = BENCHMARK.SCENARIOS[1]
    first_train, first_test = BENCHMARK.generate_problem(scenario, BENCHMARK.SEEDS[0])
    second_train, second_test = BENCHMARK.generate_problem(scenario, BENCHMARK.SEEDS[0])

    assert_array_equal(first_train.X, second_train.X)
    assert_array_equal(first_train.Y, second_train.Y)
    assert_array_equal(first_test.X, second_test.X)
    assert_array_equal(first_test.Y, second_test.Y)


def test_benchmark_scientific_values_are_repeatable_and_bounded() -> None:
    first = BENCHMARK.run_benchmark()
    second = BENCHMARK.run_benchmark()

    assert first == second
    assert len(first) == len(BENCHMARK.SCENARIOS) * len(BENCHMARK.SEEDS)
    assert [row["scenario"] for row in first] == [
        scenario.name for scenario in BENCHMARK.SCENARIOS for _ in BENCHMARK.SEEDS
    ]

    for row in first:
        assert set(row) == set(BENCHMARK.RESULT_COLUMNS)
        assert np.isfinite(row["test_mse"])
        assert row["test_mse"] >= 0.0
        for name in (
            "predictor_shared_capture",
            "predictor_signal_capture",
            "response_shared_capture",
        ):
            assert np.isfinite(row[name])
            assert 0.0 <= row[name] <= 1.0


def test_cli_writes_exact_minimal_csv_header(tmp_path: Path) -> None:
    root = _repository_root()
    output = tmp_path / "fixed_structure_recovery.csv"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")

    subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "fixed_structure_recovery.py"),
            "--output",
            str(output),
        ],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == BENCHMARK.RESULT_COLUMNS
    assert len(rows) == len(BENCHMARK.SCENARIOS) * len(BENCHMARK.SEEDS)
