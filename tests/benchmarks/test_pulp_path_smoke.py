from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_benchmark_module() -> ModuleType:
    path = _repository_root() / "benchmarks" / "pulp_path_smoke.py"
    spec = importlib.util.spec_from_file_location("pulp_path_smoke_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


def test_pulp_tables_are_read_directly_with_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def fake_read_csv(path: Any) -> pd.DataFrame:
        calls.append(Path(path))
        return pd.DataFrame({"value": [1.0, 2.0]})

    monkeypatch.setattr(BENCHMARK.pd, "read_csv", fake_read_csv)

    X, Y = BENCHMARK.read_pulp_data()

    assert calls == [BENCHMARK.DATA_DIR / "X.csv", BENCHMARK.DATA_DIR / "Y.csv"]
    assert list(X.columns) == ["value"]
    assert list(Y.columns) == ["value"]


def test_path_search_uses_public_defaults_and_exposes_conditional_rows() -> None:
    X, Y = BENCHMARK.read_pulp_data()

    with threadpool_limits(limits=1):
        search = BENCHMARK.fit_search(X, Y)

    expected_max_rank = min(
        X.shape[1],
        search.cv_n_train_min_ - 1,
        math.ceil(len(X) / search.samples_per_predictor_rank),
    )
    assert search.samples_per_predictor_rank == 5.0
    assert search.cv == 5
    assert search.max_predictor_rank == "rule"
    assert search.max_predictor_rank_ == expected_max_rank
    assert search.refit is False
    assert not hasattr(search, "best_estimator_")
    assert search.predictor_rank_policy_ == "optimized"

    path = search.component_path_results_
    np.testing.assert_array_equal(path["n_components"], BENCHMARK.N_COMPONENTS_VALUES)
    np.testing.assert_array_equal(path["n_splits"], np.full(4, 5, dtype=np.intp))
    assert path["predictor_rank_policy"].tolist() == ["optimized"] * 4

    for index, h in enumerate(BENCHMARK.N_COMPONENTS_VALUES):
        rank = int(path["predictor_rank"][index])
        assert rank == search.best_predictor_rank_by_n_components_[h]
        candidate = np.flatnonzero(
            (search.cv_results_["n_components"] == h)
            & (search.cv_results_["predictor_rank"] == rank)
        )
        assert candidate.size == 1
        cv_index = int(candidate[0])
        assert path["response_standardized_cv_mse_mean"][index] == pytest.approx(
            search.cv_results_["mean_response_standardized_mse"][cv_index]
        )
        assert path["response_standardized_cv_mse_fold_sd"][index] == pytest.approx(
            search.cv_results_["std_response_standardized_mse"][cv_index]
        )


def test_smoke_check_values_are_repeatable_finite_and_ordered() -> None:
    with threadpool_limits(limits=1):
        first = BENCHMARK.run_benchmark()
        second = BENCHMARK.run_benchmark()

    assert first == second
    assert len(first) == len(BENCHMARK.N_COMPONENTS_VALUES)
    assert [row["n_components"] for row in first] == list(BENCHMARK.N_COMPONENTS_VALUES)
    for row in first:
        assert tuple(row) == BENCHMARK.RESULT_COLUMNS
        assert isinstance(row["n_components"], int)
        assert isinstance(row["predictor_rank"], int)
        assert row["predictor_rank_policy"] == "optimized"
        assert 1 <= row["n_components"] <= row["predictor_rank"]
        assert np.isfinite(row["response_standardized_cv_mse_mean"])
        assert row["response_standardized_cv_mse_mean"] >= 0.0
        assert np.isfinite(row["response_standardized_cv_mse_fold_sd"])
        assert row["response_standardized_cv_mse_fold_sd"] >= 0.0
        assert row["n_splits"] == 5


def test_cli_writes_exact_component_path_csv_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "pulp_path_smoke.csv"
    rows = [
        {
            "n_components": 1,
            "predictor_rank": 3,
            "predictor_rank_policy": "optimized",
            "response_standardized_cv_mse_mean": 1.2,
            "response_standardized_cv_mse_fold_sd": 0.1,
            "n_splits": 5,
        },
        {
            "n_components": 2,
            "predictor_rank": 4,
            "predictor_rank_policy": "optimized",
            "response_standardized_cv_mse_mean": 0.8,
            "response_standardized_cv_mse_fold_sd": 0.2,
            "n_splits": 5,
        },
    ]
    monkeypatch.setattr(BENCHMARK, "run_benchmark", lambda: rows)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pulp_path_smoke.py", "--output", str(output)],
    )

    BENCHMARK.main()

    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = list(reader)

    assert tuple(reader.fieldnames or ()) == BENCHMARK.RESULT_COLUMNS
    assert written == [{name: str(value) for name, value in row.items()} for row in rows]
