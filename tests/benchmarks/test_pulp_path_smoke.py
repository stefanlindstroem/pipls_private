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


def test_path_search_uses_public_selection_and_complete_ordered_oof_output() -> None:
    X, Y = BENCHMARK.read_pulp_data()

    with threadpool_limits(limits=1):
        search = BENCHMARK.fit_search(X, Y)

    report = search.validation_report_
    expected_max_rank = min(
        X.shape[1],
        search.cv_n_train_min_,
        math.ceil(search.cv_n_train_min_ / search.samples_per_predictor_rank),
    )
    assert search.samples_per_predictor_rank == 5.0
    assert search.cv == 5
    assert search.max_predictor_rank == "rule"
    assert search.max_predictor_rank_ == expected_max_rank
    assert search.best_n_components_ == search.best_params_["n_components"]
    assert search.best_predictor_rank_ == search.best_params_["predictor_rank"]
    assert search.best_n_components_ == search.best_pipls_.n_components
    assert search.best_predictor_rank_ == search.best_pipls_.predictor_rank
    assert report.n_components == search.best_n_components_
    assert report.predictor_rank == search.best_predictor_rank_
    assert report.estimate_kind == "selection-conditioned"
    assert report.complete_oof_coverage
    assert report.oof_predictions is not None
    assert report.oof_prediction_counts is not None
    assert report.oof_predictions.shape == Y.shape
    assert np.array_equal(report.oof_prediction_counts, np.ones(len(X), dtype=np.intp))


def test_smoke_check_values_are_repeatable_finite_and_explicitly_conditioned() -> None:
    with threadpool_limits(limits=1):
        first = BENCHMARK.run_benchmark()
        second = BENCHMARK.run_benchmark()

    assert first == second
    assert len(first) == 1
    row = first[0]
    assert tuple(row) == BENCHMARK.RESULT_COLUMNS
    assert isinstance(row["selected_n_components"], int)
    assert isinstance(row["selected_predictor_rank"], int)
    assert 1 <= row["selected_n_components"] <= row["selected_predictor_rank"]
    assert np.isfinite(row["selection_conditioned_response_standardized_mse"])
    assert row["selection_conditioned_response_standardized_mse"] >= 0.0
    assert np.isfinite(row["selection_conditioned_pooled_oof_r2"])


def test_cli_writes_exact_minimal_csv_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "pulp_path_smoke.csv"
    row = {
        "selected_n_components": 2,
        "selected_predictor_rank": 3,
        "selection_conditioned_response_standardized_mse": 1.2,
        "selection_conditioned_pooled_oof_r2": 0.1,
    }
    monkeypatch.setattr(BENCHMARK, "run_benchmark", lambda: [row])
    monkeypatch.setattr(
        sys,
        "argv",
        ["pulp_path_smoke.py", "--output", str(output)],
    )

    BENCHMARK.main()

    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == BENCHMARK.RESULT_COLUMNS
    assert rows == [{name: str(value) for name, value in row.items()}]
