from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks.run_synthetic import (
    _expand_scale_spec,
    load_manifest,
    run_benchmark,
    validate_result_record,
)

_RESOURCE_METRICS = {
    "fit_time_seconds",
    "predict_time_seconds",
    "peak_rss_bytes_optional",
}
_CAPTURE_METRICS = {
    "predictor_shared_capture",
    "predictor_signal_capture",
    "response_shared_capture",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ci_records() -> list[dict[str, Any]]:
    return run_benchmark(tier="ci")


def test_scale_specifications_expand_deterministically() -> None:
    constant = _expand_scale_spec({"kind": "constant", "value": 2.5}, length=4)
    logarithmic = _expand_scale_spec(
        {"kind": "logspace", "start": 0.1, "stop": 10.0},
        length=3,
    )

    np.testing.assert_array_equal(constant, np.full(4, 2.5))
    np.testing.assert_allclose(logarithmic, np.array([0.1, 1.0, 10.0]))


def test_ci_runner_produces_schema_valid_finite_records(
    ci_records: list[dict[str, Any]],
) -> None:
    root = _repository_root()
    manifest = load_manifest()
    tier = manifest["tiers"]["ci"]
    schema_path = root / manifest["results"]["schema"]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    expected_count = (
        len(tier["scenarios"])
        * len(manifest["seeds"][tier["seed_set"]])
        * len(tier["methods"])
    )

    assert len(ci_records) == expected_count
    assert all(record["status"] == "ok" for record in ci_records)
    assert len(
        {
            (record["scenario_id"], record["seed"], record["method_id"])
            for record in ci_records
        }
    ) == expected_count

    declared_metrics = {
        metric
        for group in manifest["metrics"].values()
        for metric in group
    }
    for record in ci_records:
        validate_result_record(record, schema)
        metrics = record["metrics"]
        assert set(metrics) == declared_metrics
        for name, value in metrics.items():
            if value is not None:
                assert np.isfinite(value), name
        for name in _CAPTURE_METRICS:
            assert 0.0 <= metrics[name] <= 1.0
        assert metrics["fit_time_seconds"] >= 0.0
        assert metrics["predict_time_seconds"] >= 0.0


def test_ci_cli_is_executable_and_numerically_repeatable(
    tmp_path: Path,
    ci_records: list[dict[str, Any]],
) -> None:
    root = _repository_root()
    output = tmp_path / "synthetic-ci.jsonl"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "run_synthetic.py"),
            "--tier",
            "ci",
            "--output",
            str(output),
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    repeated = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(repeated) == len(ci_records)

    for first, second in zip(ci_records, repeated, strict=True):
        assert first["schema_version"] == second["schema_version"]
        assert first["suite_id"] == second["suite_id"]
        assert first["tier"] == second["tier"]
        assert first["scenario_id"] == second["scenario_id"]
        assert first["seed"] == second["seed"]
        assert first["method_id"] == second["method_id"]
        assert first["status"] == second["status"] == "ok"
        assert first["parameters"] == second["parameters"]
        for name, first_value in first["metrics"].items():
            if name in _RESOURCE_METRICS:
                continue
            second_value = second["metrics"][name]
            if first_value is None:
                assert second_value is None
            else:
                assert second_value == pytest.approx(first_value, rel=1e-12, abs=1e-12)


def test_unimplemented_tiers_are_rejected() -> None:
    with pytest.raises(ValueError, match="Only the 'ci'.*tier is implemented"):
        run_benchmark(tier="standard")
