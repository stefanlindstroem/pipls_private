from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

_REQUIRED_GENERATOR_KEYS = {
    "n_train",
    "n_test",
    "n_features",
    "n_targets",
    "n_shared",
    "n_predictor_specific",
    "n_response_specific",
    "shared_strength",
    "predictor_specific_strength",
    "response_specific_strength",
    "shared_distribution",
    "predictor_specific_distribution",
    "response_specific_distribution",
    "feature_scale",
    "target_scale",
    "noise",
}
_ALLOWED_METHOD_KINDS = {"pipls_fixed", "pipls_path", "sklearn_pls"}
_ALLOWED_SCALE_KINDS = {"constant", "logspace"}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_manifest() -> dict[str, Any]:
    path = _repository_root() / "benchmarks" / "manifests" / "synthetic-v1.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _assert_positive_strength_spec(value: object, *, expected_length: int) -> None:
    if expected_length == 0:
        if isinstance(value, (int, float)):
            assert float(value) > 0.0
        else:
            assert value == []
        return
    if isinstance(value, (int, float)):
        assert float(value) > 0.0
        return
    assert isinstance(value, list)
    assert len(value) == expected_length
    assert all(isinstance(item, (int, float)) and float(item) > 0.0 for item in value)


def _assert_scale_spec(value: object) -> None:
    assert isinstance(value, dict)
    kind = value.get("kind")
    assert kind in _ALLOWED_SCALE_KINDS
    if kind == "constant":
        assert isinstance(value.get("value"), (int, float))
        assert float(value["value"]) > 0.0
    else:
        assert isinstance(value.get("start"), (int, float))
        assert isinstance(value.get("stop"), (int, float))
        assert float(value["start"]) > 0.0
        assert float(value["stop"]) > 0.0


def test_synthetic_benchmark_manifest_is_structurally_valid() -> None:
    manifest = _load_manifest()

    assert manifest["schema_version"] == 1
    assert isinstance(manifest["suite_id"], str) and manifest["suite_id"]
    assert manifest["generation"]["function"] == "pipls.datasets.make_pipls_train_test"
    result_schema = _repository_root() / manifest["results"]["schema"]
    assert result_schema.is_file()

    seed_sets = manifest["seeds"]
    assert isinstance(seed_sets, dict) and seed_sets
    for seeds in seed_sets.values():
        assert isinstance(seeds, list) and seeds
        assert len(seeds) == len(set(seeds))
        assert all(isinstance(seed, int) and seed >= 0 for seed in seeds)

    methods = manifest["methods"]
    method_ids = [method["id"] for method in methods]
    assert len(method_ids) == len(set(method_ids))
    assert {method["kind"] for method in methods} <= _ALLOWED_METHOD_KINDS

    scenarios = manifest["scenarios"]
    scenario_ids = [scenario["id"] for scenario in scenarios]
    assert len(scenario_ids) == len(set(scenario_ids))
    assert scenarios

    for scenario in scenarios:
        generator = scenario["generator"]
        assert set(generator) == _REQUIRED_GENERATOR_KEYS
        n_train = generator["n_train"]
        n_test = generator["n_test"]
        n_features = generator["n_features"]
        n_targets = generator["n_targets"]
        n_shared = generator["n_shared"]
        n_predictor_specific = generator["n_predictor_specific"]
        n_response_specific = generator["n_response_specific"]

        assert all(
            isinstance(value, int) and value > 0
            for value in (n_train, n_test, n_features, n_targets)
        )
        assert all(
            isinstance(value, int) and value >= 0
            for value in (n_shared, n_predictor_specific, n_response_specific)
        )
        assert n_shared > 0
        assert n_shared + n_predictor_specific <= n_features
        assert n_shared + n_response_specific <= n_targets
        assert n_train > max(
            n_shared + n_predictor_specific,
            n_shared + n_response_specific,
        )
        _assert_positive_strength_spec(
            generator["shared_strength"],
            expected_length=n_shared,
        )
        _assert_positive_strength_spec(
            generator["predictor_specific_strength"],
            expected_length=n_predictor_specific,
        )
        _assert_positive_strength_spec(
            generator["response_specific_strength"],
            expected_length=n_response_specific,
        )
        assert generator["shared_distribution"] in {"normal", "uniform"}
        assert generator["predictor_specific_distribution"] in {"normal", "uniform"}
        assert generator["response_specific_distribution"] in {"normal", "uniform"}
        _assert_scale_spec(generator["feature_scale"])
        _assert_scale_spec(generator["target_scale"])
        noise = generator["noise"]
        assert isinstance(noise, list) and len(noise) == 2
        assert all(isinstance(item, (int, float)) and float(item) >= 0.0 for item in noise)

    tiers = manifest["tiers"]
    assert isinstance(tiers, dict) and tiers
    known_scenarios = set(scenario_ids)
    known_methods = set(method_ids)
    for tier in tiers.values():
        assert tier["scenarios"]
        assert tier["methods"]
        assert set(tier["scenarios"]) <= known_scenarios
        assert set(tier["methods"]) <= known_methods
        assert tier["seed_set"] in seed_sets
        assert tier["timing_gate"] is False
        assert isinstance(tier["expected_runtime_seconds"], int)
        assert tier["expected_runtime_seconds"] > 0


def test_result_schema_defines_a_flat_ordered_csv_contract() -> None:
    root = _repository_root()
    manifest = _load_manifest()
    results = manifest["results"]
    schema_path = root / results["schema"]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert results["format"] == "csv"
    assert results["schema_version"] == 2
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == results["schema_version"]
    assert schema["x-csv"]["encoding"] == "utf-8"
    assert schema["x-csv"]["delimiter"] == ","
    assert schema["x-csv"]["header"] is True
    assert schema["x-csv"]["null"] == ""

    columns = schema["x-csv"]["columns"]
    assert columns[:7] == [
        "schema_version",
        "suite_id",
        "tier",
        "scenario_id",
        "seed",
        "method_id",
        "status",
    ]
    assert len(columns) == len(set(columns))
    assert set(columns) == set(schema["properties"]) == set(schema["required"])
    assert all(
        property_schema.get("type") != "object" for property_schema in schema["properties"].values()
    )


def test_generated_benchmark_results_are_ignored() -> None:
    ignored = (_repository_root() / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "benchmarks/results/" in ignored
