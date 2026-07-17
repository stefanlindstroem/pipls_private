"""Execute the manifest-defined Pi-PLS synthetic CI benchmark tier."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import yaml
from joblib import parallel_config
from numpy.typing import NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

import pipls
from pipls import PiPLSPathCV, PiPLSRegression
from pipls.datasets import PiPLSDataset, PiPLSSyntheticTruth, make_pipls_train_test

FloatArray = NDArray[np.float64]
JsonObject = dict[str, Any]

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFEST = _REPOSITORY_ROOT / "benchmarks" / "manifests" / "synthetic-v1.yaml"
_IMPLEMENTED_TIER = "ci"


def load_manifest(path: Path = _DEFAULT_MANIFEST) -> JsonObject:
    """Load the versioned synthetic benchmark manifest."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("The synthetic benchmark manifest must contain a YAML mapping.")
    return data


def run_benchmark(
    *,
    tier: str = _IMPLEMENTED_TIER,
    manifest_path: Path = _DEFAULT_MANIFEST,
) -> list[JsonObject]:
    """Run the implemented manifest tier and return schema-valid result records."""

    if tier != _IMPLEMENTED_TIER:
        raise ValueError(
            f"Only the {_IMPLEMENTED_TIER!r} synthetic benchmark tier is implemented; "
            f"got {tier!r}."
        )

    manifest = load_manifest(manifest_path)
    tier_config = _mapping(manifest["tiers"], name="tiers").get(tier)
    if not isinstance(tier_config, dict):
        raise ValueError(f"The manifest does not define tier {tier!r}.")

    scenarios = {
        _string(item["id"], name="scenario id"): _mapping(item, name="scenario")
        for item in _sequence(manifest["scenarios"], name="scenarios")
    }
    methods = {
        _string(item["id"], name="method id"): _mapping(item, name="method")
        for item in _sequence(manifest["methods"], name="methods")
    }
    seed_set = _string(tier_config["seed_set"], name="seed_set")
    seeds = _sequence(_mapping(manifest["seeds"], name="seeds")[seed_set], name="seeds")
    schema_path = _resolve_repository_path(
        _string(_mapping(manifest["results"], name="results")["schema"], name="schema")
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    records: list[JsonObject] = []
    for scenario_id in _string_sequence(tier_config["scenarios"], name="tier scenarios"):
        scenario = scenarios[scenario_id]
        for seed_value in seeds:
            seed = _integer(seed_value, name="benchmark seed")
            train, test = _generate_problem(scenario, seed=seed)
            for method_id in _string_sequence(tier_config["methods"], name="tier methods"):
                method = methods[method_id]
                record = _run_method(
                    manifest=manifest,
                    tier=tier,
                    scenario=scenario,
                    seed=seed,
                    method=method,
                    train=train,
                    test=test,
                )
                validate_result_record(record, schema)
                records.append(record)
    return records


def write_jsonl(records: Sequence[Mapping[str, Any]], output: Path | None) -> None:
    """Write records as deterministic JSON Lines to a file or standard output."""

    text = "".join(
        json.dumps(dict(record), sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
        for record in records
    )
    if output is None:
        sys.stdout.write(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def validate_result_record(record: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate a result against the deliberately small version-1 JSON schema."""

    _validate_schema_value(record, schema, path="record")


def _run_method(
    *,
    manifest: Mapping[str, Any],
    tier: str,
    scenario: Mapping[str, Any],
    seed: int,
    method: Mapping[str, Any],
    train: PiPLSDataset,
    test: PiPLSDataset,
) -> JsonObject:
    scenario_id = _string(scenario["id"], name="scenario id")
    method_id = _string(method["id"], name="method id")
    parameters = _resolve_method_parameters(method, scenario=scenario, seed=seed, manifest=manifest)
    base = _base_record(
        manifest=manifest,
        tier=tier,
        scenario_id=scenario_id,
        seed=seed,
        method_id=method_id,
        parameters=parameters,
    )

    try:
        estimator = _make_estimator(method, parameters=parameters, seed=seed, manifest=manifest)
        fit_started = time.perf_counter()
        if isinstance(estimator, PiPLSPathCV):
            with threadpool_limits(limits=1), parallel_config(
                backend="threading", n_jobs=-1
            ):
                estimator.fit(train.X, train.Y)
        else:
            with threadpool_limits(limits=1):
                estimator.fit(train.X, train.Y)
        fit_time = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        prediction = np.asarray(estimator.predict(test.X), dtype=np.float64)
        predict_time = time.perf_counter() - predict_started
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)

        fitted_pipls = _fitted_pipls(estimator)
        metrics = _benchmark_metrics(
            method=method,
            estimator=estimator,
            fitted_pipls=fitted_pipls,
            train=train,
            test=test,
            prediction=prediction,
            fit_time=fit_time,
            predict_time=predict_time,
        )
        return {**base, "status": "ok", "metrics": metrics, "message": None}
    except Exception as error:  # pragma: no cover - defensive result capture
        return {
            **base,
            "status": "failed",
            "metrics": {},
            "message": f"{type(error).__name__}: {error}",
        }


def _generate_problem(
    scenario: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[PiPLSDataset, PiPLSDataset]:
    generator = dict(_mapping(scenario["generator"], name="scenario generator"))
    n_features = _integer(generator["n_features"], name="n_features")
    n_targets = _integer(generator["n_targets"], name="n_targets")
    generator["feature_scale"] = _expand_scale_spec(
        _mapping(generator["feature_scale"], name="feature_scale"),
        length=n_features,
    )
    generator["target_scale"] = _expand_scale_spec(
        _mapping(generator["target_scale"], name="target_scale"),
        length=n_targets,
    )
    noise = _sequence(generator["noise"], name="noise")
    if len(noise) != 2:
        raise ValueError("The benchmark noise specification must contain two values.")
    generator["noise"] = (float(noise[0]), float(noise[1]))
    generator["random_state"] = seed
    return make_pipls_train_test(**generator)


def _expand_scale_spec(spec: Mapping[str, Any], *, length: int) -> FloatArray:
    kind = _string(spec["kind"], name="scale kind")
    if kind == "constant":
        value = float(spec["value"])
        return np.full(length, value, dtype=np.float64)
    if kind == "logspace":
        start = float(spec["start"])
        stop = float(spec["stop"])
        return np.geomspace(start, stop, num=length, dtype=np.float64)
    raise ValueError(f"Unsupported scale specification kind: {kind!r}.")


def _resolve_method_parameters(
    method: Mapping[str, Any],
    *,
    scenario: Mapping[str, Any],
    seed: int,
    manifest: Mapping[str, Any],
) -> JsonObject:
    generator = _mapping(scenario["generator"], name="scenario generator")
    truth = {
        "n_shared": _integer(generator["n_shared"], name="n_shared"),
        "n_shared_plus_predictor_specific": _integer(generator["n_shared"], name="n_shared")
        + _integer(generator["n_predictor_specific"], name="n_predictor_specific"),
    }
    resolved: JsonObject = {}
    for key, value in _mapping(method["parameters"], name="method parameters").items():
        if value == "truth.n_shared":
            resolved[key] = truth["n_shared"]
        elif value == "truth.n_shared_plus_predictor_specific":
            resolved[key] = truth["n_shared_plus_predictor_specific"]
        elif value == "benchmark_seed":
            resolved[key] = seed
        else:
            resolved[key] = value

    if _string(method["kind"], name="method kind") == "pipls_path":
        cv = _mapping(manifest["cross_validation"], name="cross_validation")
        resolved["cv"] = {
            "splitter": _string(cv["splitter"], name="splitter"),
            "n_splits": _integer(cv["n_splits"], name="n_splits"),
            "shuffle": bool(cv["shuffle"]),
            "random_state": seed,
        }
        resolved["scoring"] = _string(cv["scoring"], name="scoring")
        resolved["n_jobs"] = -1
        resolved["parallel_backend"] = "threading"
        resolved["native_threads_per_worker"] = 1
    return resolved


def _make_estimator(
    method: Mapping[str, Any],
    *,
    parameters: Mapping[str, Any],
    seed: int,
    manifest: Mapping[str, Any],
) -> Any:
    kind = _string(method["kind"], name="method kind")
    if kind == "pipls_fixed":
        return PiPLSRegression(**dict(parameters))
    if kind == "sklearn_pls":
        return PLSRegression(**dict(parameters))
    if kind == "pipls_path":
        cv = _mapping(manifest["cross_validation"], name="cross_validation")
        base = PiPLSRegression(
            scale=bool(parameters["scale"]),
            svd_solver=_string(parameters["svd_solver"], name="svd_solver"),
            random_state=seed,
        )
        return PiPLSPathCV(
            estimator=base,
            max_predictor_rank=_string(parameters["max_predictor_rank"], name="max rank"),
            search_method=_string(parameters["search_method"], name="search method"),
            samples_per_predictor_rank=float(parameters["samples_per_predictor_rank"]),
            cv=KFold(
                n_splits=_integer(cv["n_splits"], name="n_splits"),
                shuffle=bool(cv["shuffle"]),
                random_state=seed,
            ),
        )
    raise ValueError(f"Unsupported benchmark method kind: {kind!r}.")


def _benchmark_metrics(
    *,
    method: Mapping[str, Any],
    estimator: Any,
    fitted_pipls: PiPLSRegression | None,
    train: PiPLSDataset,
    test: PiPLSDataset,
    prediction: FloatArray,
    fit_time: float,
    predict_time: float,
) -> JsonObject:
    truth = _truth(train)
    response_scale = _safe_sample_scale(train.Y)
    standardized_residual = (test.Y - prediction) / response_scale[None, :]
    n_shared = truth.n_shared
    predictor_signal_rank = truth.n_shared + truth.n_predictor_specific

    selected_n_components: int | None
    selected_predictor_rank: int | None
    if isinstance(estimator, PiPLSPathCV):
        selected_n_components = int(estimator.best_n_components_)
        selected_predictor_rank = int(estimator.best_predictor_rank_)
    elif fitted_pipls is not None:
        selected_n_components = int(fitted_pipls.n_components)
        selected_predictor_rank = int(fitted_pipls.predictor_rank_)
    else:
        selected_n_components = int(estimator.n_components)
        selected_predictor_rank = None

    x_fitted_shared, x_fitted_signal, y_fitted_shared = _fitted_subspaces(
        estimator=estimator,
        fitted_pipls=fitted_pipls,
    )
    x_scale = fitted_pipls.x_scale_ if fitted_pipls is not None else _safe_sample_scale(train.X)
    y_scale = fitted_pipls.y_scale_ if fitted_pipls is not None else response_scale
    x_shared_truth = _scaled_truth_basis(
        truth.x_shared_loadings,
        observed_scale=truth.feature_scale,
        fitted_scale=x_scale,
    )
    x_signal_truth = _scaled_truth_basis(
        np.column_stack(
            (truth.x_shared_loadings, truth.x_predictor_specific_loadings)
        ),
        observed_scale=truth.feature_scale,
        fitted_scale=x_scale,
    )
    y_shared_truth = _scaled_truth_basis(
        truth.y_shared_loadings,
        observed_scale=truth.target_scale,
        fitted_scale=y_scale,
    )

    return {
        "response_standardized_mse": float(np.mean(np.square(standardized_residual))),
        "r2_uniform_average": float(
            r2_score(test.Y, prediction, multioutput="uniform_average")
        ),
        "selected_n_components": selected_n_components,
        "selected_predictor_rank": selected_predictor_rank,
        "n_components_absolute_error_from_declared_shared_rank": abs(
            selected_n_components - n_shared
        ),
        "predictor_rank_absolute_error_from_declared_signal_rank": (
            None
            if selected_predictor_rank is None
            else abs(selected_predictor_rank - predictor_signal_rank)
        ),
        "predictor_shared_capture": _subspace_capture(x_shared_truth, x_fitted_shared),
        "predictor_signal_capture": _subspace_capture(x_signal_truth, x_fitted_signal),
        "response_shared_capture": _subspace_capture(y_shared_truth, y_fitted_shared),
        "prediction_relative_difference": None,
        "coefficient_relative_difference": None,
        "fit_time_seconds": float(fit_time),
        "predict_time_seconds": float(predict_time),
        "peak_rss_bytes_optional": None,
    }


def _fitted_subspaces(
    *, estimator: Any, fitted_pipls: PiPLSRegression | None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if fitted_pipls is not None:
        return (
            np.asarray(fitted_pipls.decomposition_.P, dtype=np.float64),
            np.asarray(fitted_pipls.decomposition_.Pi, dtype=np.float64),
            np.asarray(fitted_pipls.decomposition_.Q, dtype=np.float64),
        )
    return (
        np.asarray(estimator.x_rotations_, dtype=np.float64),
        np.asarray(estimator.x_rotations_, dtype=np.float64),
        np.asarray(estimator.y_rotations_, dtype=np.float64),
    )


def _scaled_truth_basis(
    loadings: FloatArray,
    *,
    observed_scale: FloatArray,
    fitted_scale: FloatArray,
) -> FloatArray:
    transformed = (observed_scale / fitted_scale)[:, None] * loadings
    return _orthonormal_basis(transformed)


def _orthonormal_basis(matrix: FloatArray) -> FloatArray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape[1] == 0:
        return np.empty((array.shape[0], 0), dtype=np.float64)
    left, singular_values, _ = np.linalg.svd(array, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(array.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    return np.asarray(left[:, :rank], dtype=np.float64)


def _subspace_capture(truth_basis: FloatArray, fitted_basis: FloatArray) -> float:
    truth_orthonormal = _orthonormal_basis(truth_basis)
    fitted_orthonormal = _orthonormal_basis(fitted_basis)
    if truth_orthonormal.shape[1] == 0:
        return 1.0
    overlap = truth_orthonormal.T @ fitted_orthonormal
    value = float(np.sum(np.square(overlap)) / truth_orthonormal.shape[1])
    return float(np.clip(value, 0.0, 1.0))


def _safe_sample_scale(matrix: FloatArray) -> FloatArray:
    scale = np.std(np.asarray(matrix, dtype=np.float64), axis=0, ddof=1)
    return np.where(scale == 0.0, 1.0, scale).astype(np.float64, copy=False)


def _truth(dataset: PiPLSDataset) -> PiPLSSyntheticTruth:
    if dataset.truth is None:
        raise ValueError("Synthetic benchmark datasets must expose latent truth.")
    return dataset.truth


def _fitted_pipls(estimator: Any) -> PiPLSRegression | None:
    if isinstance(estimator, PiPLSRegression):
        return estimator
    if isinstance(estimator, PiPLSPathCV):
        return estimator.best_pipls_
    return None


def _base_record(
    *,
    manifest: Mapping[str, Any],
    tier: str,
    scenario_id: str,
    seed: int,
    method_id: str,
    parameters: Mapping[str, Any],
) -> JsonObject:
    return {
        "schema_version": _integer(manifest["schema_version"], name="schema_version"),
        "suite_id": _string(manifest["suite_id"], name="suite_id"),
        "tier": tier,
        "scenario_id": scenario_id,
        "seed": seed,
        "method_id": method_id,
        "versions": {
            "python": platform.python_version(),
            "pipls": pipls.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "parameters": dict(parameters),
    }


def _validate_schema_value(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    expected = schema.get("type")
    if expected is not None and not _matches_json_type(value, expected):
        raise ValueError(f"{path} does not match JSON schema type {expected!r}.")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} must equal {schema['const']!r}.")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']!r}.")
    if isinstance(value, str) and "minLength" in schema and len(value) < int(schema["minLength"]):
        raise ValueError(f"{path} is shorter than the schema minimum.")
    if isinstance(value, int) and not isinstance(value, bool) and "minimum" in schema:
        if value < int(schema["minimum"]):
            raise ValueError(f"{path} is below the schema minimum.")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        missing = required - set(value)
        if missing:
            raise ValueError(f"{path} is missing required fields: {sorted(missing)!r}.")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise ValueError(f"{path} contains unsupported fields: {sorted(extra)!r}.")
        for key, item in value.items():
            child_schema = properties.get(key)
            if child_schema is None:
                additional = schema.get("additionalProperties")
                if isinstance(additional, Mapping):
                    child_schema = additional
            if isinstance(child_schema, Mapping):
                _validate_schema_value(item, child_schema, path=f"{path}.{key}")


def _matches_json_type(value: Any, expected: Any) -> bool:
    expected_types = expected if isinstance(expected, list) else [expected]
    return any(_matches_one_json_type(value, item) for item in expected_types)


def _matches_one_json_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(value)
        )
    return False


def _resolve_repository_path(value: str) -> Path:
    path = (_REPOSITORY_ROOT / value).resolve()
    if not path.is_relative_to(_REPOSITORY_ROOT):
        raise ValueError("Benchmark manifest paths must stay within the repository.")
    return path


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list.")
    return value


def _string_sequence(value: Any, *, name: str) -> list[str]:
    return [_string(item, name=name) for item in _sequence(value, name=name)]


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default=_IMPLEMENTED_TIER, help="Benchmark tier (currently: ci).")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Path to the versioned synthetic benchmark manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON Lines to this path instead of standard output.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line benchmark entry point."""

    args = _parse_args(argv)
    try:
        records = run_benchmark(tier=args.tier, manifest_path=args.manifest)
        write_jsonl(records, args.output)
    except (OSError, ValueError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(f"benchmark error: {error}", file=sys.stderr)
        return 2
    return 1 if any(record["status"] != "ok" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
