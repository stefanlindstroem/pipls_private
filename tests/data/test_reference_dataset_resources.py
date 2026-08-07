from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from importlib import resources
from typing import Any

import numpy as np

from pipls.datasets import PiPLSDataset, load_pulp, load_sugarcane, load_tobacco

_DATASET_LOADERS: dict[str, Callable[[], PiPLSDataset]] = {
    "pulp": load_pulp,
    "sugarcane": load_sugarcane,
    "tobacco": load_tobacco,
}


def _resource_root(dataset_id: str):
    return resources.files("pipls").joinpath("_data").joinpath(dataset_id)


def _metadata(dataset_id: str) -> dict[str, Any]:
    loaded = json.loads(
        _resource_root(dataset_id).joinpath("metadata.json").read_text(encoding="utf-8")
    )
    assert isinstance(loaded, dict)
    return loaded


def _canonical_array_hash(array: np.ndarray) -> str:
    canonical = np.asarray(array, dtype=np.dtype("<f8"), order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def test_reference_dataset_resources_match_declared_integrity() -> None:
    for dataset_id, loader in _DATASET_LOADERS.items():
        metadata = _metadata(dataset_id)
        resource_root = _resource_root(dataset_id)
        assert metadata["dataset"]["id"] == dataset_id

        resource_hashes = metadata["integrity"]["resource_sha256"]
        for name, expected_hash in resource_hashes.items():
            payload = resource_root.joinpath(name).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == expected_hash

        dataset = loader()
        array_hashes = metadata["integrity"]["array_sha256"]
        assert _canonical_array_hash(dataset.X) == array_hashes["data_float64_c_order"]
        assert _canonical_array_hash(dataset.Y) == array_hashes[
            "target_float64_c_order"
        ]
