from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from pipls.datasets import PiPLSDataset, load_pulp, load_sugarcane, load_tobacco

_DATASET_LOADERS: dict[str, Callable[[], PiPLSDataset]] = {
    "pulp": load_pulp,
    "sugarcane": load_sugarcane,
    "tobacco": load_tobacco,
}
_RESOURCE_FILES = {"X.csv", "Y.csv", "metadata.json", "README.md", "LICENSE.txt"}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _metadata(dataset_id: str) -> dict[str, Any]:
    path = _repository_root() / "src" / "pipls" / "_data" / dataset_id / "metadata.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _canonical_array_hash(array: np.ndarray) -> str:
    canonical = np.asarray(array, dtype=np.dtype("<f8"), order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _collect_named_strings(value: object, name: str) -> set[str]:
    if isinstance(value, dict):
        collected: set[str] = set()
        for key, child in value.items():
            if key == name and isinstance(child, str) and child.strip():
                collected.add(child.strip())
            collected.update(_collect_named_strings(child, name))
        return collected
    if isinstance(value, list):
        collected: set[str] = set()
        for child in value:
            collected.update(_collect_named_strings(child, name))
        return collected
    return set()


def test_reference_dataset_resource_directories_are_complete() -> None:
    root = _repository_root()
    resource_root = root / "src" / "pipls" / "_data"

    assert not (root / "datasets").exists()
    assert sorted(path.name for path in resource_root.iterdir() if path.is_dir()) == sorted(
        _DATASET_LOADERS
    )

    for dataset_id in _DATASET_LOADERS:
        files = {
            path.name
            for path in (resource_root / dataset_id).iterdir()
            if path.is_file()
        }
        assert files == _RESOURCE_FILES, dataset_id


def test_reference_dataset_resources_match_declared_integrity() -> None:
    resource_root = _repository_root() / "src" / "pipls" / "_data"

    for dataset_id, loader in _DATASET_LOADERS.items():
        metadata = _metadata(dataset_id)
        assert metadata["dataset"]["id"] == dataset_id

        resource_hashes = metadata["integrity"]["resource_sha256"]
        for name, expected_hash in resource_hashes.items():
            payload = (resource_root / dataset_id / name).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == expected_hash

        dataset = loader()
        array_hashes = metadata["integrity"]["array_sha256"]
        assert _canonical_array_hash(dataset.X) == array_hashes["data_float64_c_order"]
        assert _canonical_array_hash(dataset.Y) == array_hashes[
            "target_float64_c_order"
        ]


def test_reference_dataset_matrices_have_one_active_location() -> None:
    root = _repository_root()
    active_roots = (root / "src", root / "datasets")

    for dataset_id in _DATASET_LOADERS:
        matrices = sorted(
            path.relative_to(root).as_posix()
            for active_root in active_roots
            if active_root.exists()
            for path in active_root.rglob("*.csv")
            if dataset_id in path.parts and path.name in {"X.csv", "Y.csv"}
        )
        assert matrices == [
            f"src/pipls/_data/{dataset_id}/X.csv",
            f"src/pipls/_data/{dataset_id}/Y.csv",
        ]


def test_public_dataset_guide_documents_sources_and_raw_resources() -> None:
    guide = (_repository_root() / "docs" / "datasets.md").read_text(encoding="utf-8")

    for dataset_id in _DATASET_LOADERS:
        metadata = _metadata(dataset_id)
        source = metadata["source"]
        dois = _collect_named_strings(source, "doi")
        urls = _collect_named_strings(source, "url")
        assert dois, dataset_id

        for doi in dois:
            url = f"https://doi.org/{doi}"
            assert url in urls, (dataset_id, doi)
            assert f"]({url})" in guide, (dataset_id, doi)

        for name in sorted(_RESOURCE_FILES):
            assert f"src/pipls/_data/{dataset_id}/{name}" in guide
        assert f"pipls/_data/{dataset_id}/" in guide
