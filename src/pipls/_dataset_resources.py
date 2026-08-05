"""Packaged reference-dataset loading and integrity checks."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from typing import Literal, cast, overload

import numpy as np

from ._dataset_types import (
    _REQUIRED_PROVENANCE_KEYS,
    FloatArray,
    PiPLSDataset,
    _validated_matrix,
)


@dataclass(frozen=True)
class _PackagedDatasetConfig:
    dataset_id: str
    display_name: str
    sample_id_width: int


_PULP_DATASET = _PackagedDatasetConfig(
    dataset_id="pulp",
    display_name="Pulp",
    sample_id_width=2,
)
_SUGARCANE_DATASET = _PackagedDatasetConfig(
    dataset_id="sugarcane",
    display_name="Sugarcane",
    sample_id_width=2,
)
_TOBACCO_DATASET = _PackagedDatasetConfig(
    dataset_id="tobacco",
    display_name="Tobacco",
    sample_id_width=3,
)


@overload
def load_pulp(*, return_X_y: Literal[False] = False) -> PiPLSDataset:
    ...


@overload
def load_pulp(*, return_X_y: Literal[True]) -> tuple[FloatArray, FloatArray]:
    ...


def load_pulp(
    *,
    return_X_y: bool = False,
) -> PiPLSDataset | tuple[FloatArray, FloatArray]:
    """Load the packaged Pulp fiber-property regression dataset.

    The dataset contains 46 thermomechanical-pulp samples, 14 fiber-property
    predictors, and eight pulp or handsheet responses. The analysis-facing
    matrices preserve the numeric values, column order, and row order selected
    from the publication supplementary material. No preprocessing is applied.

    Parameters
    ----------
    return_X_y : bool, default=False
        If ``True``, return the read-only predictor and response arrays directly.
        Otherwise return an immutable :class:`PiPLSDataset` with labels,
        provenance, sample identifiers, and metadata.

    Returns
    -------
    PiPLSDataset or tuple of ndarray
        Structured dataset by default, or ``(X, Y)`` when ``return_X_y=True``.

    Notes
    -----
    The dataset is adapted from supplementary material for Lindström et al.
    (2025), *Computers & Chemical Engineering*, 199, 109143,
    doi:10.1016/j.compchemeng.2025.109143, under CC BY 4.0.
    """

    return _load_packaged_dataset(
        _PULP_DATASET,
        return_X_y=return_X_y,
    )


@overload
def load_sugarcane(*, return_X_y: Literal[False] = False) -> PiPLSDataset:
    ...


@overload
def load_sugarcane(*, return_X_y: Literal[True]) -> tuple[FloatArray, FloatArray]:
    ...


def load_sugarcane(
    *,
    return_X_y: bool = False,
) -> PiPLSDataset | tuple[FloatArray, FloatArray]:
    """Load the packaged Sugarcane LabSpec regression dataset.

    The dataset contains 57 sugarcane samples, 1,721 visible-near-infrared
    absorbance predictors at integer wavelengths from 780 through 2500 nm,
    and four chemical or feed-quality responses. The matrices preserve the
    retained numeric values, column order, and row order derived from the
    public source tables. No preprocessing is applied during loading.

    Parameters
    ----------
    return_X_y : bool, default=False
        If ``True``, return the read-only predictor and response arrays directly.
        Otherwise return an immutable :class:`PiPLSDataset` with labels,
        provenance, sample identifiers, and metadata.

    Returns
    -------
    PiPLSDataset or tuple of ndarray
        Structured dataset by default, or ``(X, Y)`` when ``return_X_y=True``.

    Notes
    -----
    The dataset is adapted from Chaix, Bendoula, and Zgouz (2021), Mendeley
    Data, Version 1, doi:10.17632/mjttsjfj2s.1, under CC BY 4.0. Source samples
    103, 105, and 111 are excluded because total sugar is missing.
    """

    return _load_packaged_dataset(
        _SUGARCANE_DATASET,
        return_X_y=return_X_y,
    )


@overload
def load_tobacco(*, return_X_y: Literal[False] = False) -> PiPLSDataset:
    ...


@overload
def load_tobacco(*, return_X_y: Literal[True]) -> tuple[FloatArray, FloatArray]:
    ...


def load_tobacco(
    *,
    return_X_y: bool = False,
) -> PiPLSDataset | tuple[FloatArray, FloatArray]:
    """Load the packaged Tobacco leaf FT-NIR regression dataset.

    The dataset contains 347 tobacco leaf samples, 1,557 raw FT-NIR
    absorbance predictors ordered from approximately 10,001 down to
    4,000 cm^-1, and 13 chemical-component responses. The matrices
    preserve the retained numeric values, column order, and row order
    derived from the public source workbooks. No preprocessing is applied.

    Parameters
    ----------
    return_X_y : bool, default=False
        If ``True``, return the read-only predictor and response arrays directly.
        Otherwise return an immutable :class:`PiPLSDataset` with labels,
        provenance, sample identifiers, and metadata.

    Returns
    -------
    PiPLSDataset or tuple of ndarray
        Structured dataset by default, or ``(X, Y)`` when ``return_X_y=True``.

    Notes
    -----
    The dataset is adapted from Chen, Guo, Wang, and Zhao (2025),
    Mendeley Data, Version 1, doi:10.17632/9z7dgdtggk.1, under CC BY 4.0.
    """

    return _load_packaged_dataset(
        _TOBACCO_DATASET,
        return_X_y=return_X_y,
    )


def _load_packaged_dataset(
    config: _PackagedDatasetConfig,
    *,
    return_X_y: bool,
) -> PiPLSDataset | tuple[FloatArray, FloatArray]:
    if not isinstance(return_X_y, bool):
        raise TypeError("return_X_y must be a boolean.")

    metadata = _load_dataset_metadata(config)
    feature_names = _metadata_string_tuple(config, metadata, "feature_names")
    target_names = _metadata_string_tuple(config, metadata, "target_names")
    dimensions = _metadata_mapping(config, metadata, "dimensions")
    integrity = _metadata_mapping(config, metadata, "integrity")
    resource_hashes = _metadata_mapping(config, integrity, "resource_sha256")
    array_hashes = _metadata_mapping(config, integrity, "array_sha256")

    X = _load_dataset_csv(
        config,
        "X.csv",
        expected_names=feature_names,
        expected_resource_hash=_metadata_string(config, resource_hashes, "X.csv"),
    )
    Y = _load_dataset_csv(
        config,
        "Y.csv",
        expected_names=target_names,
        expected_resource_hash=_metadata_string(config, resource_hashes, "Y.csv"),
    )

    expected_shape = (
        _metadata_integer(config, dimensions, "n_samples"),
        _metadata_integer(config, dimensions, "n_features"),
        _metadata_integer(config, dimensions, "n_targets"),
    )
    if X.shape != expected_shape[:2] or Y.shape != (
        expected_shape[0],
        expected_shape[2],
    ):
        raise RuntimeError(
            f"Packaged {config.display_name} matrices do not match metadata dimensions."
        )

    _verify_dataset_array_hash(
        config,
        X,
        expected=_metadata_string(config, array_hashes, "data_float64_c_order"),
        name="data",
    )
    _verify_dataset_array_hash(
        config,
        Y,
        expected=_metadata_string(config, array_hashes, "target_float64_c_order"),
        name="target",
    )

    provenance_raw = _metadata_mapping(config, metadata, "provenance")
    provenance = {
        key: _metadata_string(config, provenance_raw, key)
        for key in _REQUIRED_PROVENANCE_KEYS
    }
    dataset = PiPLSDataset(
        X=X,
        Y=Y,
        feature_names=feature_names,
        target_names=target_names,
        sample_ids=tuple(
            f"{config.dataset_id}-{index:0{config.sample_id_width}d}"
            for index in range(1, X.shape[0] + 1)
        ),
        provenance=provenance,
        metadata=metadata,
    )
    if return_X_y:
        return dataset.X, dataset.Y
    return dataset


def _dataset_resource_bytes(config: _PackagedDatasetConfig, name: str) -> bytes:
    resource = (
        resources.files("pipls")
        .joinpath("_data")
        .joinpath(config.dataset_id)
        .joinpath(name)
    )
    try:
        return resource.read_bytes()
    except (FileNotFoundError, OSError) as error:
        message = f"Packaged {config.display_name} resource {name!r} is unavailable."
        raise RuntimeError(message) from error


def _load_dataset_metadata(config: _PackagedDatasetConfig) -> Mapping[str, object]:
    try:
        loaded = json.loads(
            _dataset_resource_bytes(config, "metadata.json").decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Packaged {config.display_name} metadata is invalid.") from error
    if not isinstance(loaded, dict):
        raise RuntimeError(
            f"Packaged {config.display_name} metadata must be a JSON object."
        )
    return cast(Mapping[str, object], loaded)


def _load_dataset_csv(
    config: _PackagedDatasetConfig,
    name: str,
    *,
    expected_names: tuple[str, ...],
    expected_resource_hash: str,
) -> FloatArray:
    raw = _dataset_resource_bytes(config, name)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_resource_hash:
        message = (
            f"Packaged {config.display_name} resource {name!r} "
            "failed its integrity check."
        )
        raise RuntimeError(message)

    try:
        rows = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""))
        header = tuple(next(rows))
        values = [[float(value) for value in row] for row in rows]
    except (StopIteration, UnicodeDecodeError, ValueError) as error:
        message = (
            f"Packaged {config.display_name} resource {name!r} "
            "is not a valid numeric CSV."
        )
        raise RuntimeError(message) from error
    if header != expected_names:
        message = (
            f"Packaged {config.display_name} resource {name!r} "
            "has unexpected column names."
        )
        raise RuntimeError(message)
    if any(len(row) != len(header) for row in values):
        message = (
            f"Packaged {config.display_name} resource {name!r} "
            "has an irregular row width."
        )
        raise RuntimeError(message)
    return _validated_matrix(values, name=name, allow_vector=False)


def _verify_dataset_array_hash(
    config: _PackagedDatasetConfig,
    array: FloatArray,
    *,
    expected: str,
    name: str,
) -> None:
    canonical = np.asarray(array, dtype=np.dtype("<f8"), order="C")
    digest = hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
    if digest != expected:
        message = (
            f"Packaged {config.display_name} {name} array failed its integrity check."
        )
        raise RuntimeError(message)


def _metadata_mapping(
    config: _PackagedDatasetConfig,
    values: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = values.get(key)
    if not isinstance(value, Mapping):
        message = (
            f"Packaged {config.display_name} metadata field {key!r} must be an object."
        )
        raise RuntimeError(message)
    return value


def _metadata_string(
    config: _PackagedDatasetConfig,
    values: Mapping[str, object],
    key: str,
) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        message = (
            f"Packaged {config.display_name} metadata field {key!r} must be a string."
        )
        raise RuntimeError(message)
    return value


def _metadata_integer(
    config: _PackagedDatasetConfig,
    values: Mapping[str, object],
    key: str,
) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        message = (
            f"Packaged {config.display_name} metadata field {key!r} must be an integer."
        )
        raise RuntimeError(message)
    return value


def _metadata_string_tuple(
    config: _PackagedDatasetConfig,
    values: Mapping[str, object],
    key: str,
) -> tuple[str, ...]:
    value = values.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        message = (
            f"Packaged {config.display_name} metadata field {key!r} "
            "must be a string array."
        )
        raise RuntimeError(message)
    return cast(tuple[str, ...], tuple(value))
