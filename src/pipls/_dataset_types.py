"""Immutable dataset containers and validation support."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

K = TypeVar("K")
V = TypeVar("V")


class _FrozenMapping(Mapping[K, V], Generic[K, V]):
    """Small immutable and pickleable mapping used by public dataset objects."""

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[K, V]) -> None:
        self._data = dict(values)

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __reduce__(self) -> tuple[object, tuple[dict[K, V]]]:
        return (_FrozenMapping, (self._data,))


_REQUIRED_PROVENANCE_KEYS = ("source", "license", "citation", "version")


@dataclass(frozen=True)
class PiPLSDataset:
    r"""Immutable validated multivariate regression dataset.

    Plain arrays and data frames passed directly to ``fit(X, Y)`` remain the
    primary real-data interface. This container carries packaged reference
    datasets and structured experiment data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Numeric predictor matrix.
    Y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Numeric response data. One-dimensional input is stored as one column.
    feature_names : sequence of str
        Unique nonempty predictor names.
    target_names : sequence of str
        Unique nonempty response names.
    sample_ids : sequence of str
        Unique nonempty sample identifiers.
    provenance : mapping of str to str
        Nonempty ``source``, ``license``, ``citation``, and ``version`` entries.
    metadata : mapping of str to object, default={}
        Recursively frozen dataset metadata. NumPy metadata arrays must not use
        object dtype, because object-array elements can remain mutable.
    Attributes
    ----------
    X, Y : ndarray
        Read-only ``float64`` predictor and two-dimensional response matrices.
    feature_names, target_names, sample_ids : tuple of str
        Validated axis labels.
    provenance, metadata : mapping
        Immutable mappings.
    """

    X: FloatArray
    Y: FloatArray
    feature_names: Sequence[str]
    target_names: Sequence[str]
    sample_ids: Sequence[str]
    provenance: Mapping[str, str]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        X = _validated_matrix(self.X, name="X", allow_vector=False)
        Y = _validated_matrix(self.Y, name="Y", allow_vector=True)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must contain the same number of samples.")
        if X.shape[0] == 0 or X.shape[1] == 0 or Y.shape[1] == 0:
            raise ValueError("X and Y must contain at least one sample and one column.")

        feature_names = _validated_names(
            self.feature_names,
            expected=X.shape[1],
            name="feature_names",
        )
        target_names = _validated_names(
            self.target_names,
            expected=Y.shape[1],
            name="target_names",
        )
        sample_ids = _validated_names(
            self.sample_ids,
            expected=X.shape[0],
            name="sample_ids",
        )
        provenance = _validated_provenance(self.provenance)
        metadata = _freeze_mapping(self.metadata, name="metadata")

        object.__setattr__(self, "X", X)
        object.__setattr__(self, "Y", Y)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "target_names", target_names)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "metadata", metadata)

    @property
    def n_samples(self) -> int:
        """Number of aligned observations."""

        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Number of predictor columns."""

        return int(self.X.shape[1])

    @property
    def n_targets(self) -> int:
        """Number of response columns."""

        return int(self.Y.shape[1])

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Reconstruct through validation when unpickling."""

        return (
            type(self),
            (
                self.X,
                self.Y,
                self.feature_names,
                self.target_names,
                self.sample_ids,
                self.provenance,
                self.metadata,
            ),
        )


def _validated_matrix(
    value: object,
    *,
    name: str,
    allow_vector: bool,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")
    if allow_vector and raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if raw.ndim != 2:
        expected = "one or two dimensions" if allow_vector else "two dimensions"
        raise ValueError(f"{name} must have {expected}.")
    array = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def _validated_names(
    values: Sequence[str],
    *,
    expected: int,
    name: str,
) -> tuple[str, ...]:
    result = tuple(values)
    if len(result) != expected:
        raise ValueError(f"{name} must contain exactly {expected} values.")
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise TypeError(f"{name} must contain non-empty strings.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values.")
    return result


def _validated_provenance(values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError("provenance must be a mapping.")
    copied: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise TypeError("provenance keys must be non-empty strings.")
        if not isinstance(value, str) or not value.strip():
            raise TypeError("provenance values must be non-empty strings.")
        copied[key] = value
    missing = [key for key in _REQUIRED_PROVENANCE_KEYS if key not in copied]
    if missing:
        raise ValueError(f"provenance is missing required keys: {missing}.")
    return _FrozenMapping(copied)


def _freeze_mapping(values: Mapping[str, object], *, name: str) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    result: dict[str, object] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise TypeError(f"{name} keys must be non-empty strings.")
        result[key] = _freeze_metadata_value(value, path=f"{name}[{key!r}]")
    return _FrozenMapping(result)


def _freeze_metadata_value(value: object, *, path: str) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{path} must be finite.")
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(
                f"{path} must not use an object-dtype NumPy array; "
                "use nested sequences or mappings for heterogeneous metadata."
            )
        array = np.array(value, copy=True)
        if array.dtype.kind in "fci" and not np.all(np.isfinite(array)):
            raise ValueError(f"{path} must contain only finite values.")
        array.setflags(write=False)
        return array
    if isinstance(value, Mapping):
        return _freeze_mapping(value, name=path)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_metadata_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} has unsupported type {type(value).__name__}; use immutable scalars, "
        "sequences, mappings, or NumPy arrays."
    )
