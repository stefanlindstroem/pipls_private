"""Shared validation helpers for immutable public result records."""

from __future__ import annotations

from collections.abc import Collection
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be a positive integer.")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return converted


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be a finite real number.")
    return converted


def _nonnegative_finite_float(value: object, *, name: str) -> float:
    converted = _finite_float(value, name=name)
    if converted < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return converted


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean.")
    return bool(value)


def _literal_string(
    value: object,
    *,
    name: str,
    allowed: Collection[str],
) -> str:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(repr(choice) for choice in sorted(allowed))
        raise ValueError(f"{name} must be one of {choices}.")
    return value


def _read_only_float_array(
    value: ArrayLike,
    *,
    name: str,
    ndim: int = 1,
    require_finite: bool = True,
) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got shape {array.shape}.")
    if require_finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def _read_only_int_array(
    value: ArrayLike,
    *,
    name: str,
    ndim: int = 1,
) -> IntArray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got shape {raw.shape}.")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must contain integers.")
    array = np.array(raw, dtype=np.intp, copy=True)
    array.setflags(write=False)
    return array
