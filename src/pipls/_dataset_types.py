"""Immutable dataset containers and validation support."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypeVar

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
class PiPLSRegressionTruth:
    r"""Immutable latent structure used by the configurable regression generator.

    All arrays are read-only copies. The result stores only loading blocks that
    contribute to the generated predictor or response signal.

    Attributes
    ----------
    shared_scores : ndarray of shape (n_samples, n_shared)
        Latent scores shared by predictors and responses.
    predictor_specific_scores : ndarray of shape (n_samples, n_predictor_specific)
        Latent scores affecting only predictors.
    response_specific_scores : ndarray of shape (n_samples, n_response_specific)
        Latent scores affecting only responses.
    x_shared_loadings, y_shared_loadings : ndarray
        Predictor and response loading blocks for shared directions.
    x_predictor_specific_loadings : ndarray of shape (n_features, n_predictor_specific)
        Predictor-specific loading block.
    y_response_specific_loadings : ndarray of shape (n_targets, n_response_specific)
        Response-specific loading block.
    x_signal, x_noise : ndarray of shape (n_samples, n_features)
        Predictor signal and additive noise, whose sum is the generated ``X``.
    y_signal, y_noise : ndarray of shape (n_samples, n_targets)
        Response signal and additive noise, whose sum is the generated ``Y``.
    feature_scale : ndarray of shape (n_features,)
        Multiplicative observed-predictor scales.
    target_scale : ndarray of shape (n_targets,)
        Multiplicative observed-response scales.
    shared_strengths, predictor_specific_strengths, response_specific_strengths : ndarray
        Strength of each latent direction in its corresponding block.
    """

    shared_scores: FloatArray
    predictor_specific_scores: FloatArray
    response_specific_scores: FloatArray
    x_shared_loadings: FloatArray
    x_predictor_specific_loadings: FloatArray
    y_shared_loadings: FloatArray
    y_response_specific_loadings: FloatArray
    x_signal: FloatArray
    y_signal: FloatArray
    x_noise: FloatArray
    y_noise: FloatArray
    feature_scale: FloatArray
    target_scale: FloatArray
    shared_strengths: FloatArray
    predictor_specific_strengths: FloatArray
    response_specific_strengths: FloatArray

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            object.__setattr__(self, name, _read_only_float_array(value, name=name))

    @property
    def n_shared(self) -> int:
        """Number of latent directions shared by predictors and responses."""

        return int(self.shared_scores.shape[1])

    @property
    def n_predictor_specific(self) -> int:
        """Number of latent directions appearing only in predictors."""

        return int(self.predictor_specific_scores.shape[1])

    @property
    def n_response_specific(self) -> int:
        """Number of latent directions appearing only in responses."""

        return int(self.response_specific_scores.shape[1])


@dataclass(frozen=True)
class PiPLSLatentGeometryTruth:
    r"""Immutable manuscript latent geometry for one synthetic dataset.

    This record stores the terms of the latent-geometry equation in the
    Synthetic generators section of the dataset API reference. All score,
    loading, signal, and noise arrays are defensive read-only ``float64``
    copies. Loading matrices follow the companion-manuscript orientation, with
    latent dimensions on rows and observed variables on columns.

    Attributes
    ----------
    predictor_specific_scores : ndarray of shape (n_samples, n_predictor_specific)
        Predictor-specific latent matrix $\boldsymbol{\Lambda}_{\mathrm{p}}$.
    shared_scores : ndarray of shape (n_samples, n_shared)
        Shared latent matrix $\boldsymbol{\Lambda}_{\mathrm{s}}$.
    response_specific_scores : ndarray of shape (n_samples, n_response_specific)
        Response-specific latent matrix $\boldsymbol{\Lambda}_{\mathrm{r}}$.
    predictor_specific_loadings : ndarray of shape (n_predictor_specific, n_features)
        Predictor-specific loading matrix $\mathbf{L}_{\mathrm{p}}$.
    shared_predictor_loadings : ndarray of shape (n_shared, n_features)
        Shared predictor loading matrix $\mathbf{L}_{\mathrm{sp}}$.
    shared_response_loadings : ndarray of shape (n_shared, n_targets)
        Shared response loading matrix $\mathbf{L}_{\mathrm{sr}}$.
    response_specific_loadings : ndarray of shape (n_response_specific, n_targets)
        Response-specific loading matrix $\mathbf{L}_{\mathrm{r}}$.
    x_signal, x_noise : ndarray of shape (n_samples, n_features)
        Noise-free predictor signal and additive noise.
    y_signal, y_noise : ndarray of shape (n_samples, n_targets)
        Noise-free response signal and additive noise.
    """

    predictor_specific_scores: FloatArray
    shared_scores: FloatArray
    response_specific_scores: FloatArray
    predictor_specific_loadings: FloatArray
    shared_predictor_loadings: FloatArray
    shared_response_loadings: FloatArray
    response_specific_loadings: FloatArray
    x_signal: FloatArray
    y_signal: FloatArray
    x_noise: FloatArray
    y_noise: FloatArray

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = _read_only_float_array(getattr(self, name), name=name)
            if value.ndim != 2:
                raise ValueError(f"{name} must be two-dimensional.")
            object.__setattr__(self, name, value)
        _validate_latent_geometry_truth(self)

    @property
    def n_shared(self) -> int:
        r"""Number of shared latent directions $d_{\mathrm{s}}$."""

        return int(self.shared_scores.shape[1])

    @property
    def n_predictor_specific(self) -> int:
        r"""Number of predictor-specific latent directions $d_{\mathrm{p}}$."""

        return int(self.predictor_specific_scores.shape[1])

    @property
    def n_response_specific(self) -> int:
        r"""Number of response-specific latent directions $d_{\mathrm{r}}$."""

        return int(self.response_specific_scores.shape[1])

    def __reduce__(self) -> tuple[object, tuple[FloatArray, ...]]:
        return (
            type(self),
            tuple(getattr(self, name) for name in self.__dataclass_fields__),
        )


_SyntheticTruth: TypeAlias = PiPLSRegressionTruth | PiPLSLatentGeometryTruth


@dataclass(frozen=True)
class PiPLSDataset:
    r"""Immutable validated multivariate regression dataset.

    Plain arrays and data frames passed directly to ``fit(X, Y)`` remain the
    primary real-data interface. This container carries packaged reference
    datasets, synthetic data, and structured experiment data.

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
    truth : PiPLSRegressionTruth, PiPLSLatentGeometryTruth, or None, default=None
        Optional synthetic latent structure consistent with ``X`` and ``Y``.

    Attributes
    ----------
    X, Y : ndarray
        Read-only ``float64`` predictor and two-dimensional response matrices.
    feature_names, target_names, sample_ids : tuple of str
        Validated axis labels.
    provenance, metadata : mapping
        Immutable mappings.
    truth : PiPLSRegressionTruth, PiPLSLatentGeometryTruth, or None
        Optional synthetic truth object.
    """

    X: FloatArray
    Y: FloatArray
    feature_names: Sequence[str]
    target_names: Sequence[str]
    sample_ids: Sequence[str]
    provenance: Mapping[str, str]
    metadata: Mapping[str, object] = field(default_factory=dict)
    truth: _SyntheticTruth | None = None

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

        if self.truth is not None:
            _validate_dataset_truth(
                self.truth,
                n_samples=X.shape[0],
                n_features=X.shape[1],
                n_targets=Y.shape[1],
            )
            if not np.allclose(X, self.truth.x_signal + self.truth.x_noise):
                raise ValueError("X must equal truth.x_signal + truth.x_noise.")
            if not np.allclose(Y, self.truth.y_signal + self.truth.y_noise):
                raise ValueError("Y must equal truth.y_signal + truth.y_noise.")

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
                self.truth,
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


def _validate_dataset_truth(
    truth: _SyntheticTruth,
    *,
    n_samples: int,
    n_features: int,
    n_targets: int,
) -> None:
    if isinstance(truth, PiPLSLatentGeometryTruth):
        if truth.x_signal.shape != (n_samples, n_features):
            raise ValueError(
                f"truth.x_signal must have shape {(n_samples, n_features)}."
            )
        if truth.y_signal.shape != (n_samples, n_targets):
            raise ValueError(
                f"truth.y_signal must have shape {(n_samples, n_targets)}."
            )
        return
    if not isinstance(truth, PiPLSRegressionTruth):
        raise TypeError(
            "truth must be PiPLSRegressionTruth or PiPLSLatentGeometryTruth."
        )

    n_shared = truth.n_shared
    n_predictor_specific = truth.n_predictor_specific
    n_response_specific = truth.n_response_specific
    expected_shapes = {
        "shared_scores": (n_samples, n_shared),
        "predictor_specific_scores": (n_samples, n_predictor_specific),
        "response_specific_scores": (n_samples, n_response_specific),
        "x_shared_loadings": (n_features, n_shared),
        "x_predictor_specific_loadings": (n_features, n_predictor_specific),
        "y_shared_loadings": (n_targets, n_shared),
        "y_response_specific_loadings": (n_targets, n_response_specific),
        "x_signal": (n_samples, n_features),
        "y_signal": (n_samples, n_targets),
        "x_noise": (n_samples, n_features),
        "y_noise": (n_samples, n_targets),
        "feature_scale": (n_features,),
        "target_scale": (n_targets,),
        "shared_strengths": (n_shared,),
        "predictor_specific_strengths": (n_predictor_specific,),
        "response_specific_strengths": (n_response_specific,),
    }
    for name, expected in expected_shapes.items():
        if getattr(truth, name).shape != expected:
            raise ValueError(f"truth.{name} must have shape {expected}.")
    if np.any(truth.feature_scale <= 0.0) or np.any(truth.target_scale <= 0.0):
        raise ValueError("truth observed-variable scales must be positive.")
    strength_arrays = (
        truth.shared_strengths,
        truth.predictor_specific_strengths,
        truth.response_specific_strengths,
    )
    if any(np.any(strengths <= 0.0) for strengths in strength_arrays):
        raise ValueError("truth latent strengths must be positive.")


def _validate_latent_geometry_truth(truth: PiPLSLatentGeometryTruth) -> None:
    n_samples = truth.shared_scores.shape[0]
    if truth.predictor_specific_scores.shape[0] != n_samples:
        raise ValueError("All latent score matrices must contain the same samples.")
    if truth.response_specific_scores.shape[0] != n_samples:
        raise ValueError("All latent score matrices must contain the same samples.")

    n_features = truth.shared_predictor_loadings.shape[1]
    n_targets = truth.shared_response_loadings.shape[1]
    expected_shapes = {
        "predictor_specific_loadings": (
            truth.n_predictor_specific,
            n_features,
        ),
        "shared_predictor_loadings": (truth.n_shared, n_features),
        "shared_response_loadings": (truth.n_shared, n_targets),
        "response_specific_loadings": (
            truth.n_response_specific,
            n_targets,
        ),
        "x_signal": (n_samples, n_features),
        "x_noise": (n_samples, n_features),
        "y_signal": (n_samples, n_targets),
        "y_noise": (n_samples, n_targets),
    }
    for name, expected in expected_shapes.items():
        if getattr(truth, name).shape != expected:
            raise ValueError(f"{name} must have shape {expected}.")

    expected_x_signal = (
        truth.predictor_specific_scores @ truth.predictor_specific_loadings
        + truth.shared_scores @ truth.shared_predictor_loadings
    )
    expected_y_signal = (
        truth.shared_scores @ truth.shared_response_loadings
        + truth.response_specific_scores @ truth.response_specific_loadings
    )
    if not np.allclose(truth.x_signal, expected_x_signal):
        raise ValueError("x_signal must follow the manuscript latent-geometry equation.")
    if not np.allclose(truth.y_signal, expected_y_signal):
        raise ValueError("y_signal must follow the manuscript latent-geometry equation.")


def _read_only_float_array(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")
    array = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array
