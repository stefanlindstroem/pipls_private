"""Validated dataset containers and deterministic synthetic Pi-PLS data."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
Distribution: TypeAlias = Literal["normal", "uniform"]
NumericSpec: TypeAlias = float | Sequence[float]
NoiseSpec: TypeAlias = float | tuple[float, float]

__all__ = [
    "PiPLSDataset",
    "PiPLSLatentGeometryTruth",
    "PiPLSSyntheticTruth",
    "make_pipls_latent_geometry",
    "make_pipls_regression",
    "make_pipls_train_test",
]


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
_UINT32_MAX = 2**32 - 1


@dataclass(frozen=True)
class PiPLSSyntheticTruth:
    r"""Immutable latent structure used to generate a synthetic dataset.

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

    This record follows the orientation of the companion manuscript directly:

    \begin{equation}
    \mathbf{X}
    =
    \boldsymbol{\Lambda}_{\mathrm{p}}\mathbf{L}_{\mathrm{p}}
    +
    \boldsymbol{\Lambda}_{\mathrm{s}}\mathbf{L}_{\mathrm{sp}}
    +
    \boldsymbol{\varepsilon}_{\mathrm{X}},
    \qquad
    \mathbf{Y}
    =
    \boldsymbol{\Lambda}_{\mathrm{s}}\mathbf{L}_{\mathrm{sr}}
    +
    \boldsymbol{\Lambda}_{\mathrm{r}}\mathbf{L}_{\mathrm{r}}
    +
    \boldsymbol{\varepsilon}_{\mathrm{Y}}.
    \end{equation}

    All score, loading, signal, and noise arrays are defensive read-only
    ``float64`` copies. Loading matrices therefore have latent dimensions on
    rows and observed variables on columns.

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


_SyntheticTruth: TypeAlias = PiPLSSyntheticTruth | PiPLSLatentGeometryTruth


@dataclass(frozen=True)
class PiPLSDataset:
    r"""Immutable validated multivariate regression dataset.

    Plain arrays and data frames passed directly to ``fit(X, Y)`` remain the
    primary real-data interface. This container is an optional convenience for
    synthetic data and structured experiments.

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
    truth : PiPLSSyntheticTruth, PiPLSLatentGeometryTruth, or None, default=None
        Optional synthetic latent structure consistent with ``X`` and ``Y``.

    Attributes
    ----------
    X, Y : ndarray
        Read-only ``float64`` predictor and two-dimensional response matrices.
    feature_names, target_names, sample_ids : tuple of str
        Validated axis labels.
    provenance, metadata : mapping
        Immutable mappings.
    truth : PiPLSSyntheticTruth, PiPLSLatentGeometryTruth, or None
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
            _validate_truth(
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
    def data(self) -> FloatArray:
        """Scikit-learn-style alias for :attr:`X`."""

        return self.X

    @property
    def target(self) -> FloatArray:
        """Scikit-learn-style alias for :attr:`Y`."""

        return self.Y

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


@dataclass(frozen=True)
class _SyntheticLoadings:
    x_shared_loadings: FloatArray
    x_predictor_specific_loadings: FloatArray
    y_shared_loadings: FloatArray
    y_response_specific_loadings: FloatArray


@dataclass(frozen=True)
class _SyntheticConfig:
    n_features: int
    n_targets: int
    n_shared: int
    n_predictor_specific: int
    n_response_specific: int
    shared_strengths: FloatArray
    predictor_specific_strengths: FloatArray
    response_specific_strengths: FloatArray
    shared_distribution: Distribution
    predictor_specific_distribution: Distribution
    response_specific_distribution: Distribution
    feature_scale: FloatArray
    target_scale: FloatArray
    x_noise: float
    y_noise: float
    random_state: int


def make_pipls_latent_geometry(
    *,
    n_samples: int,
    n_features: int,
    n_targets: int,
    n_shared: int,
    n_predictor_specific: int = 0,
    n_response_specific: int = 0,
    noise: NoiseSpec = 0.0,
    random_state: int = 0,
) -> PiPLSDataset:
    r"""Generate the Gaussian latent geometry used in the companion manuscript.

    The function implements the manuscript data model directly:

    \begin{equation}
    \mathbf{X}
    =
    \boldsymbol{\Lambda}_{\mathrm{p}}\mathbf{L}_{\mathrm{p}}
    +
    \boldsymbol{\Lambda}_{\mathrm{s}}\mathbf{L}_{\mathrm{sp}}
    +
    \boldsymbol{\varepsilon}_{\mathrm{X}},
    \qquad
    \mathbf{Y}
    =
    \boldsymbol{\Lambda}_{\mathrm{s}}\mathbf{L}_{\mathrm{sr}}
    +
    \boldsymbol{\Lambda}_{\mathrm{r}}\mathbf{L}_{\mathrm{r}}
    +
    \boldsymbol{\varepsilon}_{\mathrm{Y}}.
    \end{equation}

    Every entry of the three latent-score matrices and four loading matrices is
    drawn independently from $\mathcal{N}(0,1)$. Predictor and response
    noise entries are independent Gaussian draws with the requested standard
    deviations. No score centering, score standardization, loading
    orthonormalization, latent-strength scaling, or observed-variable scaling is
    applied.

    This manuscript-aligned generator is separate from
    :func:`make_pipls_regression`, which remains the configurable package
    generator used by existing examples and benchmarks.

    Parameters
    ----------
    n_samples : int
        Number of observations; at least one.
    n_features : int
        Number of predictor variables $p$.
    n_targets : int
        Number of response variables $q$.
    n_shared : int
        Shared latent dimension $d_{\mathrm{s}}$.
    n_predictor_specific : int, default=0
        Predictor-specific latent dimension $d_{\mathrm{p}}$.
    n_response_specific : int, default=0
        Response-specific latent dimension $d_{\mathrm{r}}$.
    noise : float or tuple of float, default=0.0
        Common noise standard deviation, or separate
        ``(sigma_x, sigma_y)`` values.
    random_state : int, default=0
        Local deterministic unsigned 32-bit random seed.

    Returns
    -------
    PiPLSDataset
        Generated matrices, manuscript-oriented latent truth, metadata, and
        provenance.
    """

    n_samples = _positive_integer(n_samples, name="n_samples")
    n_features = _positive_integer(n_features, name="n_features")
    n_targets = _positive_integer(n_targets, name="n_targets")
    n_shared = _nonnegative_integer(n_shared, name="n_shared")
    n_predictor_specific = _nonnegative_integer(
        n_predictor_specific,
        name="n_predictor_specific",
    )
    n_response_specific = _nonnegative_integer(
        n_response_specific,
        name="n_response_specific",
    )
    if n_predictor_specific + n_shared > n_features:
        raise ValueError("n_predictor_specific + n_shared must not exceed n_features.")
    if n_shared + n_response_specific > n_targets:
        raise ValueError("n_shared + n_response_specific must not exceed n_targets.")

    sigma_x, sigma_y = _resolve_noise(noise)
    seed = _validated_seed(random_state)
    rng = np.random.default_rng(seed)

    predictor_specific_scores = rng.standard_normal(
        (n_samples, n_predictor_specific)
    )
    shared_scores = rng.standard_normal((n_samples, n_shared))
    response_specific_scores = rng.standard_normal(
        (n_samples, n_response_specific)
    )
    predictor_specific_loadings = rng.standard_normal(
        (n_predictor_specific, n_features)
    )
    shared_predictor_loadings = rng.standard_normal((n_shared, n_features))
    shared_response_loadings = rng.standard_normal((n_shared, n_targets))
    response_specific_loadings = rng.standard_normal(
        (n_response_specific, n_targets)
    )
    x_noise = sigma_x * rng.standard_normal((n_samples, n_features))
    y_noise = sigma_y * rng.standard_normal((n_samples, n_targets))

    x_signal = (
        predictor_specific_scores @ predictor_specific_loadings
        + shared_scores @ shared_predictor_loadings
    )
    y_signal = (
        shared_scores @ shared_response_loadings
        + response_specific_scores @ response_specific_loadings
    )
    truth = PiPLSLatentGeometryTruth(
        predictor_specific_scores=predictor_specific_scores,
        shared_scores=shared_scores,
        response_specific_scores=response_specific_scores,
        predictor_specific_loadings=predictor_specific_loadings,
        shared_predictor_loadings=shared_predictor_loadings,
        shared_response_loadings=shared_response_loadings,
        response_specific_loadings=response_specific_loadings,
        x_signal=x_signal,
        y_signal=y_signal,
        x_noise=x_noise,
        y_noise=y_noise,
    )
    metadata: Mapping[str, object] = {
        "schema_version": 1,
        "generator": "make_pipls_latent_geometry",
        "random_state": seed,
        "latent_dimensions": {
            "predictor_specific": n_predictor_specific,
            "shared": n_shared,
            "response_specific": n_response_specific,
        },
        "noise_standard_deviation": {"X": sigma_x, "Y": sigma_y},
        "distribution": "independent standard normal scores and loadings",
    }
    provenance = {
        "source": "generated:pipls.datasets.make_pipls_latent_geometry",
        "license": "BSD-3-Clause",
        "citation": (
            "Agrawal, Vishal; Nilsson, Fritjof; Lindström, Stefan B. "
            "Panoramic Partial Least Squares (Pi-PLS): Transparent, parsimonious, "
            "and more interpretable multivariate regression model. Manuscript under revision."
        ),
        "version": "1",
    }
    return PiPLSDataset(
        X=x_signal + x_noise,
        Y=y_signal + y_noise,
        feature_names=tuple(f"x_{index:03d}" for index in range(n_features)),
        target_names=tuple(f"y_{index:03d}" for index in range(n_targets)),
        sample_ids=tuple(f"sample_{index:04d}" for index in range(n_samples)),
        provenance=provenance,
        metadata=metadata,
        truth=truth,
    )


def make_pipls_regression(
    *,
    n_samples: int,
    n_features: int,
    n_targets: int,
    n_shared: int,
    n_predictor_specific: int = 0,
    n_response_specific: int = 0,
    shared_strength: NumericSpec = 1.0,
    predictor_specific_strength: NumericSpec = 1.0,
    response_specific_strength: NumericSpec = 1.0,
    shared_distribution: Distribution = "normal",
    predictor_specific_distribution: Distribution = "normal",
    response_specific_distribution: Distribution = "normal",
    feature_scale: NumericSpec = 1.0,
    target_scale: NumericSpec = 1.0,
    noise: NoiseSpec = 0.1,
    random_state: int = 0,
) -> PiPLSDataset:
    r"""Generate one deterministic Pi-PLS latent-structure dataset.

    Shared latent scores affect both ``X`` and ``Y``. Predictor-specific scores
    affect only ``X`` and response-specific scores affect only ``Y``.

    Parameters
    ----------
    n_samples : int
        Number of observations; at least two.
    n_features : int
        Number of predictor variables.
    n_targets : int
        Number of response variables.
    n_shared : int
        Number of latent directions shared by predictors and responses.
    n_predictor_specific : int, default=0
        Number of predictor-only latent directions.
    n_response_specific : int, default=0
        Number of response-only latent directions.
    shared_strength : float or sequence of float, default=1.0
        Strengths of the shared directions.
    predictor_specific_strength : float or sequence of float, default=1.0
        Strengths of the predictor-only directions.
    response_specific_strength : float or sequence of float, default=1.0
        Strengths of the response-only directions.
    shared_distribution : {"normal", "uniform"}, default="normal"
        Shared-score distribution.
    predictor_specific_distribution : {"normal", "uniform"}, default="normal"
        Predictor-only score distribution.
    response_specific_distribution : {"normal", "uniform"}, default="normal"
        Response-only score distribution.
    feature_scale : float or sequence of float, default=1.0
        Scalar or predictor-wise observed scales.
    target_scale : float or sequence of float, default=1.0
        Scalar or response-wise observed scales.
    noise : float or tuple of float, default=0.1
        Common noise standard deviation, or separate ``(x_noise, y_noise)`` values.
    random_state : int, default=0
        Local deterministic random seed.

    Returns
    -------
    PiPLSDataset
        Generated data, metadata, provenance, and latent truth.
    """

    n_samples = _positive_integer(n_samples, name="n_samples", minimum=2)
    config = _validated_synthetic_config(
        n_features=n_features,
        n_targets=n_targets,
        n_shared=n_shared,
        n_predictor_specific=n_predictor_specific,
        n_response_specific=n_response_specific,
        shared_strength=shared_strength,
        predictor_specific_strength=predictor_specific_strength,
        response_specific_strength=response_specific_strength,
        shared_distribution=shared_distribution,
        predictor_specific_distribution=predictor_specific_distribution,
        response_specific_distribution=response_specific_distribution,
        feature_scale=feature_scale,
        target_scale=target_scale,
        noise=noise,
        random_state=random_state,
    )
    _validate_sample_capacity(n_samples, config, name="n_samples")
    rng = np.random.default_rng(config.random_state)
    loadings = _draw_synthetic_loadings(rng, config)
    return _draw_dataset_block(
        rng,
        config,
        loadings,
        n_samples=n_samples,
        sample_prefix="sample",
        split_role="full",
        generator_name="make_pipls_regression",
    )


def make_pipls_train_test(
    *,
    n_train: int,
    n_test: int,
    n_features: int,
    n_targets: int,
    n_shared: int,
    n_predictor_specific: int = 0,
    n_response_specific: int = 0,
    shared_strength: NumericSpec = 1.0,
    predictor_specific_strength: NumericSpec = 1.0,
    response_specific_strength: NumericSpec = 1.0,
    shared_distribution: Distribution = "normal",
    predictor_specific_distribution: Distribution = "normal",
    response_specific_distribution: Distribution = "normal",
    feature_scale: NumericSpec = 1.0,
    target_scale: NumericSpec = 1.0,
    noise: NoiseSpec = 0.1,
    random_state: int = 0,
) -> tuple[PiPLSDataset, PiPLSDataset]:
    r"""Generate independent train and test blocks from one latent model.

    Loadings, strengths, and observed-variable scales are shared. Latent scores and
    noise are generated independently for the two blocks. No fitted preprocessing
    is performed.

    Parameters
    ----------
    n_train, n_test : int
        Numbers of training and test observations; each at least two.
    n_features : int
        Number of predictor variables.
    n_targets : int
        Number of response variables.
    n_shared : int
        Number of latent directions shared by predictors and responses.
    n_predictor_specific : int, default=0
        Number of predictor-only latent directions.
    n_response_specific : int, default=0
        Number of response-only latent directions.
    shared_strength : float or sequence of float, default=1.0
        Strengths of the shared directions.
    predictor_specific_strength : float or sequence of float, default=1.0
        Strengths of the predictor-only directions.
    response_specific_strength : float or sequence of float, default=1.0
        Strengths of the response-only directions.
    shared_distribution : {"normal", "uniform"}, default="normal"
        Shared-score distribution.
    predictor_specific_distribution : {"normal", "uniform"}, default="normal"
        Predictor-only score distribution.
    response_specific_distribution : {"normal", "uniform"}, default="normal"
        Response-only score distribution.
    feature_scale : float or sequence of float, default=1.0
        Scalar or predictor-wise observed scales.
    target_scale : float or sequence of float, default=1.0
        Scalar or response-wise observed scales.
    noise : float or tuple of float, default=0.1
        Common noise standard deviation, or separate ``(x_noise, y_noise)`` values.
    random_state : int, default=0
        Local deterministic random seed.

    Returns
    -------
    train, test : tuple of PiPLSDataset
        Independent blocks sharing one generated latent model.
    """

    n_train = _positive_integer(n_train, name="n_train", minimum=2)
    n_test = _positive_integer(n_test, name="n_test", minimum=2)
    config = _validated_synthetic_config(
        n_features=n_features,
        n_targets=n_targets,
        n_shared=n_shared,
        n_predictor_specific=n_predictor_specific,
        n_response_specific=n_response_specific,
        shared_strength=shared_strength,
        predictor_specific_strength=predictor_specific_strength,
        response_specific_strength=response_specific_strength,
        shared_distribution=shared_distribution,
        predictor_specific_distribution=predictor_specific_distribution,
        response_specific_distribution=response_specific_distribution,
        feature_scale=feature_scale,
        target_scale=target_scale,
        noise=noise,
        random_state=random_state,
    )
    _validate_sample_capacity(n_train, config, name="n_train")
    _validate_sample_capacity(n_test, config, name="n_test")
    rng = np.random.default_rng(config.random_state)
    loadings = _draw_synthetic_loadings(rng, config)
    train = _draw_dataset_block(
        rng,
        config,
        loadings,
        n_samples=n_train,
        sample_prefix="train",
        split_role="train",
        generator_name="make_pipls_train_test",
    )
    test = _draw_dataset_block(
        rng,
        config,
        loadings,
        n_samples=n_test,
        sample_prefix="test",
        split_role="test",
        generator_name="make_pipls_train_test",
    )
    return train, test


def _validated_synthetic_config(
    *,
    n_features: int,
    n_targets: int,
    n_shared: int,
    n_predictor_specific: int,
    n_response_specific: int,
    shared_strength: NumericSpec,
    predictor_specific_strength: NumericSpec,
    response_specific_strength: NumericSpec,
    shared_distribution: Distribution,
    predictor_specific_distribution: Distribution,
    response_specific_distribution: Distribution,
    feature_scale: NumericSpec,
    target_scale: NumericSpec,
    noise: NoiseSpec,
    random_state: int,
) -> _SyntheticConfig:
    n_features = _positive_integer(n_features, name="n_features")
    n_targets = _positive_integer(n_targets, name="n_targets")
    n_shared = _nonnegative_integer(n_shared, name="n_shared")
    n_predictor_specific = _nonnegative_integer(
        n_predictor_specific,
        name="n_predictor_specific",
    )
    n_response_specific = _nonnegative_integer(
        n_response_specific,
        name="n_response_specific",
    )
    if n_shared + n_predictor_specific > n_features:
        raise ValueError("n_shared + n_predictor_specific must not exceed n_features.")
    if n_shared + n_response_specific > n_targets:
        raise ValueError("n_shared + n_response_specific must not exceed n_targets.")

    distributions = (
        ("shared_distribution", shared_distribution),
        ("predictor_specific_distribution", predictor_specific_distribution),
        ("response_specific_distribution", response_specific_distribution),
    )
    for name, value in distributions:
        if value not in ("normal", "uniform"):
            raise ValueError(f"{name} must be 'normal' or 'uniform'.")

    x_noise, y_noise = _resolve_noise(noise)
    return _SyntheticConfig(
        n_features=n_features,
        n_targets=n_targets,
        n_shared=n_shared,
        n_predictor_specific=n_predictor_specific,
        n_response_specific=n_response_specific,
        shared_strengths=_resolve_positive_vector(
            shared_strength,
            length=n_shared,
            name="shared_strength",
        ),
        predictor_specific_strengths=_resolve_positive_vector(
            predictor_specific_strength,
            length=n_predictor_specific,
            name="predictor_specific_strength",
        ),
        response_specific_strengths=_resolve_positive_vector(
            response_specific_strength,
            length=n_response_specific,
            name="response_specific_strength",
        ),
        shared_distribution=shared_distribution,
        predictor_specific_distribution=predictor_specific_distribution,
        response_specific_distribution=response_specific_distribution,
        feature_scale=_resolve_positive_vector(
            feature_scale,
            length=n_features,
            name="feature_scale",
        ),
        target_scale=_resolve_positive_vector(
            target_scale,
            length=n_targets,
            name="target_scale",
        ),
        x_noise=x_noise,
        y_noise=y_noise,
        random_state=_validated_seed(random_state),
    )


def _validate_sample_capacity(
    n_samples: int,
    config: _SyntheticConfig,
    *,
    name: str,
) -> None:
    required_rank = max(
        config.n_shared + config.n_predictor_specific,
        config.n_shared + config.n_response_specific,
    )
    if n_samples <= required_rank:
        raise ValueError(
            f"{name} must exceed the largest declared centered latent rank ({required_rank})."
        )


def _draw_synthetic_loadings(
    rng: np.random.Generator,
    config: _SyntheticConfig,
) -> _SyntheticLoadings:
    x_basis = _orthonormal_columns(
        rng,
        n_rows=config.n_features,
        n_columns=config.n_shared + config.n_predictor_specific,
    )
    y_basis = _orthonormal_columns(
        rng,
        n_rows=config.n_targets,
        n_columns=config.n_shared + config.n_response_specific,
    )
    return _SyntheticLoadings(
        x_shared_loadings=x_basis[:, : config.n_shared],
        x_predictor_specific_loadings=x_basis[:, config.n_shared :],
        y_shared_loadings=y_basis[:, : config.n_shared],
        y_response_specific_loadings=y_basis[:, config.n_shared :],
    )


def _draw_dataset_block(
    rng: np.random.Generator,
    config: _SyntheticConfig,
    loadings: _SyntheticLoadings,
    *,
    n_samples: int,
    sample_prefix: str,
    split_role: str,
    generator_name: str,
) -> PiPLSDataset:
    shared_scores = _draw_standardized_scores(
        rng,
        n_samples=n_samples,
        n_components=config.n_shared,
        distribution=config.shared_distribution,
    )
    predictor_specific_scores = _draw_standardized_scores(
        rng,
        n_samples=n_samples,
        n_components=config.n_predictor_specific,
        distribution=config.predictor_specific_distribution,
    )
    response_specific_scores = _draw_standardized_scores(
        rng,
        n_samples=n_samples,
        n_components=config.n_response_specific,
        distribution=config.response_specific_distribution,
    )

    x_signal_unscaled = _signal_block(
        shared_scores,
        config.shared_strengths,
        loadings.x_shared_loadings,
    ) + _signal_block(
        predictor_specific_scores,
        config.predictor_specific_strengths,
        loadings.x_predictor_specific_loadings,
    )
    y_signal_unscaled = _signal_block(
        shared_scores,
        config.shared_strengths,
        loadings.y_shared_loadings,
    ) + _signal_block(
        response_specific_scores,
        config.response_specific_strengths,
        loadings.y_response_specific_loadings,
    )

    x_noise = rng.normal(scale=config.x_noise, size=(n_samples, config.n_features))
    y_noise = rng.normal(scale=config.y_noise, size=(n_samples, config.n_targets))
    x_signal = x_signal_unscaled * config.feature_scale
    y_signal = y_signal_unscaled * config.target_scale
    x_noise = x_noise * config.feature_scale
    y_noise = y_noise * config.target_scale
    X = x_signal + x_noise
    Y = y_signal + y_noise

    truth = PiPLSSyntheticTruth(
        shared_scores=shared_scores,
        predictor_specific_scores=predictor_specific_scores,
        response_specific_scores=response_specific_scores,
        x_shared_loadings=loadings.x_shared_loadings,
        x_predictor_specific_loadings=loadings.x_predictor_specific_loadings,
        y_shared_loadings=loadings.y_shared_loadings,
        y_response_specific_loadings=loadings.y_response_specific_loadings,
        x_signal=x_signal,
        y_signal=y_signal,
        x_noise=x_noise,
        y_noise=y_noise,
        feature_scale=config.feature_scale,
        target_scale=config.target_scale,
        shared_strengths=config.shared_strengths,
        predictor_specific_strengths=config.predictor_specific_strengths,
        response_specific_strengths=config.response_specific_strengths,
    )
    metadata: Mapping[str, object] = {
        "schema_version": 1,
        "generator": generator_name,
        "split_role": split_role,
        "random_state": config.random_state,
        "latent_dimensions": {
            "shared": config.n_shared,
            "predictor_specific": config.n_predictor_specific,
            "response_specific": config.n_response_specific,
        },
        "distributions": {
            "shared": config.shared_distribution,
            "predictor_specific": config.predictor_specific_distribution,
            "response_specific": config.response_specific_distribution,
        },
        "noise": {"X": config.x_noise, "Y": config.y_noise},
    }
    provenance = {
        "source": f"generated:pipls.datasets.{generator_name}",
        "license": "BSD-3-Clause",
        "citation": "Pi-PLS Python package deterministic synthetic generator",
        "version": "1",
    }
    return PiPLSDataset(
        X=X,
        Y=Y,
        feature_names=tuple(f"x_{index:03d}" for index in range(config.n_features)),
        target_names=tuple(f"y_{index:03d}" for index in range(config.n_targets)),
        sample_ids=tuple(f"{sample_prefix}_{index:04d}" for index in range(n_samples)),
        provenance=provenance,
        metadata=metadata,
        truth=truth,
    )


def _signal_block(
    scores: FloatArray,
    strengths: FloatArray,
    loadings: FloatArray,
) -> FloatArray:
    return np.asarray((scores * strengths) @ loadings.T, dtype=np.float64)


def _draw_standardized_scores(
    rng: np.random.Generator,
    *,
    n_samples: int,
    n_components: int,
    distribution: Distribution,
) -> FloatArray:
    if n_components == 0:
        return np.empty((n_samples, 0), dtype=np.float64)
    if distribution == "normal":
        scores = rng.normal(size=(n_samples, n_components))
    else:
        scores = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(n_samples, n_components))
    scores -= scores.mean(axis=0, keepdims=True)
    scales = scores.std(axis=0, ddof=1)
    if np.any(scales == 0.0):
        raise RuntimeError("Synthetic latent score generation produced a zero-variance column.")
    scores /= scales
    return np.asarray(scores, dtype=np.float64)


def _orthonormal_columns(
    rng: np.random.Generator,
    *,
    n_rows: int,
    n_columns: int,
) -> FloatArray:
    if n_columns == 0:
        return np.empty((n_rows, 0), dtype=np.float64)
    basis, triangular = np.linalg.qr(
        rng.normal(size=(n_rows, n_columns)),
        mode="reduced",
    )
    signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    return np.asarray(basis * signs, dtype=np.float64)


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


def _validate_truth(
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
    if not isinstance(truth, PiPLSSyntheticTruth):
        raise TypeError(
            "truth must be PiPLSSyntheticTruth or PiPLSLatentGeometryTruth."
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


def _positive_integer(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _resolve_positive_vector(value: NumericSpec, *, length: int, name: str) -> FloatArray:
    if length == 0:
        if np.isscalar(value):
            return np.empty(0, dtype=np.float64)
        array = np.asarray(value, dtype=np.float64)
        if array.size != 0:
            raise ValueError(f"{name} must be empty when its latent rank is zero.")
        return np.empty(0, dtype=np.float64)
    if np.isscalar(value):
        array = np.full(length, value, dtype=np.float64)
    else:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 1 or array.shape[0] != length:
            raise ValueError(f"{name} must be a scalar or contain exactly {length} values.")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} values must be positive and finite.")
    return np.asarray(array, dtype=np.float64)


def _resolve_noise(value: NoiseSpec) -> tuple[float, float]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("noise must be a scalar or a two-tuple (x_noise, y_noise).")
        x_noise, y_noise = (float(value[0]), float(value[1]))
    else:
        x_noise = y_noise = float(value)
    if not np.isfinite(x_noise) or not np.isfinite(y_noise):
        raise ValueError("noise values must be finite.")
    if x_noise < 0.0 or y_noise < 0.0:
        raise ValueError("noise values must be nonnegative.")
    return x_noise, y_noise


def _validated_seed(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("random_state must be an integer.")
    seed = int(value)
    if seed < 0 or seed > _UINT32_MAX:
        raise ValueError(f"random_state must lie in [0, {_UINT32_MAX}].")
    return seed
