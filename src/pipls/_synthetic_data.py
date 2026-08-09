"""Deterministic synthetic data generators for Π-PLS."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from ._dataset_types import (
    FloatArray,
    PiPLSDataset,
    PiPLSLatentGeometryTruth,
    PiPLSRegressionTruth,
)

Distribution: TypeAlias = Literal["normal", "uniform"]
NumericSpec: TypeAlias = float | Sequence[float]
NoiseSpec: TypeAlias = float | tuple[float, float]

_UINT32_MAX = 2**32 - 1


@dataclass(frozen=True)
class _RegressionLoadings:
    x_shared_loadings: FloatArray
    x_predictor_specific_loadings: FloatArray
    y_shared_loadings: FloatArray
    y_response_specific_loadings: FloatArray


@dataclass(frozen=True)
class _RegressionGeneratorConfig:
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
    generator used by existing examples and validation workflows.

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
    r"""Generate one deterministic Π-PLS latent-structure dataset.

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
    config = _validated_regression_config(
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
    loadings = _draw_regression_loadings(rng, config)
    return _draw_regression_block(
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
    config = _validated_regression_config(
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
    loadings = _draw_regression_loadings(rng, config)
    train = _draw_regression_block(
        rng,
        config,
        loadings,
        n_samples=n_train,
        sample_prefix="train",
        split_role="train",
        generator_name="make_pipls_train_test",
    )
    test = _draw_regression_block(
        rng,
        config,
        loadings,
        n_samples=n_test,
        sample_prefix="test",
        split_role="test",
        generator_name="make_pipls_train_test",
    )
    return train, test


def _validated_regression_config(
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
) -> _RegressionGeneratorConfig:
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
    return _RegressionGeneratorConfig(
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
    config: _RegressionGeneratorConfig,
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


def _draw_regression_loadings(
    rng: np.random.Generator,
    config: _RegressionGeneratorConfig,
) -> _RegressionLoadings:
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
    return _RegressionLoadings(
        x_shared_loadings=x_basis[:, : config.n_shared],
        x_predictor_specific_loadings=x_basis[:, config.n_shared :],
        y_shared_loadings=y_basis[:, : config.n_shared],
        y_response_specific_loadings=y_basis[:, config.n_shared :],
    )


def _draw_regression_block(
    rng: np.random.Generator,
    config: _RegressionGeneratorConfig,
    loadings: _RegressionLoadings,
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

    x_signal_unscaled = _latent_signal(
        shared_scores,
        config.shared_strengths,
        loadings.x_shared_loadings,
    ) + _latent_signal(
        predictor_specific_scores,
        config.predictor_specific_strengths,
        loadings.x_predictor_specific_loadings,
    )
    y_signal_unscaled = _latent_signal(
        shared_scores,
        config.shared_strengths,
        loadings.y_shared_loadings,
    ) + _latent_signal(
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

    truth = PiPLSRegressionTruth(
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


def _latent_signal(
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
