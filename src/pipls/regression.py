"""Public fixed-parameter Pi-PLS regression estimator."""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array, check_is_fitted

from ._core import SVDSolver, fit_pipls_core
from ._sklearn_compat import _validate_estimator_data
from .decomposition import PiPLSDecomposition
from .exceptions import StatisticalSupportWarning
from .model_selection import _training_response_scale

FloatArray = NDArray[np.float64]
_MIN_TRUSTED_SAMPLES_PER_PREDICTOR_RANK = 4.0
_MAX_RANDOM_STATE = int(np.iinfo(np.uint32).max)


class PiPLSRegression(
    ClassNamePrefixFeaturesOutMixin,  # type: ignore[misc]
    TransformerMixin,  # type: ignore[misc]
    RegressorMixin,  # type: ignore[misc]
    MultiOutputMixin,  # type: ignore[misc]
    BaseEstimator,  # type: ignore[misc]
):
    r"""Pi-PLS regression for one fixed pair $(h, r_\pi)$.

    Parameters
    ----------
    n_components:
        Response-side latent dimension $h$.
    scale:
        If true, center and divide predictor and response columns by their
        training-sample standard deviations. If false, center without scaling.
    copy:
        If true, copy input arrays before preprocessing.
    predictor_rank:
        Explicit positive predictor truncation rank $r_\pi$.
    svd_solver:
        Predictor SVD policy. ``"full"`` uses the exact thin SVD,
        ``"randomized"`` always uses randomized truncated SVD, and ``"auto"``
        uses randomized SVD only for sufficiently large matrices and low retained
        rank. The response-side and coupling SVDs always remain exact.
    random_state:
        Integer seed in $[0, 2^{32}-1]$ used by randomized SVD. The default makes
        ``"auto"`` and ``"randomized"`` reproducible. ``None`` is accepted only
        with ``svd_solver="full"``.

    Notes
    -----
    This estimator performs no parameter selection or cross-validation. Use
    :class:`pipls.PiPLSPathCV` for the standard triangular model-selection path.
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        scale: bool = True,
        copy: bool = True,
        predictor_rank: int = 2,
        svd_solver: SVDSolver = "auto",
        random_state: int | None = 0,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.copy = copy
        self.predictor_rank = predictor_rank
        self.svd_solver = svd_solver
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> PiPLSRegression:
        """Fit one fixed Pi-PLS model."""

        self._validate_constructor_parameters()
        validated = _validate_estimator_data(
            self,
            X,
            y,
            reset=True,
            accept_sparse=False,
            dtype=np.float64,
            multi_output=True,
            y_numeric=True,
            ensure_min_samples=2,
            copy=self.copy,
        )
        X_checked, y_checked = cast(tuple[Any, Any], validated)
        X_array = np.asarray(X_checked, dtype=np.float64)
        y_array_raw = np.array(y_checked, dtype=np.float64, copy=self.copy)
        self._y_was_1d = y_array_raw.ndim == 1
        y_array = y_array_raw.reshape(-1, 1) if self._y_was_1d else y_array_raw

        self.n_targets_ = int(y_array.shape[1])
        if self.n_components > self.n_targets_:
            raise ValueError(
                "n_components must not exceed the number of response columns: "
                f"got n_components={self.n_components}, n_targets={self.n_targets_}."
            )

        n_samples, n_features = X_array.shape
        max_predictor_rank = min(n_features, n_samples - 1)
        predictor_rank = int(self.predictor_rank)
        if predictor_rank > max_predictor_rank:
            raise ValueError(
                "predictor_rank must satisfy predictor_rank <= min(n_features, "
                "n_samples - 1) after centering; "
                f"got predictor_rank={predictor_rank}, n_features={n_features}, "
                f"n_samples={n_samples}."
            )
        if n_samples < _MIN_TRUSTED_SAMPLES_PER_PREDICTOR_RANK * predictor_rank:
            warnings.warn(
                f"The fitted model has {n_samples / predictor_rank:.3g} samples per "
                "retained predictor-rank direction, below the recommended minimum of 4. "
                "The model may have poor statistical support; use external validation.",
                StatisticalSupportWarning,
                stacklevel=2,
            )
        self._fit_fixed_rank(
            X_array,
            y_array,
            predictor_rank=predictor_rank,
            max_predictor_rank=max_predictor_rank,
        )
        return self

    def predict(self, X: ArrayLike, copy: bool = True) -> FloatArray:
        """Predict responses in their original units.

        Parameters
        ----------
        X:
            Predictor matrix.
        copy:
            Whether validation may copy ``X``. This mirrors
            :class:`sklearn.cross_decomposition.PLSRegression`.
        """

        check_is_fitted(self, attributes=["coef_", "intercept_"])
        X_checked = cast(
            FloatArray,
            _validate_estimator_data(
                self,
                X,
                reset=False,
                accept_sparse=False,
                dtype=np.float64,
                copy=copy,
            ),
        )
        prediction = cast(
            FloatArray,
            np.asarray(X_checked, dtype=np.float64) @ self.coef_.T + self.intercept_,
        )
        if self._y_was_1d:
            return prediction[:, 0]
        return prediction

    def transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        copy: bool = True,
    ) -> FloatArray | tuple[FloatArray, FloatArray]:
        """Transform predictors, and optionally responses, to latent scores."""

        check_is_fitted(self, attributes=["P_", "Q_", "x_mean_", "y_mean_"])
        X_checked = cast(
            FloatArray,
            _validate_estimator_data(
                self,
                X,
                reset=False,
                accept_sparse=False,
                dtype=np.float64,
                copy=copy,
            ),
        )
        X_cs = (np.asarray(X_checked, dtype=np.float64) - self.x_mean_) / self.x_scale_
        x_scores = cast(FloatArray, X_cs @ self.P_)
        if y is None:
            return x_scores

        y_checked = check_array(
            y,
            ensure_2d=False,
            dtype=np.float64,
            ensure_min_samples=X_cs.shape[0],
            copy=copy,
        )
        y_array = np.asarray(y_checked, dtype=np.float64)
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)
        if y_array.shape[0] != X_cs.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples: "
                f"got {X_cs.shape[0]} and {y_array.shape[0]}."
            )
        if y_array.shape[1] != self.n_targets_:
            raise ValueError(
                "y has an incompatible number of targets: "
                f"expected {self.n_targets_}, got {y_array.shape[1]}."
            )
        y_cs = (y_array - self.y_mean_) / self.y_scale_
        y_scores = cast(FloatArray, y_cs @ self.Q_)
        return x_scores, y_scores

    def fit_transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Fit the model and return predictor and response scores.

        As for scikit-learn's ``PLSRegression``, supplying ``y`` returns the
        pair ``(x_scores, y_scores)``.
        """

        if y is None:
            raise ValueError("y is required to fit PiPLSRegression.")
        self.fit(X, y)
        return self.x_scores_.copy(), self.y_scores_.copy()

    def inverse_transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
    ) -> FloatArray | tuple[FloatArray, FloatArray]:
        """Reconstruct predictors, and optionally responses, from latent scores.

        Reconstruction is least-squares and is exact only when the retained
        latent spaces span the corresponding centered/scaled data spaces.
        """

        check_is_fitted(self, attributes=["x_loadings_", "y_loadings_"])
        x_scores = check_array(X, ensure_2d=True, dtype=np.float64)
        if x_scores.shape[1] != self.n_components:
            raise ValueError(
                "X has an incompatible number of latent components: "
                f"expected {self.n_components}, got {x_scores.shape[1]}."
            )
        X_original = (x_scores @ self.x_loadings_.T) * self.x_scale_ + self.x_mean_
        if y is None:
            return np.asarray(X_original, dtype=np.float64)

        y_scores = check_array(y, ensure_2d=False, dtype=np.float64)
        if y_scores.ndim == 1:
            y_scores = y_scores.reshape(-1, 1)
        if y_scores.shape[0] != x_scores.shape[0]:
            raise ValueError(
                "X and y scores must contain the same number of samples: "
                f"got {x_scores.shape[0]} and {y_scores.shape[0]}."
            )
        if y_scores.shape[1] != self.n_components:
            raise ValueError(
                "y has an incompatible number of latent components: "
                f"expected {self.n_components}, got {y_scores.shape[1]}."
            )
        y_original = (y_scores @ self.y_loadings_.T) * self.y_scale_ + self.y_mean_
        return (
            np.asarray(X_original, dtype=np.float64),
            np.asarray(y_original, dtype=np.float64),
        )

    def score(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
        """Return uniformly averaged $R^2$ in original response units."""

        return float(
            r2_score(
                y,
                self.predict(X),
                sample_weight=sample_weight,
                multioutput="uniform_average",
            )
        )

    def _more_tags(self) -> dict[str, bool]:
        """Legacy scikit-learn tags for releases before the Tags dataclasses."""

        return {"multioutput": True, "poor_score": True}

    def __sklearn_tags__(self) -> Any:
        """Declare multi-output regression and PLS-like score expectations."""

        parent = getattr(super(), "__sklearn_tags__", None)
        if parent is None:  # pragma: no cover - scikit-learn 1.4/1.5
            return self._more_tags()
        tags = parent()
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = True
        if tags.regressor_tags is not None:
            tags.regressor_tags.poor_score = True
        return tags


    def _fit_fixed_rank(
        self,
        X: FloatArray,
        y: FloatArray,
        *,
        predictor_rank: int,
        max_predictor_rank: int,
    ) -> None:
        self.x_mean_ = np.mean(X, axis=0)
        self.y_mean_ = np.mean(y, axis=0)
        self.response_scale_for_scoring_ = _training_response_scale(y)
        X -= self.x_mean_
        y -= self.y_mean_

        if self.scale:
            self.x_scale_ = _safe_sample_scale(X)
            self.y_scale_ = _safe_sample_scale(y)
            X /= self.x_scale_
            y /= self.y_scale_
        else:
            self.x_scale_ = np.ones(X.shape[1], dtype=np.float64)
            self.y_scale_ = np.ones(y.shape[1], dtype=np.float64)

        X_cs = X
        y_cs = y
        result = fit_pipls_core(
            X_cs,
            y_cs,
            predictor_rank=predictor_rank,
            n_components=self.n_components,
            svd_solver=self.svd_solver,
            random_state=self.random_state,
        )

        self.predictor_rank_ = predictor_rank
        self.max_predictor_rank_ = max_predictor_rank
        self.decomposition_ = PiPLSDecomposition._from_core_result(result)
        self.Pi_ = self.decomposition_.Pi
        self.C_ = self.decomposition_.C
        self.W_ = self.decomposition_.W
        self.P_ = self.decomposition_.P
        self.D_ = self.decomposition_.D
        self.dilation_ = self.decomposition_.dilation
        self.Q_ = self.decomposition_.Q
        self.x_rotations_ = self.P_
        self.y_rotations_ = self.Q_
        self.x_weights_ = self.P_
        self.y_weights_ = self.Q_
        self._n_features_out = self.n_components
        self.x_rank_ = result.x_rank
        self.x_rank_is_exact_ = result.x_rank_is_exact
        self.rank_tolerance_ = result.rank_tolerance
        self.svd_solver_ = result.predictor_svd_solver

        self.coef_matrix_ = (
            result.regression_map * self.y_scale_[None, :] / self.x_scale_[:, None]
        )
        self.coef_ = self.coef_matrix_.T
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_matrix_
        self.x_scores_ = X_cs @ self.P_
        self.y_scores_ = y_cs @ self.Q_
        x_loadings, _, _, _ = np.linalg.lstsq(self.x_scores_, X_cs, rcond=None)
        y_loadings, _, _, _ = np.linalg.lstsq(self.y_scores_, y_cs, rcond=None)
        self.x_loadings_ = np.asarray(x_loadings.T, dtype=np.float64)
        self.y_loadings_ = np.asarray(y_loadings.T, dtype=np.float64)

    def _validate_constructor_parameters(self) -> None:
        _validate_positive_int(self.n_components, name="n_components")
        _validate_positive_int(self.predictor_rank, name="predictor_rank")
        if self.n_components > self.predictor_rank:
            raise ValueError(
                "n_components must satisfy n_components <= predictor_rank; "
                f"got n_components={self.n_components}, "
                f"predictor_rank={self.predictor_rank}."
            )
        if not isinstance(self.scale, (bool, np.bool_)):
            raise ValueError(f"scale must be boolean; got {self.scale!r}.")
        if not isinstance(self.copy, (bool, np.bool_)):
            raise ValueError(f"copy must be boolean; got {self.copy!r}.")
        if not isinstance(self.svd_solver, str) or self.svd_solver not in (
            "full",
            "randomized",
            "auto",
        ):
            raise ValueError(
                'svd_solver must be "full", "randomized", or "auto"; '
                f"got {self.svd_solver!r}."
            )
        _validate_random_state(self.random_state, svd_solver=self.svd_solver)


def _validate_random_state(
    random_state: int | None,
    *,
    svd_solver: SVDSolver,
) -> None:
    if random_state is None:
        if svd_solver != "full":
            raise ValueError(
                'random_state=None is accepted only with svd_solver="full"; '
                f"got svd_solver={svd_solver!r}."
            )
        return
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(
        random_state,
        (int, np.integer),
    ):
        raise ValueError(
            "random_state must be None or an integer in "
            f"[0, {_MAX_RANDOM_STATE}]; got {random_state!r}."
        )
    value = int(random_state)
    if value < 0 or value > _MAX_RANDOM_STATE:
        raise ValueError(
            "random_state must be None or an integer in "
            f"[0, {_MAX_RANDOM_STATE}]; got {random_state!r}."
        )


def _safe_sample_scale(centered: FloatArray) -> FloatArray:
    scale = np.std(centered, axis=0, ddof=1)
    return cast(FloatArray, np.where(scale == 0.0, 1.0, scale))


def _validate_positive_int(value: object, *, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    if int(value) < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
