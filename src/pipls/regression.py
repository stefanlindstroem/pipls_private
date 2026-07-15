"""Public fixed-rank Pi-PLS estimator."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._core import fit_pipls_core

FloatArray = NDArray[np.float64]


class PiPLSRegression(TransformerMixin, RegressorMixin, BaseEstimator):  # type: ignore[misc]
    r"""Pi-PLS regression with an explicitly fixed predictor rank.

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
        Explicit predictor truncation rank $r_\pi$. Only integer values are
        supported in this implementation phase.
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        scale: bool = True,
        copy: bool = True,
        predictor_rank: int = 2,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.copy = copy
        self.predictor_rank = predictor_rank

    def fit(self, X: ArrayLike, y: ArrayLike) -> PiPLSRegression:
        """Fit the fixed-rank Pi-PLS model."""

        X_checked, y_checked = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            multi_output=True,
            y_numeric=True,
        )
        self.n_features_in_ = int(X_checked.shape[1])
        X_array = np.array(X_checked, dtype=np.float64, copy=self.copy)
        y_array_raw = np.asarray(y_checked, dtype=np.float64)
        self._y_was_1d = y_array_raw.ndim == 1
        y_array = np.array(
            y_array_raw.reshape(-1, 1) if self._y_was_1d else y_array_raw,
            dtype=np.float64,
            copy=self.copy,
        )

        self._validate_constructor_parameters()
        self.n_targets_ = int(y_array.shape[1])

        self.x_mean_ = np.mean(X_array, axis=0)
        self.y_mean_ = np.mean(y_array, axis=0)
        X_centered = X_array - self.x_mean_
        y_centered = y_array - self.y_mean_

        if self.scale:
            self.x_scale_ = _safe_sample_scale(X_centered)
            self.y_scale_ = _safe_sample_scale(y_centered)
        else:
            self.x_scale_ = np.ones(X_array.shape[1], dtype=np.float64)
            self.y_scale_ = np.ones(y_array.shape[1], dtype=np.float64)

        X_cs = X_centered / self.x_scale_
        y_cs = y_centered / self.y_scale_

        result = fit_pipls_core(
            X_cs,
            y_cs,
            predictor_rank=self.predictor_rank,
            n_components=self.n_components,
        )

        self.predictor_rank_ = int(self.predictor_rank)
        self.Pi_ = result.Pi
        self.C_ = result.C
        self.W_ = result.W
        self.P_ = result.P
        self.D_ = result.D
        self.dilation_ = np.diag(result.D).copy()
        self.Q_ = result.Q
        self.x_rotations_ = self.P_
        self.y_rotations_ = self.Q_
        self.x_rank_ = result.x_rank
        self.rank_tolerance_ = result.rank_tolerance

        self.coef_matrix_ = (
            result.regression_map * self.y_scale_[None, :] / self.x_scale_[:, None]
        )
        self.coef_ = self.coef_matrix_.T
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_matrix_

        self.x_scores_ = X_cs @ self.P_
        self.y_scores_ = y_cs @ self.Q_
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict responses in their original units."""

        check_is_fitted(self, attributes=["coef_", "intercept_"])
        X_checked = _check_predictor_matrix(X, n_features=self.n_features_in_)
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
    ) -> FloatArray | tuple[FloatArray, FloatArray]:
        """Transform predictors, and optionally responses, to latent scores."""

        check_is_fitted(self, attributes=["P_", "Q_", "x_mean_", "y_mean_"])
        X_checked = _check_predictor_matrix(X, n_features=self.n_features_in_)
        X_cs = (np.asarray(X_checked, dtype=np.float64) - self.x_mean_) / self.x_scale_
        x_scores = cast(FloatArray, X_cs @ self.P_)
        if y is None:
            return x_scores

        y_checked = check_array(
            y,
            ensure_2d=False,
            dtype=np.float64,
            ensure_min_samples=X_cs.shape[0],
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

    def _validate_constructor_parameters(self) -> None:
        _validate_positive_int(self.n_components, name="n_components")
        _validate_positive_int(self.predictor_rank, name="predictor_rank")
        if self.n_components > self.predictor_rank:
            raise ValueError(
                "n_components must satisfy n_components <= predictor_rank; "
                f"got n_components={self.n_components}, predictor_rank={self.predictor_rank}."
            )
        if not isinstance(self.scale, (bool, np.bool_)):
            raise ValueError(f"scale must be boolean; got {self.scale!r}.")
        if not isinstance(self.copy, (bool, np.bool_)):
            raise ValueError(f"copy must be boolean; got {self.copy!r}.")


def _check_predictor_matrix(X: ArrayLike, *, n_features: int) -> FloatArray:
    checked = check_array(X, accept_sparse=False, dtype=np.float64)
    array = np.asarray(checked, dtype=np.float64)
    if array.shape[1] != n_features:
        raise ValueError(
            "X has an incompatible number of features: "
            f"expected {n_features}, got {array.shape[1]}."
        )
    return array


def _safe_sample_scale(centered: FloatArray) -> FloatArray:
    if centered.shape[0] <= 1:
        return np.ones(centered.shape[1], dtype=np.float64)
    scale = np.std(centered, axis=0, ddof=1)
    return np.where(scale == 0.0, 1.0, scale).astype(np.float64, copy=False)


def _validate_positive_int(value: Any, *, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    if int(value) < 1:
        raise ValueError(f"{name} must be at least 1; got {value!r}.")
