"""Compare Π-PLS response policies and ordinary PLS on the reference datasets."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from _support.pls_component_path import PLSComponentPath, evaluate_pls_component_path
from numpy.typing import NDArray
from sklearn.model_selection import KFold

from pipls import PiPLSRegression, PiPLSSearchCV
from pipls.datasets import (
    PiPLSDataset,
    load_pulp,
    load_sugarcane,
    load_tobacco,
    make_pipls_regression,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "pls_path_comparison"
DATASET_NAMES = ("pulp", "sugarcane", "tobacco")
ResponseSubspace = Literal["cross_covariance", "least_squares"]
CVSplit = tuple[NDArray[np.intp], NDArray[np.intp]]


@dataclass(frozen=True)
class SyntheticStressSpec:
    """Fixed geometry for the near-saturated synthetic comparison case."""

    n_samples: int = 25
    n_features: int = 40
    n_targets: int = 10
    n_shared: int = 5
    n_predictor_specific: int = 15
    n_response_specific: int = 0
    noise: float = 0.3
    random_state: int = 0


SYNTHETIC_STRESS_SPEC = SyntheticStressSpec()


def _make_synthetic_stress_case() -> PiPLSDataset:
    """Generate the fixed Decision-0157 near-saturated stress case."""

    spec = SYNTHETIC_STRESS_SPEC
    return make_pipls_regression(
        n_samples=spec.n_samples,
        n_features=spec.n_features,
        n_targets=spec.n_targets,
        n_shared=spec.n_shared,
        n_predictor_specific=spec.n_predictor_specific,
        n_response_specific=spec.n_response_specific,
        noise=spec.noise,
        random_state=spec.random_state,
    )


@dataclass(frozen=True)
class DatasetComparison:
    """In-memory result of one matched PLS-family path comparison."""

    dataset: str
    cv_splits: list[CVSplit]
    cross_covariance_search: PiPLSSearchCV
    least_squares_search: PiPLSSearchCV
    pls_path: PLSComponentPath
    output_path: Path


def _load_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "pulp":
        return load_pulp(return_X_y=True)
    if dataset == "sugarcane":
        return load_sugarcane(return_X_y=True)
    if dataset == "tobacco":
        return load_tobacco(return_X_y=True)
    raise ValueError(f"Unknown reference dataset: {dataset!r}.")


def _make_search(
    dataset: str,
    *,
    response_subspace: ResponseSubspace,
    cv_splits: list[CVSplit],
) -> PiPLSSearchCV:
    estimator = PiPLSRegression(
        n_components=1,
        predictor_rank=1,
        response_subspace=response_subspace,
        svd_solver="full" if dataset == "tobacco" else "auto",
    )
    if dataset == "pulp":
        return PiPLSSearchCV(estimator=estimator, cv=cv_splits)
    if dataset == "sugarcane":
        return PiPLSSearchCV(
            estimator=estimator,
            search_method="adaptive",
            cv=cv_splits,
        )
    if dataset == "tobacco":
        return PiPLSSearchCV(
            estimator=estimator,
            search_method="adaptive",
            n_jobs=1,
            cv=cv_splits,
        )
    raise ValueError(f"Unknown reference dataset: {dataset!r}.")


def run_dataset(dataset: str) -> DatasetComparison:
    """Run one three-way matched-CV comparison and write its PDF figure."""

    X, Y = _load_dataset(dataset)
    cv_splits = [
        (np.asarray(train, dtype=np.intp), np.asarray(validation, dtype=np.intp))
        for train, validation in KFold(
            n_splits=5,
            shuffle=True,
            random_state=0,
        ).split(X, Y)
    ]

    cross_covariance_search = _make_search(
        dataset,
        response_subspace="cross_covariance",
        cv_splits=cv_splits,
    ).fit(X, Y)
    least_squares_search = _make_search(
        dataset,
        response_subspace="least_squares",
        cv_splits=cv_splits,
    ).fit(X, Y)

    cross_covariance_path = cross_covariance_search.component_path_
    least_squares_path = least_squares_search.component_path_
    if not np.array_equal(
        cross_covariance_path.n_components,
        least_squares_path.n_components,
    ):
        raise RuntimeError(
            "The two Π-PLS response-subspace paths must contain the same "
            "component counts."
        )

    pls_path = evaluate_pls_component_path(
        X,
        Y,
        max_n_components=int(cross_covariance_path.n_components[-1]),
        cv=cv_splits,
    )
    if not np.array_equal(cross_covariance_path.n_components, pls_path.n_components):
        raise RuntimeError(
            "Π-PLS and ordinary PLS paths must contain the same component counts."
        )

    figure, axis = plt.subplots(figsize=(8.4, 5.2), layout="constrained")
    axis.errorbar(
        cross_covariance_path.n_components,
        cross_covariance_path.cv_mse_mean,
        yerr=cross_covariance_path.cv_mse_std,
        fmt="o-",
        capsize=4,
        label=r"$\Pi$-PLS (cross-covariance)",
    )
    axis.errorbar(
        least_squares_path.n_components,
        least_squares_path.cv_mse_mean,
        yerr=least_squares_path.cv_mse_std,
        fmt="^--",
        capsize=4,
        label=r"$\Pi$-PLS (least squares)",
    )
    axis.errorbar(
        pls_path.n_components,
        pls_path.cv_mse_mean,
        yerr=pls_path.cv_mse_std,
        fmt="s:",
        capsize=4,
        label=f"PLS ({pls_path.algorithm})",
    )
    upper = max(
        float(np.max(cross_covariance_path.cv_mse_mean + cross_covariance_path.cv_mse_std)),
        float(np.max(least_squares_path.cv_mse_mean + least_squares_path.cv_mse_std)),
        float(np.max(pls_path.cv_mse_mean + pls_path.cv_mse_std)),
    )
    axis.set_title(f"{dataset.capitalize()} PLS-family component-path comparison")
    axis.set_xlabel("Nr of components")
    axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
    axis.set_xticks(cross_covariance_path.n_components)
    axis.set_ylim(0.0, max(1.0, 1.05 * upper))
    axis.margins(x=0.05)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    output_path = RESULTS_DIR / f"{dataset}_component_path_comparison.pdf"
    figure.savefig(output_path)
    plt.close(figure)

    print(f"{dataset.capitalize()}: X shape={X.shape}, Y shape={Y.shape}")
    print(f"  shared validation protocol: {len(cv_splits)} materialized folds")
    print(
        "  cross_covariance (peer-reviewed default) predictor ranks: "
        f"{cross_covariance_path.predictor_rank.tolist()}"
    )
    print(
        "  least_squares (software extension; not part of the peer-reviewed "
        "publication) predictor ranks: "
        f"{least_squares_path.predictor_rank.tolist()}"
    )
    print(
        "  path CV-MSE values are model-development evidence, not independent "
        "post-selection validation."
    )

    return DatasetComparison(
        dataset=dataset,
        cv_splits=cv_splits,
        cross_covariance_search=cross_covariance_search,
        least_squares_search=least_squares_search,
        pls_path=pls_path,
        output_path=output_path,
    )


def main(argv: Sequence[str] | None = None) -> dict[str, DatasetComparison]:
    """Run one requested dataset or the complete three-dataset comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=DATASET_NAMES,
        help="Run only one reference dataset; the default runs all three.",
    )
    args = parser.parse_args(argv)
    datasets = DATASET_NAMES if args.dataset is None else (args.dataset,)
    results = {dataset: run_dataset(dataset) for dataset in datasets}
    print(f"Wrote results to {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    main()
