"""Compare Pi-PLS and ordinary PLS component paths on the reference datasets."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _support.pls_component_path import evaluate_pls_component_path
from sklearn.model_selection import KFold

from pipls import PiPLSRegression, PiPLSSearchCV

DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "pls_path_comparison"
CV = KFold(n_splits=5, shuffle=True, random_state=0)

for dataset, path_search in (
    ("pulp", PiPLSSearchCV(cv=CV)),
    ("sugarcane", PiPLSSearchCV(cv=CV)),
    (
        "tobacco",
        PiPLSSearchCV(
            estimator=PiPLSRegression(
                n_components=1,
                predictor_rank=1,
                svd_solver="full",
            ),
            n_jobs=1,
            cv=CV,
        ),
    ),
):
    X = pd.read_csv(DATASETS_DIR / dataset / "X.csv")
    Y = pd.read_csv(DATASETS_DIR / dataset / "Y.csv")

    path_search.fit(X, Y)
    pipls_path = path_search.component_path_
    pls_path = evaluate_pls_component_path(
        X,
        Y,
        max_n_components=int(pipls_path.n_components[-1]),
        cv=CV,
    )
    if not np.array_equal(pipls_path.n_components, pls_path.n_components):
        raise RuntimeError("Pi-PLS and PLS paths must contain the same component counts.")

    figure, axis = plt.subplots(
        figsize=(8.0, 5.0),
        layout="constrained",
    )
    axis.errorbar(
        pipls_path.n_components,
        pipls_path.cv_mse_mean,
        yerr=pipls_path.cv_mse_standard_error,
        fmt="o-",
        capsize=4,
        label=r"$\Pi$-PLS",
    )
    for n_components, cv_mse, predictor_rank in zip(
        pipls_path.n_components,
        pipls_path.cv_mse_mean,
        pipls_path.predictor_rank,
        strict=True,
    ):
        axis.annotate(
            rf"$r_\pi={int(predictor_rank)}$",
            (n_components, cv_mse),
            xytext=(0, 8),
            textcoords="offset points",
            horizontalalignment="center",
        )
    axis.errorbar(
        pls_path.n_components,
        pls_path.cv_mse_mean,
        yerr=pls_path.cv_mse_standard_error,
        fmt="s--",
        capsize=4,
        label=f"PLS ({pls_path.algorithm})",
    )
    upper = max(
        float(np.max(pipls_path.cv_mse_mean + pipls_path.cv_mse_standard_error)),
        float(np.max(pls_path.cv_mse_mean + pls_path.cv_mse_standard_error)),
    )
    axis.set_title(
        rf"{dataset.capitalize()} $\Pi$-PLS and PLS component-path comparison"
    )
    axis.set_xlabel("Number of response components")
    axis.set_ylabel("Mean response-standardized CV-MSE (±1 SE)")
    axis.set_xticks(pipls_path.n_components)
    axis.set_ylim(0.0, max(1.0, 1.05 * upper))
    axis.margins(x=0.05)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(RESULTS_DIR / f"{dataset}_component_path_comparison.pdf")
    plt.close(figure)

    print(f"{dataset.capitalize()}: X shape={X.shape}, Y shape={Y.shape}")

print(f"Wrote results to {RESULTS_DIR}")
