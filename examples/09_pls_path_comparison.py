"""Compare Pi-PLS and ordinary PLS component paths on the reference datasets."""

from pathlib import Path

import pandas as pd
from _support.plot_component_path import plot_component_path_comparison
from _support.pls_component_path import evaluate_pls_component_path

from pipls import PiPLSPathCV, PiPLSRegression

DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "pls_path_comparison"

for dataset, path_search in (
    ("pulp", PiPLSPathCV(refit=False)),
    ("sugarcane", PiPLSPathCV(refit=False)),
    (
        "tobacco",
        PiPLSPathCV(
            estimator=PiPLSRegression(svd_solver="full"),
            refit=False,
            n_jobs=1,
        ),
    ),
):
    X = pd.read_csv(DATASETS_DIR / dataset / "X.csv")
    Y = pd.read_csv(DATASETS_DIR / dataset / "Y.csv")
    path_search.fit(X, Y)

    pipls_csv = RESULTS_DIR / f"{dataset}_pipls_component_path.csv"
    pls_csv = RESULTS_DIR / f"{dataset}_pls_component_path.csv"
    pd.DataFrame(path_search.component_path_results_).to_csv(pipls_csv, index=False)
    evaluate_pls_component_path(
        X,
        Y,
        max_n_components=int(path_search.n_components_values_[-1]),
    ).to_csv(pls_csv, index=False)
    plot_component_path_comparison(
        pipls_csv,
        pls_csv,
        RESULTS_DIR / f"{dataset}_component_path_comparison.pdf",
        title=f"{dataset.capitalize()} component-path comparison",
    )

    print(f"{dataset.capitalize()}: X shape={X.shape}, Y shape={Y.shape}")

print(f"Wrote results to {RESULTS_DIR}")
