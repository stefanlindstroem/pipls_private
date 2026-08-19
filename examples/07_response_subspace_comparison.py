"""Compare the two Π-PLS response-subspace policies on shared CV splits."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

from pipls import PiPLSRegression, PiPLSSearchCV
from pipls.datasets import load_pulp

OUTPUT_PATH = (
    Path(__file__).resolve().parent / "results" / "response_subspace_comparison.pdf"
)

X, Y = load_pulp(return_X_y=True)
CV_SPLITS = list(KFold(n_splits=5, shuffle=True, random_state=0).split(X, Y))

cross_covariance_template = PiPLSRegression(
    n_components=1,
    predictor_rank=1,
    response_subspace="cross_covariance",
)
least_squares_template = PiPLSRegression(
    n_components=1,
    predictor_rank=1,
    response_subspace="least_squares",
)

cross_covariance_search = PiPLSSearchCV(
    estimator=cross_covariance_template,
    cv=CV_SPLITS,
).fit(X, Y)
least_squares_search = PiPLSSearchCV(
    estimator=least_squares_template,
    cv=CV_SPLITS,
).fit(X, Y)

cross_covariance_selection = cross_covariance_search.select(
    rule="minimum_cv_mse"
)
least_squares_selection = least_squares_search.select(
    rule="minimum_cv_mse"
)

cross_covariance_path = cross_covariance_search.component_path_
least_squares_path = least_squares_search.component_path_
if not np.array_equal(
    cross_covariance_path.n_components,
    least_squares_path.n_components,
):
    raise RuntimeError("The matched searches must evaluate the same component counts.")

figure, axis = plt.subplots(figsize=(8.0, 5.0), layout="constrained")
axis.errorbar(
    cross_covariance_path.n_components,
    cross_covariance_path.cv_mse_mean,
    yerr=cross_covariance_path.cv_mse_std,
    fmt="o-",
    capsize=4,
    label="cross_covariance — peer-reviewed default",
)
axis.errorbar(
    least_squares_path.n_components,
    least_squares_path.cv_mse_mean,
    yerr=least_squares_path.cv_mse_std,
    fmt="s--",
    capsize=4,
    label="least_squares — software extension",
)
axis.scatter(
    [cross_covariance_selection.n_components],
    [cross_covariance_selection.cv_mse_mean],
    marker="D",
    s=70,
    zorder=3,
)
axis.scatter(
    [least_squares_selection.n_components],
    [least_squares_selection.cv_mse_mean],
    marker="D",
    s=70,
    zorder=3,
)
upper = max(
    float(
        np.max(
            cross_covariance_path.cv_mse_mean
            + cross_covariance_path.cv_mse_std
        )
    ),
    float(
        np.max(
            least_squares_path.cv_mse_mean
            + least_squares_path.cv_mse_std
        )
    ),
)
axis.set_xlabel("Number of response components")
axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
axis.set_title(r"Pulp $\Pi$-PLS response-subspace comparison")
axis.set_xticks(cross_covariance_path.n_components)
axis.set_ylim(0.0, max(1.0, 1.05 * upper))
axis.grid(axis="y", alpha=0.25)
axis.legend()
figure.savefig(OUTPUT_PATH)
plt.close(figure)

print("Π-PLS response-subspace comparison")
print(f"Shared validation protocol: {len(CV_SPLITS)} materialized folds")
print(
    "cross_covariance (peer-reviewed default): "
    f"n_components={cross_covariance_selection.n_components}, "
    f"predictor_rank={cross_covariance_selection.predictor_rank}, "
    f"CV-MSE={cross_covariance_selection.cv_mse_mean:.6f}"
)
print(
    "least_squares (software extension; not part of the peer-reviewed "
    "publication): "
    f"n_components={least_squares_selection.n_components}, "
    f"predictor_rank={least_squares_selection.predictor_rank}, "
    f"CV-MSE={least_squares_selection.cv_mse_mean:.6f}"
)
print(
    "These are model-development CV results, not independent "
    "post-selection validation."
)
print(f"Wrote PDF figure to {OUTPUT_PATH}")
