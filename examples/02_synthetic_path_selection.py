"""Select a paired-mode count and fit Pi-PLS on deterministic synthetic data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --8<-- [start:import-synthetic-kfold]
from sklearn.model_selection import KFold

# --8<-- [end:import-synthetic-kfold]
from pipls import PiPLSSearchCV
from pipls.datasets import make_pipls_train_test
from pipls.inspection import prediction_diagnostics

ANALYSIS_DIR = Path(__file__).resolve().parent / "results" / "synthetic_tutorial"
CHOSEN_N_COMPONENTS = 2
# --8<-- [start:define-synthetic-cv]
CV = KFold(n_splits=5, shuffle=True, random_state=0)
# --8<-- [end:define-synthetic-cv]

# --8<-- [start:generate-synthetic-data]
train, test = make_pipls_train_test(
    n_train=120,
    n_test=60,
    n_features=8,
    n_targets=3,
    n_shared=2,
    n_predictor_specific=2,
    n_response_specific=1,
    shared_strength=(2.5, 1.5),
    noise=(0.2, 0.25),
    random_state=0,
)
# --8<-- [end:generate-synthetic-data]

# --8<-- [start:fit-synthetic-model]
search = PiPLSSearchCV(cv=CV).fit(train.X, train.Y)
model = search.refit(
    train.X,
    train.Y,
    n_components=CHOSEN_N_COMPONENTS,
)
# --8<-- [end:fit-synthetic-model]

# --8<-- [start:inspect-synthetic-selection]
selection = model.selection_
path = search.component_path_
rank_profile = search.predictor_rank_profile(selection.n_components)
# --8<-- [end:inspect-synthetic-selection]

# --8<-- [start:evaluate-synthetic-predictions]
test_predictions = model.predict(test.X)
diagnostics = prediction_diagnostics(
    test.Y,
    test_predictions,
    prediction_kind="external test predictions",
)
# --8<-- [end:evaluate-synthetic-predictions]

# --8<-- [start:plot-synthetic-component-path]
figure, axis = plt.subplots(figsize=(7.0, 4.5), layout="constrained")
axis.errorbar(
    path.n_components,
    path.cv_mse_mean,
    yerr=path.cv_mse_std,
    fmt="o-",
    capsize=4,
)
axis.scatter(
    [selection.n_components],
    [selection.cv_mse_mean],
    marker="D",
    s=70,
    label=f"Chosen: {selection.n_components} components",
    zorder=3,
)
axis.set_xlabel("Number of components")
axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
axis.set_title(r"Synthetic $\Pi$-PLS component path")
axis.set_xticks(path.n_components)
upper = float(np.max(path.cv_mse_mean + path.cv_mse_std))
axis.set_ylim(0.0, max(1.0, 1.05 * upper))
axis.grid(axis="y", alpha=0.25)
axis.legend()
figure.savefig(ANALYSIS_DIR / "component_path.pdf")
plt.close(figure)
# --8<-- [end:plot-synthetic-component-path]

# --8<-- [start:plot-synthetic-rank-profile]
figure, axis = plt.subplots(figsize=(7.0, 4.5), layout="constrained")
axis.errorbar(
    rank_profile.predictor_rank,
    rank_profile.cv_mse_mean,
    yerr=rank_profile.cv_mse_std,
    fmt="o-",
    capsize=4,
)
axis.scatter(
    [rank_profile.selection.predictor_rank],
    [rank_profile.selection.cv_mse_mean],
    marker="D",
    s=70,
    label=f"CV-MSE minimum: rank {rank_profile.selection.predictor_rank}",
    zorder=3,
)
axis.set_xlabel("Predictor rank")
axis.set_ylabel("Mean response-standardized CV-MSE (±1 SD)")
axis.set_title(
    rf"Synthetic $\Pi$-PLS predictor-rank profile at "
    f"{selection.n_components} components"
)
axis.set_xticks(rank_profile.predictor_rank)
upper = float(
    np.max(rank_profile.cv_mse_mean + rank_profile.cv_mse_std)
)
axis.set_ylim(0.0, max(1.0, 1.05 * upper))
axis.grid(axis="y", alpha=0.25)
axis.legend()
figure.savefig(ANALYSIS_DIR / "predictor_rank_profile.pdf")
plt.close(figure)
# --8<-- [end:plot-synthetic-rank-profile]

# --8<-- [start:plot-synthetic-predictions]
figure, axis = plt.subplots(figsize=(6.2, 5.0), layout="constrained")
for response, name in enumerate(test.target_names):
    axis.scatter(
        diagnostics.observed_standardized[:, response],
        diagnostics.predicted_standardized[:, response],
        label=name,
        alpha=0.75,
    )
values = np.concatenate(
    [
        diagnostics.observed_standardized.ravel(),
        diagnostics.predicted_standardized.ravel(),
    ]
)
lower = float(values.min())
upper = float(values.max())
margin = 0.05 * (upper - lower) if upper > lower else 1.0
limits = (lower - margin, upper + margin)
axis.plot(limits, limits, linewidth=1.0, linestyle="--", color="0.35")
axis.set_xlim(limits)
axis.set_ylim(limits)
axis.set_xlabel("Observed response (standardized)")
axis.set_ylabel("Predicted response (standardized)")
axis.set_title(rf"Synthetic $\Pi$-PLS — {diagnostics.prediction_kind}")
axis.legend(title="Response")
figure.savefig(ANALYSIS_DIR / "observed_vs_predicted.pdf")
plt.close(figure)
# --8<-- [end:plot-synthetic-predictions]

print("Synthetic Pi-PLS path-selection example")
print(f"Training data: X{train.X.shape}, Y{train.Y.shape}")
print(f"Independent test data: X{test.X.shape}, Y{test.Y.shape}")
print(
    "Known latent structure: "
    f"{train.truth.n_shared} shared directions and "
    f"{train.truth.n_predictor_specific} predictor-specific directions"
)
print(
    "Selected fixed model: "
    f"n_components={selection.n_components}, "
    f"predictor_rank={selection.predictor_rank}"
)
print(f"External-test R^2: {model.score(test.X, test.Y):.3f}")
print(f"Wrote PDF figures to {ANALYSIS_DIR}")
