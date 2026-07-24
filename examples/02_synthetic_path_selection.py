"""Select and fit a Pi-PLS model on deterministic synthetic data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipls import PiPLSPathCV, PiPLSRegression
from pipls.datasets import make_pipls_train_test
from pipls.inspection import prediction_diagnostics

ANALYSIS_DIR = Path(__file__).resolve().parent / "results" / "synthetic_tutorial"
CHOSEN_N_COMPONENTS = 2

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

# --8<-- [start:evaluate-synthetic-path]
search = PiPLSPathCV().fit(train.X, train.Y)
path = search.component_path_
selected = path.for_n_components(CHOSEN_N_COMPONENTS)
# --8<-- [end:evaluate-synthetic-path]

# --8<-- [start:plot-synthetic-component-path]
figure, axis = plt.subplots(figsize=(7.0, 4.5), layout="constrained")
axis.errorbar(
    path.n_components,
    path.cv_mse_mean,
    yerr=path.cv_mse_fold_sd,
    fmt="o-",
    capsize=4,
)
axis.scatter(
    [selected.n_components],
    [selected.cv_mse_mean],
    marker="D",
    s=70,
    label=f"Chosen: {selected.n_components} components",
    zorder=3,
)
axis.set_xlabel("Number of components")
axis.set_ylabel("Response-standardized CV-MSE")
axis.set_title("Synthetic Pi-PLS component path")
axis.set_xticks(path.n_components)
axis.grid(axis="y", alpha=0.25)
axis.legend()
figure.savefig(ANALYSIS_DIR / "component_path.pdf")
plt.close(figure)
# --8<-- [end:plot-synthetic-component-path]

# --8<-- [start:plot-synthetic-rank-profile]
rank_profile = search.predictor_rank_profile(selected.n_components)

figure, axis = plt.subplots(figsize=(7.0, 4.5), layout="constrained")
axis.errorbar(
    rank_profile.predictor_rank,
    rank_profile.cv_mse_mean,
    yerr=rank_profile.cv_mse_fold_sd,
    fmt="o-",
    capsize=4,
)
axis.scatter(
    [rank_profile.selected.predictor_rank],
    [rank_profile.selected.cv_mse_mean],
    marker="D",
    s=70,
    label=f"CV-MSE minimum: rank {rank_profile.selected.predictor_rank}",
    zorder=3,
)
axis.set_xlabel("Predictor rank")
axis.set_ylabel("Response-standardized CV-MSE")
axis.set_title(f"Predictor-rank profile at {selected.n_components} components")
axis.set_xticks(rank_profile.predictor_rank)
axis.grid(axis="y", alpha=0.25)
axis.legend()
figure.savefig(ANALYSIS_DIR / "predictor_rank_profile.pdf")
plt.close(figure)
# --8<-- [end:plot-synthetic-rank-profile]

# --8<-- [start:fit-predict-synthetic-model]
model = PiPLSRegression(
    n_components=selected.n_components,
    predictor_rank=selected.predictor_rank,
).fit(train.X, train.Y)

test_predictions = model.predict(test.X)
diagnostics = prediction_diagnostics(
    test.Y,
    test_predictions,
    prediction_kind="external test predictions",
)

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
axis.set_title(f"Synthetic external-test predictions\n{diagnostics.prediction_kind}")
axis.legend(title="Response")
figure.savefig(ANALYSIS_DIR / "observed_vs_predicted.pdf")
plt.close(figure)
# --8<-- [end:fit-predict-synthetic-model]

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
    f"n_components={selected.n_components}, "
    f"predictor_rank={selected.predictor_rank}"
)
print(f"External-test R^2: {model.score(test.X, test.Y):.3f}")
print(f"Wrote PDF figures to {ANALYSIS_DIR}")
