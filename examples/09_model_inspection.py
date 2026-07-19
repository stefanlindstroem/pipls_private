"""Inspect a fixed Pi-PLS model on a deterministic external test set."""

from pathlib import Path

import matplotlib.pyplot as plt

from pipls import PiPLSRegression
from pipls.datasets import make_pipls_train_test
from pipls.inspection import pipls_display_factors, prediction_diagnostics
from pipls.plotting import plot_pipls_decomposition, plot_prediction_diagnostics

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "model_inspection"
DECOMPOSITION_PDF = RESULTS_DIR / "pipls_decomposition.pdf"
PREDICTION_PDF = RESULTS_DIR / "prediction_diagnostics.pdf"

train, test = make_pipls_train_test(
    n_train=100,
    n_test=40,
    n_features=12,
    n_targets=3,
    n_shared=2,
    n_predictor_specific=2,
    n_response_specific=1,
    shared_strength=(2.0, 1.0),
    noise=(0.15, 0.2),
    random_state=0,
)

model = PiPLSRegression(n_components=2, predictor_rank=4).fit(train.X, train.Y)
factors = pipls_display_factors(model.decomposition_)
diagnostics = prediction_diagnostics(
    test.Y,
    model.predict(test.X),
    prediction_kind="external test predictions",
)

feature_names = [f"x{index + 1}" for index in range(train.n_features)]
target_names = [f"y{index + 1}" for index in range(train.n_targets)]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
decomposition_figure, _ = plot_pipls_decomposition(
    factors,
    predictor_style="bar",
    predictor_names=feature_names,
    response_names=target_names,
    title="Synthetic Pi-PLS decomposition",
)
decomposition_figure.savefig(DECOMPOSITION_PDF)
plt.close(decomposition_figure)

prediction_figure, _ = plot_prediction_diagnostics(
    diagnostics,
    response_names=target_names,
    title="Synthetic Pi-PLS prediction diagnostics",
)
prediction_figure.savefig(PREDICTION_PDF)
plt.close(prediction_figure)

print(f"external test R2: {model.score(test.X, test.Y):.6f}")
print(f"decomposition PDF: {DECOMPOSITION_PDF}")
print(f"prediction-diagnostic PDF: {PREDICTION_PDF}")
