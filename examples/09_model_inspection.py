"""Inspect fixed Pi-PLS and ordinary PLS models on an external test problem."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

from pipls import PiPLSRegression
from pipls.datasets import make_pipls_train_test
from pipls.inspection import (
    pipls_display_factors,
    pls_latent_structure,
    prediction_diagnostics,
)
from pipls.plotting import (
    plot_pipls_decomposition,
    plot_pls_coefficients,
    plot_pls_scores,
    plot_pls_x_loadings,
    plot_pls_y_loadings,
    plot_prediction_diagnostics,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "model_inspection"
DECOMPOSITION_PDF = RESULTS_DIR / "pipls_decomposition.pdf"
PREDICTION_PDF = RESULTS_DIR / "prediction_diagnostics.pdf"
PLS_SCORES_PDF = RESULTS_DIR / "pls_scores.pdf"
PLS_X_LOADINGS_PDF = RESULTS_DIR / "pls_x_loadings.pdf"
PLS_Y_LOADINGS_PDF = RESULTS_DIR / "pls_y_loadings.pdf"
PLS_COEFFICIENTS_PDF = RESULTS_DIR / "pls_coefficients.pdf"

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

pls_model = PLSRegression(n_components=2, scale=True).fit(train.X, train.Y)
pls_structure = pls_latent_structure(pls_model)

feature_names = [
    "Process variable A",
    "Process variable B",
    "Process variable C",
    "Process variable D",
    "Process variable E",
    "Process variable F",
    "Process variable G",
    "Process variable H",
    "Process variable I",
    "Process variable J",
    "Process variable K",
    "Process variable L",
]
target_names = ["Quality response A", "Quality response B", "Quality response C"]

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

pls_scores_figure, _ = plot_pls_scores(
    pls_structure,
    components=(0, 1),
    title="Synthetic ordinary PLS scores",
)
pls_scores_figure.savefig(PLS_SCORES_PDF)
plt.close(pls_scores_figure)

pls_x_loadings_figure, _ = plot_pls_x_loadings(
    pls_structure,
    predictor_style="bar",
    predictor_names=feature_names,
    components=[0, 1],
    title="Synthetic ordinary PLS X loadings",
)
pls_x_loadings_figure.savefig(PLS_X_LOADINGS_PDF)
plt.close(pls_x_loadings_figure)

pls_y_loadings_figure, _ = plot_pls_y_loadings(
    pls_structure,
    response_names=target_names,
    components=[0, 1],
    title="Synthetic ordinary PLS Y loadings",
)
pls_y_loadings_figure.savefig(PLS_Y_LOADINGS_PDF)
plt.close(pls_y_loadings_figure)

pls_coefficients_figure, _ = plot_pls_coefficients(
    pls_structure,
    predictor_style="bar",
    predictor_names=feature_names,
    response_names=target_names,
    title="Synthetic ordinary PLS coefficients",
)
pls_coefficients_figure.savefig(PLS_COEFFICIENTS_PDF)
plt.close(pls_coefficients_figure)

print(f"Pi-PLS external test R2: {model.score(test.X, test.Y):.6f}")
print(f"ordinary PLS external test R2: {pls_model.score(test.X, test.Y):.6f}")
print(f"Pi-PLS decomposition PDF: {DECOMPOSITION_PDF}")
print(f"prediction-diagnostic PDF: {PREDICTION_PDF}")
print(f"ordinary PLS score PDF: {PLS_SCORES_PDF}")
print(f"ordinary PLS X-loading PDF: {PLS_X_LOADINGS_PDF}")
print(f"ordinary PLS Y-loading PDF: {PLS_Y_LOADINGS_PDF}")
print(f"ordinary PLS coefficient PDF: {PLS_COEFFICIENTS_PDF}")
