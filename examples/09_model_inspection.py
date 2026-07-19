"""Inspect fixed Pi-PLS and ordinary PLS models on the Pulp dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from pipls import PiPLSRegression
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPOSITORY_ROOT / "datasets" / "pulp"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "model_inspection"
DECOMPOSITION_PDF = RESULTS_DIR / "pipls_decomposition.pdf"
PREDICTION_PDF = RESULTS_DIR / "prediction_diagnostics.pdf"
PLS_SCORES_PDF = RESULTS_DIR / "pls_scores.pdf"
PLS_X_LOADINGS_PDF = RESULTS_DIR / "pls_x_loadings.pdf"
PLS_Y_LOADINGS_PDF = RESULTS_DIR / "pls_y_loadings.pdf"
PLS_COEFFICIENTS_PDF = RESULTS_DIR / "pls_coefficients.pdf"

# These fixed values are chosen only to demonstrate fitted-model inspection.
N_COMPONENTS = 3
PREDICTOR_RANK = 10
COEFFICIENT_RESPONSES = (0, 1, 2)

# File reading and label acquisition are example-level tasks. The plotting API
# receives names explicitly and does not know whether they came from CSV headers,
# another metadata source, or a manually supplied list.
X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")
predictor_names = X.columns.astype(str).tolist()
response_names = Y.columns.astype(str).tolist()

model = PiPLSRegression(
    n_components=N_COMPONENTS,
    predictor_rank=PREDICTOR_RANK,
).fit(X, Y)
factors = pipls_display_factors(model.decomposition_)
diagnostics = prediction_diagnostics(
    Y,
    model.predict(X),
    prediction_kind="fitted values",
)

pls_model = PLSRegression(n_components=N_COMPONENTS, scale=True).fit(X, Y)
pls_structure = pls_latent_structure(pls_model)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
decomposition_figure, _ = plot_pipls_decomposition(
    factors,
    predictor_style="bar",
    predictor_names=predictor_names,
    response_names=response_names,
    title="Pulp Pi-PLS decomposition",
)
decomposition_figure.savefig(DECOMPOSITION_PDF)
plt.close(decomposition_figure)

prediction_figure, _ = plot_prediction_diagnostics(
    diagnostics,
    response_names=response_names,
    title="Pulp Pi-PLS prediction diagnostics",
)
prediction_figure.savefig(PREDICTION_PDF)
plt.close(prediction_figure)

pls_scores_figure, _ = plot_pls_scores(
    pls_structure,
    components=(0, 1),
    title="Pulp ordinary PLS scores",
)
pls_scores_figure.savefig(PLS_SCORES_PDF)
plt.close(pls_scores_figure)

pls_x_loadings_figure, _ = plot_pls_x_loadings(
    pls_structure,
    predictor_style="bar",
    predictor_names=predictor_names,
    components=[0, 1],
    title="Pulp ordinary PLS X loadings",
)
pls_x_loadings_figure.savefig(PLS_X_LOADINGS_PDF)
plt.close(pls_x_loadings_figure)

pls_y_loadings_figure, _ = plot_pls_y_loadings(
    pls_structure,
    response_names=response_names,
    components=[0, 1],
    title="Pulp ordinary PLS Y loadings",
)
pls_y_loadings_figure.savefig(PLS_Y_LOADINGS_PDF)
plt.close(pls_y_loadings_figure)

pls_coefficients_figure, _ = plot_pls_coefficients(
    pls_structure,
    predictor_style="bar",
    predictor_names=predictor_names,
    response_names=response_names,
    responses=COEFFICIENT_RESPONSES,
    title="Pulp ordinary PLS coefficients",
)
pls_coefficients_figure.savefig(PLS_COEFFICIENTS_PDF)
plt.close(pls_coefficients_figure)

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"predictor labels read from X.csv: {len(predictor_names)}")
print(f"response labels read from Y.csv: {len(response_names)}")
print(f"Pi-PLS fitted R2: {model.score(X, Y):.6f}")
print(f"ordinary PLS fitted R2: {pls_model.score(X, Y):.6f}")
print(f"Pi-PLS decomposition PDF: {DECOMPOSITION_PDF}")
print(f"prediction-diagnostic PDF: {PREDICTION_PDF}")
print(f"ordinary PLS score PDF: {PLS_SCORES_PDF}")
print(f"ordinary PLS X-loading PDF: {PLS_X_LOADINGS_PDF}")
print(f"ordinary PLS Y-loading PDF: {PLS_Y_LOADINGS_PDF}")
print(f"ordinary PLS coefficient PDF: {PLS_COEFFICIENTS_PDF}")
