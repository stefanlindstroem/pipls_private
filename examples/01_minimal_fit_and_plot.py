"""Fit and inspect one Pi-PLS model from literal NumPy matrices."""

from pathlib import Path

import numpy as np

from pipls import PiPLSRegression
from pipls.inspection import pipls_display_factors
from pipls.plotting import plot_pipls_decomposition

X = np.array(
    [
        [1.0, 2.0, 0.5],
        [2.0, 1.0, 1.0],
        [3.0, 4.0, 1.5],
        [4.0, 3.0, 2.0],
        [5.0, 6.0, 2.5],
        [6.0, 5.0, 3.0],
        [7.0, 8.0, 3.5],
        [8.0, 7.0, 4.0],
    ]
)
Y = np.array(
    [
        [1.2, 2.0],
        [1.8, 1.7],
        [3.1, 3.3],
        [3.7, 3.0],
        [5.2, 4.6],
        [5.8, 4.3],
        [7.1, 5.9],
        [7.7, 5.6],
    ]
)

predictor_names = ["Temperature", "Pressure", "Flow rate"]
response_names = ["Yield", "Purity"]

model = PiPLSRegression(n_components=1, predictor_rank=2).fit(X, Y)
Y_fitted = model.predict(X)
print("Fitted responses:")
print(Y_fitted)

figure, _ = plot_pipls_decomposition(
    pipls_display_factors(model.decomposition_),
    predictor_style="bar",
    predictor_names=predictor_names,
    response_names=response_names,
    title="Minimal Pi-PLS fit",
)

output_path = Path(__file__).resolve().parent / "results" / "minimal_fit_and_plot.pdf"
output_path.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output_path)
print(f"Wrote {output_path}")
