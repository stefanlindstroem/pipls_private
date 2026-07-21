"""Fit and inspect one Pi-PLS model from literal NumPy matrices."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipls import PiPLSRegression
from pipls.inspection import pipls_display_factors
from pipls.plotting import (
    plot_pipls_dilation,
    plot_pipls_predictor_directions,
    plot_pipls_response_directions,
    plot_pipls_weighted_response_directions,
)

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

model = PiPLSRegression(n_components=1, predictor_rank=2).fit(X, Y)
print(model.predict(X))

factors = pipls_display_factors(model.decomposition_)
figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), layout="constrained")
plot_pipls_predictor_directions(
    factors,
    predictor_style="bar",
    predictor_names=["Temperature", "Pressure", "Flow rate"],
    ax=axes[0, 0],
)
plot_pipls_dilation(factors, ax=axes[0, 1])
plot_pipls_response_directions(
    factors,
    response_names=["Yield", "Purity"],
    ax=axes[1, 0],
)
plot_pipls_weighted_response_directions(
    factors,
    response_names=["Yield", "Purity"],
    ax=axes[1, 1],
)
axes[0, 0].legend(title="Component")
axes[1, 0].legend(title="Component")
axes[1, 1].legend(title="Component")
figure.suptitle("Minimal Pi-PLS fit")
figure.savefig(Path(__file__).resolve().parent / "results" / "minimal_fit_and_plot.pdf")
plt.close(figure)
