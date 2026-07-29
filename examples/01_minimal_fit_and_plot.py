"""Fit and inspect one Pi-PLS model from literal NumPy matrices."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipls import PiPLSRegression
from pipls.inspection import pipls_display_factors

predictor_names = ["Temperature", "Pressure", "Flow rate"]
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

response_names = ["Yield", "Purity"]
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
predictions = model.predict(X)
print("Predictions:")
print(predictions)

factors = pipls_display_factors(model.decomposition_)

figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), layout="constrained")
predictor_positions = np.arange(len(predictor_names))
axes[0, 0].bar(predictor_positions, factors.predictor_directions[:, 0])
axes[0, 0].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axes[0, 0].set_xticks(predictor_positions)
axes[0, 0].set_xticklabels(predictor_names)
axes[0, 0].set_xlabel("Predictor")
axes[0, 0].set_ylabel(r"Predictor direction $P_{:1}$")

axes[0, 1].bar([0], [factors.dilation[0]])
axes[0, 1].set_xticks([0])
axes[0, 1].set_xticklabels(["1"])
axes[0, 1].set_xlabel("Component")
axes[0, 1].set_ylabel(r"Dilation $d_k$")

response_positions = np.arange(len(response_names))
axes[1, 0].bar(response_positions, factors.response_directions[:, 0])
axes[1, 0].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axes[1, 0].set_xticks(response_positions)
axes[1, 0].set_xticklabels(response_names)
axes[1, 0].set_xlabel("Response")
axes[1, 0].set_ylabel(r"Response direction $Q_{:1}$")

axes[1, 1].bar(response_positions, factors.weighted_response_directions[:, 0])
axes[1, 1].axhline(0.0, linewidth=0.8, linestyle="--", color="0.45")
axes[1, 1].set_xticks(response_positions)
axes[1, 1].set_xticklabels(response_names)
axes[1, 1].set_xlabel("Response")
axes[1, 1].set_ylabel(r"Weighted direction $d_1Q_{:1}$")

figure.suptitle(r"Minimal $\Pi$-PLS fit")
output_path = Path(__file__).resolve().parent / "results" / "minimal_fit_and_plot.pdf"
figure.savefig(output_path)
plt.close(figure)
print(f"Wrote PDF figure to {output_path}")
