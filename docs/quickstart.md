# Quickstart

This page shows the shortest complete Pi-PLS workflow: construct two NumPy matrices, fit one fixed
model, make predictions, and plot its paired latent predictor-response directions. It performs no
cross-validation or parameter selection. The example fixes one latent component and a predictor
rank of two; applied model development normally selects these values first.

Install the package with the optional plotting dependency:

```bash
python -m pip install "pipls[plot]"
```

Then run:

```python
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

figure, _ = plot_pipls_decomposition(
    pipls_display_factors(model.decomposition_),
    predictor_style="bar",
    predictor_names=predictor_names,
    response_names=response_names,
)
figure.savefig(Path("minimal_fit.pdf"))
```

`PiPLSRegression` follows the normal scikit-learn estimator pattern: constructor arguments describe
the model, `fit()` learns from `X` and `Y`, and `predict()` returns responses in the original
response units. Here both ranks are fixed explicitly. The model does not run cross-validation
internally.

The plotting calls are separate from fitting. `pipls_display_factors()` creates an immutable display
copy of the fitted latent factorization, and `plot_pipls_decomposition()` renders its paired
predictor directions, response directions, and component strengths. The corresponding matrices
form $P D Q^{\mathsf T}$; see the [theory overview](theory.md#diagonal-latent-coupling) for their
construction. Matplotlib remains optional and the estimator retains no plotting state.

The complete executable script is summarized in [`examples.md`](examples.md). Continue with:

1. [`parameter_selection.md`](parameter_selection.md) to scan component counts and choose a model
   from the CV-MSE curve;
2. [`model_inspection.md`](model_inspection.md) for fitted latent structure, prediction diagnostics,
   and the Pi-PLS-specific factorization;
3. [`examples.md`](examples.md) for synthetic, comparison, and complete real-data workflows.
