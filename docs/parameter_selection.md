# Parameter selection

Pi-PLS has two complexity controls: `n_components` is the number of paired latent modes, while
`predictor_rank` is the retained predictor-subspace dimension. The usual procedure is:

1. evaluate a component path with `PiPLSPathCV`;
2. choose a parsimonious component count from the CV-MSE curve;
3. read the predictor rank from the same path row;
4. fit a separate `PiPLSRegression` with both values fixed.

The [Pulp tutorial](tutorials/pulp.md#evaluate-the-component-path) shows the complete procedure. The
[path-search reference](path_analysis.md) defines all search policies, bounds, diagnostics, and
pipeline behavior.

## Evaluate the component path

```python
import pandas as pd

from pipls import PiPLSPathCV

search = PiPLSPathCV(refit=False).fit(X, Y)
component_path = pd.DataFrame(search.component_path_results_)
component_path.to_csv("component_path.csv", index=False)
```

The default `n_components_values="all"` evaluates every admissible component count. For each count,
the path reports one conditionally selected predictor rank, response-standardized CV-MSE, and fold
standard deviation.

Plot CV-MSE against component count and choose an elbow, plateau, or other scientifically justified
point. The smallest evaluated CV-MSE is informative, but it is not an automatic scientific choice.
The fold standard deviation describes variation among the realized folds; it is not a confidence
interval.

## Fit the chosen fixed model

```python
from pipls import PiPLSRegression

chosen_n_components = 3
path_by_component = component_path.set_index("n_components")
chosen_predictor_rank = int(
    path_by_component.loc[chosen_n_components, "predictor_rank"]
)

model = PiPLSRegression(
    n_components=chosen_n_components,
    predictor_rank=chosen_predictor_rank,
).fit(X, Y)
```

Both values must come from the selected path row. This fixed fit does not repeat parameter
selection. Use [cross-validation protocols](cross_validation.md) when the split design, scorer, or
OOF reporting requires more control.
