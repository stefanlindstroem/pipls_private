# Plotting

`pipls.plotting` is the transitional reference for the remaining optional Matplotlib chart
primitives. Scientific interpretation belongs in [Model inspection](../model_inspection.md). The
[Pulp tutorial](../tutorials/pulp.md#score-loading-biplot) shows the data-first biplot boundary: Pi-PLS
calculates balanced coordinates, while Matplotlib and `adjustText` own rendering and label layout.

Every function renders one chart on one optional caller-supplied `ax` and returns `(figure, axis)`.
Plotters do not create panels, legends, figure-level titles, files, displays, or closing operations.
They provide concise axis labels and titles that callers may replace. Component and response
selections use zero-based Python indices. Matplotlib is imported only when a plotting function is
called.

Conceptual reference: [Pi-PLS factors](../model_inspection.md#predictor-directions).
Scores, loadings, coefficients, biplots, observation diagnostics, and prediction diagnostics are
rendered directly from immutable inspection arrays with ordinary Matplotlib.

::: pipls.plotting.PredictorStyle
    options:
      members: false

::: pipls.plotting.plot_pipls_predictor_directions
    options:
      members: false

::: pipls.plotting.plot_pipls_dilation
    options:
      members: false

::: pipls.plotting.plot_pipls_response_directions
    options:
      members: false

::: pipls.plotting.plot_pipls_weighted_response_directions
    options:
      members: false
