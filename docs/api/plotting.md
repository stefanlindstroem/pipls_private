# Plotting

`pipls.plotting` is the operational reference for optional Matplotlib chart primitives. Scientific
interpretation belongs in [Model inspection](../model_inspection.md), and the
[Pulp tutorial](../tutorials/pulp.md#interpret-the-fitted-model) shows the complete maintained
plotting surface.

Every function renders one chart on one optional caller-supplied `ax` and returns `(figure, axis)`.
Plotters do not create panels, legends, figure-level titles, files, displays, or closing operations.
They provide concise axis labels and titles that callers may replace. Component and response
selections use zero-based Python indices. Matplotlib is imported only when a plotting function is
called.

Conceptual reference: [scores](../model_inspection.md#scores),
[biplots](../model_inspection.md#score-loading-biplot),
[loadings](../model_inspection.md#x-loadings),
[Pi-PLS factors](../model_inspection.md#predictor-directions),
[coefficients](../model_inspection.md#regression-coefficients), and
[prediction diagnostics](../model_inspection.md#observed-versus-predicted).

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

::: pipls.plotting.plot_observed_vs_predicted
    options:
      members: false

::: pipls.plotting.plot_residuals_vs_predicted
    options:
      members: false

::: pipls.plotting.plot_standardized_rmse
    options:
      members: false

::: pipls.plotting.plot_scores
    options:
      members: false

::: pipls.plotting.plot_x_loadings
    options:
      members: false

::: pipls.plotting.plot_y_loadings
    options:
      members: false

::: pipls.plotting.plot_coefficients
    options:
      members: false

::: pipls.plotting.plot_biplot
    options:
      members: false

::: pipls.plotting.plot_observation_diagnostics
    options:
      members: false
