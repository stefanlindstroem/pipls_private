# Plotting

`pipls.plotting` provides optional Matplotlib figures for results from `pipls.inspection`.
Matplotlib is imported only when a plotting function is called; the numerical package and the
inspection helpers remain usable without it. Install Matplotlib directly or use the `examples`
optional dependency group.

Every plotting function returns a Matplotlib figure and a dictionary of named axes. The functions
do not call `show()`, write files, infer scientific variable names, or add theoretical decision
limits. Component and response selections use zero-based Python indices.

::: pipls.plotting.PredictorStyle
    options:
      members: false

::: pipls.plotting.plot_pipls_decomposition
    options:
      members: false

::: pipls.plotting.plot_prediction_diagnostics
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
