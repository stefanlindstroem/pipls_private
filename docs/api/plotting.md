# Plotting

`pipls.plotting` provides optional Matplotlib figures for results from `pipls.inspection`.
Matplotlib is imported only when a plotting function is called; the numerical package and the
inspection helpers remain usable without it. Install Matplotlib directly or use the `examples`
optional dependency group.

The single-chart plotting functions accept an optional Matplotlib `ax` and return `(figure, axis)`.
With `ax=None`, they create one figure containing one axis. With a supplied axis, they draw on that
axis without clearing it or changing the surrounding figure.
Plotters provide concise semantic axis labels and titles, which callers may replace through the
returned axis. They label multi-series artists but do not create legends; legend placement and
styling belong to the caller. They do not create subplot grids or mosaics, set figure-level titles,
show or save figures, or close them. Component and response selections use zero-based Python
indices.

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
