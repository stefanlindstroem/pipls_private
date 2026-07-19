# Decision 0043: complete the fitted-model analysis surface with a Pulp biplot

## Status

Accepted and implemented.

## Context

Decision 0042 separated component-path diagnostics, fixed-model interpretation, and prediction diagnostics. Subsequent patches implemented reusable Pi-PLS and ordinary PLS inspection, plotting, and complete post-analysis workflows for Pulp, Sugarcane, and Tobacco. The remaining planned item was a conventional low-dimensional PLS score-loading biplot and a final review of the analysis API and artifact contracts.

A biplot places sample scores and variable loadings in one coordinate system, but the two matrices do not naturally have comparable scales. Any implementation must state its scaling, preserve a defined reconstruction, and avoid implying that high-dimensional spectral variables are readable as thousands of arrows.

## Decision

Add `PLSBiplotCoordinates` and `pls_biplot_coordinates()` under `pipls.inspection`. For exactly two selected zero-based components, let $t_k$ be the X-score column and $p_k$ the X-loading column. Define

\begin{equation}
a_k=\sqrt{\frac{\lVert p_k\rVert_2}{\lVert t_k\rVert_2}},\qquad
\tilde t_k=a_kt_k,\qquad
\tilde p_k=\frac{p_k}{a_k}.
\end{equation}

The result has equal score and loading norms within each selected component and preserves

\begin{equation}
\tilde T\tilde P^\mathsf{T}=T_{\mathcal K}P_{\mathcal K}^\mathsf{T}.
\end{equation}

Add `plot_pls_biplot()` under `pipls.plotting`. It displays sample coordinates and named X-loading arrows only. It does not add Y arrows, inferred groups, confidence regions, automatic labels, or importance claims.

Demonstrate the biplot only in the Pulp report using components 1 and 2. Pulp has fourteen named scalar predictors, so the arrows remain interpretable. Do not generate biplots for Sugarcane or Tobacco because their spectral predictor counts make the display unsuitable.

The Pulp report reconstructs biplot coordinates from the existing canonical `pls_scores.csv` and `pls_x_loadings.csv`. No additional CSV table is required.

## Cross-dataset review

The completed surface has these common rules:

- variable names and physical coordinates are obtained explicitly during example input;
- package inspection functions contain no pandas or Matplotlib dependency;
- package plotting functions accept immutable computed results and perform no file writing;
- component and response selections are explicit;
- full-data factors, scores, loadings, coefficients, biplots, and observation diagnostics are interpretive;
- prediction diagnostics carry explicit provenance;
- CSV tables are canonical and reports are reconstructed from them;
- Pulp uses named categorical bars and the low-dimensional biplot;
- Sugarcane uses increasing wavelength lines and no biplot;
- Tobacco uses decreasing wavenumber lines, response pagination, raw observation diagnostics, and no biplot.

## Consequences

The eight-part fitted-model analysis implementation series is complete. Further PLS diagnostics such as VIP, confidence regions, uncertainty intervals, probability limits, contribution plots, or automatic variable selection require separate decisions. The next repository phase is product documentation and release hardening.
