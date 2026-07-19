# Decision 0043: complete the fitted-model analysis surface with a Pulp biplot

## Status

Accepted, with ordinary-PLS-specific naming and numbered-example application superseded by
Decision 0045.

## Context

Decision 0042 separated component-path diagnostics, fixed-model interpretation, and prediction diagnostics. Subsequent patches implemented reusable Pi-PLS and ordinary PLS inspection, plotting, and complete post-analysis workflows for Pulp, Sugarcane, and Tobacco. The remaining planned item was a conventional low-dimensional PLS score-loading biplot and a final review of the analysis API and artifact contracts.

A biplot places sample scores and variable loadings in one coordinate system, but the two matrices do not naturally have comparable scales. Any implementation must state its scaling, preserve a defined reconstruction, and avoid implying that high-dimensional spectral variables are readable as thousands of arrows.

## Decision

The balanced biplot scaling below remains accepted. Decision 0045 requires estimator-neutral
result and function names and permits the computation from compatible PLS-family latent
structures rather than only ordinary PLS. For exactly two selected zero-based components, let
$t_k$ be the X-score column and $p_k$ the X-loading column. Define

\begin{equation}
a_k=\sqrt{\frac{\lVert p_k\rVert_2}{\lVert t_k\rVert_2}},\qquad
\tilde t_k=a_kt_k,\qquad
\tilde p_k=\frac{p_k}{a_k}.
\end{equation}

The result has equal score and loading norms within each selected component and preserves

\begin{equation}
\tilde T\tilde P^\mathsf{T}=T_{\mathcal K}P_{\mathcal K}^\mathsf{T}.
\end{equation}

The plotting contract displays sample coordinates and named X-loading arrows only. Its final
public name is estimator-neutral under Decision 0045. It does not add Y arrows, inferred groups,
confidence regions, automatic labels, or importance claims.

Demonstrate the biplot only in the Pulp report using components 1 and 2. Pulp has fourteen named
scalar predictors, so the arrows remain interpretable. Do not generate biplots for Sugarcane or
Tobacco because their spectral predictor counts make the display unsuitable.

The Pulp report continues to reconstruct biplot coordinates from canonical score and X-loading
tables. Decision 0045 requires generic shared-quantity filenames and applies the biplot to the
selected Pi-PLS model. No additional biplot table is required.

## Cross-dataset review

The completed surface has these common rules:

- variable names and physical coordinates are obtained explicitly during example input;
- package inspection functions contain no pandas or Matplotlib dependency;
- package plotting functions accept immutable computed results and perform no file writing;
- component and response selections are explicit;
- full-data factors, scores, loadings, coefficients, biplots, and observation diagnostics are
  interpretive;
- prediction diagnostics carry explicit provenance;
- CSV tables are canonical and reports are reconstructed from them;
- Pulp uses named categorical bars and the low-dimensional biplot;
- Sugarcane uses increasing wavelength lines and no biplot;
- Tobacco uses decreasing wavenumber lines, response pagination, raw observation diagnostics, and no biplot.

## Consequences

Decision 0045 completes the ownership, naming, and example correction: the biplot API is shared
PLS-family analysis, and the maintained Pulp report applies it to the selected Pi-PLS model.
Further PLS diagnostics such as VIP, confidence regions, uncertainty intervals, probability
limits, contribution plots, or automatic variable selection require separate decisions. Product
documentation and release hardening may now resume.
