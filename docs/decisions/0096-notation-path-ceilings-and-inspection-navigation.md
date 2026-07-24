# Decision 0096: notation, path ceilings, and inspection navigation

## Context

The mathematical contract uses $X$ and $Y$ for predictor and response matrices, while generated
Python signatures follow scikit-learn's `X` and `y` convention. Without an explicit explanation,
`y` can appear to imply a scalar response even though Pi-PLS is designed for multivariate
regression. The advanced path reference also used $h_{\max}$ in its admissible-set definition
without defining it beside $r_{\pi,\max}$, and the inspection concepts and generated API appeared as
unrelated peers in navigation.

## Decision

1. Mathematical documentation uses uppercase $X$ and $Y$ for predictor and response matrices.
   Python signatures and PLS-style public names retain scikit-learn's `X` and `y` convention;
   `y` may be two-dimensional and does not imply a scalar response.
2. User-facing prediction plots use response-neutral wording rather than scalar-looking formulas in
   axis labels.
3. The path reference defines both resolved ceilings before component-count and predictor-rank
   policy details:

   \begin{equation}
   r_{\pi,\max}=\min\left[p_{\min},n_{\mathrm{train,min}}-1,
   r_{\mathrm{num,min}},\left\lceil\frac{n}{c}\right\rceil\right],
   \end{equation}

   \begin{equation}
   h_{\max}=\min(q,r_{\pi,\max}).
   \end{equation}

   Explicit predictor-rank sets may still make a requested component count inadmissible if no
   supplied rank satisfies $h\le r_\pi$.
4. Inspection concepts and generated signatures remain separate pages, but navigation groups them
   under one model-inspection section and the pages link to each other.

## Consequences

The public Python API remains scikit-learn-compatible and no names change. The path reference now
introduces every symbol before policy details, and inspection users can move directly between
interpretation guidance and exact generated contracts. Numerical and estimator behavior are
unchanged.
