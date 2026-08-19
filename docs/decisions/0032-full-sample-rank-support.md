# Decision 0032: full-sample predictor-rank support

## Status

Accepted and implemented for `PiPLSSearchCV`. This decision refines the rank-bound policy in
Decision 0003; fixed `PiPLSRegression` no longer derives a rank ceiling. Decision 0154 preserves
the full-sample convention only for the explicit EPV policy and removes it from the general search
ceiling; that accepted change is implemented.

## Context

The rule-derived predictor-rank ceiling previously used the smallest cross-validation training-fold
size in both the statistical-support term and the algebraic feasibility cap. That made the search
space depend on the chosen splitter and could exclude ranks intended for the final model, which is
refitted on all observations supplied to `fit()`.

The sample count itself contains no fitted information from `X` or `Y`. Using the total number of
supplied observations to define the support heuristic therefore does not fit preprocessing or model
parameters outside the training folds.

## Decision

Let $n$ be the total number of observations supplied to `fit()`, let $p$ be the predictor dimension
used by the EPV policy, and let $c=\texttt{samples\_per\_predictor\_rank}$. The nominal EPV rank is

\begin{equation}
r_{\pi,\mathrm{epv,nominal}}
=
\min\left[p,\left\lceil\frac{n}{c}\right\rceil\right].
\end{equation}

The term $\lceil n/c\rceil$ defines the explicit statistical-support heuristic for the final
full-data model. It does not bound ordinary automatic or exhaustive search. Fold-local predictor
dimensions, centered training-fold size, and verified numerical rank remain hard feasibility caps
and may clip the effective EPV rank.

`PiPLSSearchCV` materializes one split set and reuses it for every candidate under both
`search_method="adaptive"` and `search_method="exhaustive"`. Centering, scaling, decomposition,
fitting, and scoring remain training-fold local.

The EPV default is `samples_per_predictor_rank=10.0`; $c=5$ remains a more permissive choice.
Values below 5 remain legal only under the explicit EPV policy and emit
`PredictorRankSupportWarning`. Outside EPV, a nondefault `samples_per_predictor_rank` is invalid.

## Consequences

- The EPV heuristic describes the fixed predictor rank intended for the final full-data model.
- Changing the CV splitter does not change the nominal $n/c$ term, although fold feasibility may
  still clip the effective EPV rank.
- For 46 observations and 14 predictors, the nominal EPV ranks are 5 for $c=10$ and 10 for $c=5$.
- No global centering, scaling, response statistics, or latent structure is learned before CV.
- Ordinary automatic search no longer uses the samples-per-rank term.
