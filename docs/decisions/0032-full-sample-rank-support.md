# Decision 0032: full-sample predictor-rank support

## Status

Accepted and implemented for `PiPLSSearchCV`. This decision refines the rank-bound policy in
Decision 0003; fixed `PiPLSRegression` no longer derives a rank ceiling. Decision 0154 preserves
the full-sample convention only for the explicit EPV policy and removes it from the general search
ceiling; implementation of that accepted change is pending.

## Context

The rule-derived predictor-rank ceiling previously used the smallest cross-validation training-fold
size in both the statistical-support term and the algebraic feasibility cap. That made the search
space depend on the chosen splitter and could exclude ranks intended for the final model, which is
refitted on all observations supplied to `fit()`.

The sample count itself contains no fitted information from `X` or `Y`. Using the total number of
supplied observations to define the support heuristic therefore does not fit preprocessing or model
parameters outside the training folds.

## Decision

Let $n$ be the total number of observations supplied to `fit()`, let
$n_{\mathrm{train,min}}$ be the smallest materialized cross-validation training-fold size, let
$p_{\min}$ be the smallest predictor dimension reaching the Pi-PLS estimator across those folds,
and let $c=\texttt{samples\_per\_predictor\_rank}$. The rule-derived upper rank is

\begin{equation}
r_{\pi,\max}
=
\min\left[
p_{\min},
n_{\mathrm{train,min}}-1,
\left\lceil\frac{n}{c}\right\rceil
\right].
\end{equation}

The term $\lceil n/c\rceil$ defines statistical support for the final full-data model. The terms
$p_{\min}$ and $n_{\mathrm{train,min}}-1$ are hard feasibility caps for every candidate training
fold. The subtraction by one reflects model-internal centering, which limits the rank of a centered
training matrix with $n_{\mathrm{train,min}}$ rows to at most
$n_{\mathrm{train,min}}-1$.

`PiPLSSearchCV` materializes one split set and reuses it for every candidate under both
`search_method="adaptive"` and `search_method="exhaustive"`. Centering, scaling, decomposition,
fitting, and scoring remain training-fold local.

The ordinary defaults remain `samples_per_predictor_rank=5` and `cv=5`. Values below 5 remain
legal with `max_predictor_rank="rule"` but emit `PredictorRankSupportWarning`.

## Consequences

- The heuristic candidate range describes the model that will be refitted on all supplied data.
- Changing the CV splitter does not change the support term, although unusually small training
  folds can still reduce the range through the feasibility cap.
- For the 46-row, 14-predictor Pulp data with five-fold CV, the default bound is
  $\min[14,35,\lceil46/5\rceil]=10$.
- No global centering, scaling, response statistics, or latent structure is learned before CV.
- Explicit integer ranks and explicit path maxima continue to bypass the samples-per-rank term but
  remain subject to fold feasibility and numerical-rank validation.
