# Decision: 0003-predictor-rank-selection

Status: historical selection design. Decision 0039 supersedes the `PiPLSRegression` search
modes; the rank ceiling and adaptive/exhaustive policies remain implemented in `PiPLSSearchCV`.

For total supplied sample count $n$, smallest training-set size $n_{\mathrm{train,min}}$,
predictor count $p$, and positive numeric `samples_per_predictor_rank` value $c$, the shared
upper-bound helper returns

\begin{equation}
r_{\pi,\max}
=
\min\left[
p,
n_{\mathrm{train,min}}-1,
\left\lceil\frac{n}{c}\right\rceil
\right].
\end{equation}

The ceiling operation and all three caps are normative. The helper rejects nonpositive,
nonfinite, or boolean values of $c$.

In `predictor_rank="max"` mode there is no internal cross-validation, so
$n_{\mathrm{train,min}}=n$. In both CV modes, the estimator materializes and copies one CV split
set, derives the smallest training-fold size from it as a feasibility cap, and reuses the same
splits for every evaluated candidate. The statistical-support term continues to use the full $n$
because the selected model is refitted on all supplied observations.

`predictor_rank="optimal"` scans every integer rank from `n_components` through the fold-safe
upper bound. `predictor_rank="auto"` uses the adaptive policy in decision 0007 and may evaluate
only a subset. Both select the maximum mean scikit-learn score. Scores equal within `rtol=1e-12`
and `atol=1e-15` are resolved in favor of the smaller predictor rank. The selected fixed-rank
model is refitted on all data supplied to `fit()`.

An explicit integer `predictor_rank` bypasses the rule-derived upper bound. It remains subject to
the fixed-core numerical-rank and dimensional checks. `max_predictor_rank_` records the
rule-derived bound even when an explicit integer rank is used.


## Refinement

Decision 0007 establishes the implemented naming: `"optimal"` is exhaustive and `"auto"` is
adaptive approximate search with cached evaluations and explicit diagnostics.
