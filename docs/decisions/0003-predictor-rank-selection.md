# Decision: 0003-predictor-rank-selection

Status: implemented for explicit, rule-fixed, and automatic conditional selection.

For a supplied smallest training-set size $n_{\mathrm{train,min}}$, predictor count $p$, and
positive numeric `samples_per_predictor_rank` value $c$, the shared upper-bound helper returns

\begin{equation}
r_{\pi,\max}
=
\min\left[
p,
n_{\mathrm{train,min}},
\left\lceil\frac{n_{\mathrm{train,min}}}{c}\right\rceil
\right].
\end{equation}

The ceiling operation and all three caps are normative. The helper rejects nonpositive,
nonfinite, or boolean values of $c$.

In `predictor_rank="max"` mode there is no internal cross-validation, so the sample count supplied
to `fit()` is used as $n_{\mathrm{train,min}}$. In automatic mode, the estimator materializes and
copies one CV split set, derives the smallest training-fold size from it, and reuses the same
splits and rank grid for all candidates.

Automatic mode scans every integer rank from `n_components` through the fold-safe upper bound.
It selects the maximum mean scikit-learn score. Scores equal within `rtol=1e-12` and `atol=1e-15`
are resolved in favor of the smaller predictor rank. The selected fixed-rank model is refitted on
all data supplied to `fit()`.

An explicit integer `predictor_rank` bypasses the rule-derived upper bound. It remains subject to
the fixed-core numerical-rank and dimensional checks. `max_predictor_rank_` records the
rule-derived bound even when an explicit integer rank is used.
