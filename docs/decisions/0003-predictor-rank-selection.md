# Decision: 0003-predictor-rank-selection

Status: partially implemented through the rule-fixed mode; automatic selection remains pending.

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

In the currently implemented `predictor_rank="max"` mode there is no internal cross-validation,
so the sample count supplied to `fit()` is used as $n_{\mathrm{train,min}}$. When automatic
selection is implemented, the same helper must instead receive the smallest training-fold size
from the materialized internal-CV splits.

An explicit integer `predictor_rank` bypasses this rule-derived upper bound. It remains subject to
the fixed-core numerical-rank and dimensional checks. `max_predictor_rank_` records the
rule-derived bound even when an explicit integer rank is used.

Automatic-selection staging now fixes the following private contracts before estimator integration:

- materialize and copy one CV split set for reuse by every candidate;
- derive `n_train_min` from that materialized set;
- scan every integer rank from `n_components` through the fold-safe upper bound;
- when mean losses are equal within `rtol=1e-12` and `atol=1e-15`, select the smaller rank.

The public `"auto"` mode remains pending until these primitives are integrated with fold-local estimator fitting.
