# Decision: 0003-predictor-rank-selection

Status: historical selection design. Decision 0039 supersedes the `PiPLSRegression` search
modes. Decision 0154 supersedes the general $n/c$ search ceiling and assigns that heuristic
only to the explicit EPV policy; that accepted change is implemented.

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

`PiPLSSearchCV` materializes and copies one CV split set, derives the smallest training-fold size
from it as a feasibility cap, and reuses the same splits for every evaluated candidate. The
statistical-support term continues to use the full $n$ because the selected model is refitted on all
supplied observations.

`search_method="exhaustive"` scans every admissible predictor rank from `n_components` through the
fold-safe upper bound. `search_method="adaptive"` uses the policy in Decision 0007 and may evaluate
only a subset. Private numerical score ties are resolved deterministically in favor of the smaller
predictor rank. Decision 0148 supersedes exact conditional score maximization as the final retained-
rank rule by adding separate public relative and absolute predictor-rank tolerances in
`PiPLSSearchCV`; materialized-split reuse and private numerical tie semantics in this record remain
normative, while Decision 0154 supersedes the general rank ceiling and default coverage policy.

The former explicit fixed-estimator rank-mode behavior in this record is superseded by Decision
0039. `PiPLSRegression` now fits one explicit pair; search-domain controls belong to
`PiPLSSearchCV` under Decisions 0039 and 0154.


## Refinement

Decision 0007 defines exhaustive and adaptive candidate coverage, the implemented public values
`"exhaustive"` and `"adaptive"`, cached evaluations, and explicit diagnostics.
