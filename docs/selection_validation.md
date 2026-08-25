# Selection and validation

This page defines the search, cross-validation, selection, and out-of-fold contracts used by
`PiPLSSearchCV`. For constructor signatures and fitted attributes, see
[`PiPLSSearchCV`](api/path.md). The tutorials own the worked selection workflows.

## Search domain { #search-bounds }

For paired-mode count $h$ (`n_components`) and retained predictor-subspace dimension $r_\pi$
(`predictor_rank`), the ordinary automatic search domain is triangular. Their different modeling
roles are described under
[Interpretation of the two rank controls](theory.md#interpretation-of-the-ranks).

Let $p_{\mathrm{min}}$ be the minimum predictor count after fold-local preprocessing,
$n_{\mathrm{train,min}}$ the smallest materialized training-fold size, and
$r_{\mathrm{num,min}}$ the minimum verified predictor numerical rank after terminal-estimator
centering and optional scaling. The hard predictor-rank ceiling is

\begin{equation}
r_{\pi,\mathrm{hard}}
=
\min\left[p_{\mathrm{min}},n_{\mathrm{train,min}}-1,r_{\mathrm{num,min}}\right].
\end{equation}

With `max_predictor_rank=None`, the automatic-search ceiling is this hard ceiling. An explicit
positive integer `max_predictor_rank=k` adds the user restriction

\begin{equation}
r_{\pi,\mathrm{max}}
=
\min\left[r_{\pi,\mathrm{hard}},k\right].
\end{equation}

These limits are feasibility constraints. `samples_per_predictor_rank` does not enter the general
ceiling; it applies only to the explicit EPV policy.

With `predictor_rank_values=None`, the admissible ranks at component count $h$ are every integer
from $h$ through $r_{\pi,\mathrm{max}}$. `n_components_values="all"` then evaluates component counts
from one through the smaller of the response count and the largest rank available under the chosen
predictor-rank policy. Explicit component or predictor-rank requests outside the resolved feasible
domain are rejected rather than silently dropped.

## Predictor-rank policies { #predictor-rank-policies }

`predictor_rank_values` defines the rank domain or fixed-rank policy; `search_method` controls the
coverage of a multi-rank domain.

- `None` optimizes over every feasible integer rank, subject to $r_\pi\ge h$.
- A one-element integer sequence fixes that rank for every compatible component count.
- A longer integer sequence optimizes over exactly those supplied ranks after feasibility checks.
- `"epv"` fixes one rank using the events-per-variable-inspired rule.

### EPV policy { #epv-policy }

For the EPV policy, the nominal rank is computed from the full number of observations supplied to
`fit()`:

\begin{equation}
r_{\pi,\mathrm{epv,nominal}}
=
\min\left[p,\left\lceil\frac{n}{c}\right\rceil\right],
\end{equation}

where $c$ is `samples_per_predictor_rank`. The effective EPV rank is clipped only by the hard
feasibility ceiling and an explicit integer `max_predictor_rank`, if supplied. The default is
$c=10$; $c=5$ is supported as a more permissive policy. Values below 5 are legal but emit
`PredictorRankSupportWarning`. A nondefault `samples_per_predictor_rank` is invalid outside the EPV
policy.

EPV and one-element fixed-rank policies expose one rank per compatible component count and therefore
have no conditional predictor-rank search or `PiPLSPredictorRankEvidence`.

## Candidate coverage and scoring { #scoring-and-conditioned-path-selection }

`search_method="exhaustive"` evaluates every admissible pair in a multi-rank domain.
`search_method="adaptive"` uses the same admissible endpoints but may leave interior ranks
unevaluated. `search_is_exhaustive_` records whether every admissible pair was actually evaluated.
Adaptive search makes no claim about ranks it did not evaluate.

Candidate evaluation uses the configured mean test score. The default scorer name is
`"neg_response_standardized_mse"`, which resolves to
`pipls.metrics.neg_response_standardized_mse`; maximizing it is equivalent to minimizing mean
response-standardized CV-MSE. Ordinary scikit-learn scorer names, scorer callables, and
`scoring=None` are also accepted. With another scorer, CV-MSE remains a diagnostic and need not
identify the configured-score optimum.

For an optimized rank policy, let $S_{\mathrm{max}}$ be the maximum configured mean score at one component
count. The smallest evaluated rank is retained when its score is no smaller than both
$S_{\max}-\delta_{\mathrm{rel}}|S_{\max}|$ and $S_{\max}-\delta_{\mathrm{abs}}$ (up to the private
numerical tie tolerance). `predictor_rank_relative_tolerance=None` resolves to
`sqrt(np.finfo(np.float64).eps)`; positive-infinity `predictor_rank_absolute_tolerance` disables the
absolute cap.

`PiPLSPredictorRankEvidence` records the exact configured-score reference rank and the resolved
tolerances. Under exhaustive coverage the reference is taken over the complete admissible rank
domain; under adaptive coverage it is necessarily limited to evaluated ranks.
`predictor_rank_profile(h).reference_selection` exposes the exact reference, while `.selection`
exposes the retained tolerance-qualified rank.

`cv_results_` remains the complete evaluated-candidate record. `rank_test_score` uses minimum ranks
with private `rtol=1e-12` and `atol=1e-15` comparisons; tied score groups are anchored to the leading
score in each group rather than chained through adjacent values. Public parsimony tolerances do not
change these candidate-level ranks.

## Pipelines and fold-local preprocessing { #pipelines-and-fold-local-preprocessing }

The supported search estimator is either a direct `PiPLSRegression` or a scikit-learn `Pipeline`
whose final step is `PiPLSRegression`. The complete estimator is cloned and fitted inside every
training fold. Learned preprocessing must therefore remain inside the estimator or pipeline rather
than being fitted once on the complete dataset before search.

The terminal Π-PLS estimator uses a valid construction seed pair such as `(1, 1)` because
`PiPLSRegression` always represents one explicit fixed pair. `PiPLSSearchCV` replaces
`n_components` and `predictor_rank` before feasibility checks, candidate fits, OOF fits, and final
refitting. Other template settings, including scaling, response-subspace policy, `svd_solver`, and
`random_state`, remain part of every candidate.

## Cross-validation protocols and metadata { #cross-validation-protocols-and-metadata }

`PiPLSSearchCV` accepts scikit-learn-compatible splitters and explicit split iterables, including
grouped, repeated, predefined, temporal, and other protocols when their assumptions match the
data. `groups` is an explicit `fit()` parameter and participates in scikit-learn metadata routing
when routing is enabled and requested. Split metadata belongs to the search object;
`PiPLSRegression.fit(X, y)` accepts none.

The search materializes and stores defensive read-only copies of the exact validation indices used
by `fit()`. Consequently, an iterable splitter is not consumed a second time and a stochastic
splitter is not asked to generate a new partition for OOF reporting.

Π-PLS provides no dedicated leave-one-out mode or package-specific interpretation for singleton
validation folds. Foldwise $R^2$ is rejected when a validation split contains fewer than two
observations. Pooled OOF $R^2$, when available, is a statistic over covered observations and is not
mean foldwise $R^2$.

## Ordered out-of-fold predictions { #ordered-out-of-fold-predictions }

`search.oof_report(X, y, selection=...)` validates an existing selection against the fitted search,
fits that fixed pair once per stored training fold, and returns an immutable `PiPLSOOFReport`.

The report preserves input row order. Repeated validation predictions are averaged and their counts
are retained; uncovered rows have count 0 and NaN predictions. `has_complete_oof_coverage` records
whether every row received at least one validation prediction, and `pooled_oof_r2` uses only rows
with OOF coverage.

The supplied data must have the same sample count, predictor count, and response-column count as the
fitted search. The search does not retain and compare the original values, so the caller is
responsible for passing the same observations in the same row order.

OOF evidence is selection-conditioned because the same search protocol produced the supplied
selection. Use nested cross-validation or an external test set when an unbiased post-selection
performance estimate is required.

## Split variation

`component_path_.cv_mse_std` is the population standard deviation of the realized split-specific
response-standardized MSE values. Maintained path and predictor-rank-profile figures use symmetric
$\pm 1$ SD bars from this stored quantity. These bars describe split-to-split variation; they are not
confidence intervals, do not enter selection, and are not standard errors.

## Search-owned selection rules { #search-owned-selection-rules }

`PiPLSSearchCV.select()` returns one complete immutable stored component-path row without fitting or
mutating the search.

For `rule="minimum_cv_mse"`, let $M_{\mathrm{min}}$ be the exact minimum stored mean CV-MSE. A row qualifies
only when its mean is no larger than both $(1+\delta_{\mathrm{rel}})M_{\min}$ and
$M_{\min}+\delta_{\mathrm{abs}}$. Because component counts are stored in ascending order, the first
qualifying row is returned. `relative_tolerance=None` resolves to
`sqrt(np.finfo(np.float64).eps)`; positive-infinity `absolute_tolerance` disables the absolute cap.
The returned selection retains the exact unruled minimum as `reference_minimum`, the resolved
tolerances, and the derived `cv_mse_threshold`.

`rule="best_score"` chooses the maximum configured-score row on the predictor-rank-conditioned
component path. Direct lookup by component count and `best_score` carry no CV-MSE tolerance
provenance.

The predictor rank in every selected row is the rank already retained conditionally for that
component count under the configured scorer and constructor-level predictor-rank tolerances.
`select()` does not revisit the predictor-rank profile. With a nondefault scorer, the retained
predictor rank therefore need not minimize CV-MSE within its component-count profile.

## Final-model refitting and provenance { #final-model-refitting }

`refit()` fits exactly one stored component-path row on the supplied full data. Exactly one of an
existing `selection`, a named `rule`, or `n_components` is required. An existing selection is
validated against the fitted search before fitting.

The search clones the configured estimator or pipeline, replaces the terminal Π-PLS rank pair, fits
the clone, attaches the exact immutable selection as `model.selection_`, and returns the model. It
does not mutate the search or attach the returned model to search state. Prediction,
transformation, scoring, inverse transformation, and fitted-model inspection belong to the returned
estimator.

`oof_report()` likewise consumes a selection without altering search state. The search stores
candidate evidence and reusable split indices, but no OOF report or final fitted model.

## Computational consequences { #computational-consequences }

If $N_{\mathrm{split}}$ validation splits and $N_{\mathrm{pair}}$ candidate pairs are materialized,
candidate evaluation performs

\begin{equation}
N_{\mathrm{candidate\ fit}}
=
N_{\mathrm{pair}}N_{\mathrm{split}}
\end{equation}

fold-local fits, in addition to fold-level predictor-rank feasibility probes. A final `refit()` adds
one full-data fit. Each `oof_report()` call adds one selected-pair fit per stored split, so retaining
and reusing one immutable report avoids repeated work.

Narrowing the rank or component domain changes the model-selection problem. Adaptive coverage
changes which interior ranks are evaluated. Fewer validation splits change the validation evidence.
These are not computationally neutral shortcuts.

`n_jobs` changes execution rather than the requested candidates or validation splits. Candidate
pairs are parallel tasks during search; OOF reporting parallelizes selected-pair fits across
validation splits. More workers can increase peak memory. For symptoms and practical controls, see
[Troubleshooting](troubleshooting.md#the-path-search-is-too-slow-or-uses-too-much-memory).

## Result objects and scoring functions

The search exposes immutable result records rather than mutable analysis state. Their fields are
documented below from the public NumPy-style docstrings. `component_path_` contains the
component-count path, `select()` returns one `PiPLSSelection`, `predictor_rank_profile()` exposes the
evaluated ranks and conditional evidence for one component count, and `oof_report()` returns one
`PiPLSOOFReport`.

::: pipls.component_path.PiPLSComponentPath
    options:
      show_signature: false

::: pipls.component_path.PiPLSSelection
    options:
      show_signature: false

::: pipls.component_path.PiPLSPredictorRankProfile
    options:
      show_signature: false

::: pipls.component_path.PiPLSPredictorRankEvidence
    options:
      show_signature: false

::: pipls.validation.PiPLSOOFReport
    options:
      show_signature: false
      members:
        - has_complete_oof_coverage

The positive public loss and its scikit-learn-oriented negative scorer use the same fold-local
response-standardized MSE definition described under
[Candidate coverage and scoring](#scoring-and-conditioned-path-selection).

::: pipls.metrics.response_standardized_mse
    options:
      members: false

::: pipls.metrics.neg_response_standardized_mse
    options:
      members: false
