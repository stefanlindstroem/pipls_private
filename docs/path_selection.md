# Path and selection

`PiPLSSearchCV` searches over pairs of paired-mode count $h$ (`n_components`) and retained
predictor-subspace dimension $r_\pi$ (`predictor_rank`). This page explains how the feasible search
domain is obtained, how candidates are scored, and how the two-dimensional search is reduced to one
retained `PiPLSSelection`. For constructor signatures and fitted attributes, see
[`PiPLSSearchCV`](api/path.md).

## Search domain { #search-bounds }

A `PiPLSRegression` fit uses one fixed pair $(h,r_\pi)$. During cross-validation, the same pair must
be feasible in every training split used to evaluate it. The search domain is therefore the
intersection of the splitwise feasible domains.

For training split $j$, let

- $n_j$ be its number of training observations;
- $p_j$ be its predictor count after any fold-local pipeline preprocessing; and
- $r_j$ be its verified numerical predictor rank after terminal-estimator centering and optional
  scaling.

Define the corresponding minima across all training splits as

\begin{equation}
p'=\min_j p_j,
\qquad
n'=\min_j n_j,
\qquad
r'=\min_j r_j.
\end{equation}

A predictor rank used by the search must be feasible in every split. The hard predictor-rank ceiling
is therefore

\begin{equation}
r_{\pi,\mathrm{hard}}
=
\min(p',n'-1,r').
\end{equation}

The term $n'-1$ reflects the loss of one predictor dimension under centering. The verified rank
$r'$ also captures any additional numerical rank loss after preprocessing, centering, and optional
scaling.

With `max_predictor_rank=None`, the search uses this hard ceiling. An explicit positive integer
`max_predictor_rank=k` adds a user restriction,

\begin{equation}
r_{\pi,\mathrm{max}}
=
\min(r_{\pi,\mathrm{hard}},k).
\end{equation}

Let $q$ be the number of responses, which is common to all training splits. For the ordinary full
rank domain, the admissible integer pairs are

\begin{equation}
\mathcal{D}
=
\{(h,r_\pi)\in\mathbb{N}^2:1\le h\le \min(q,r_{\pi,\mathrm{max}}),\ h\le r_\pi\le r_{\pi,\mathrm{max}}\}.
\end{equation}

The domain $\mathcal{D}$ is triangular: increasing $h$ removes all predictor ranks below $h$.
`predictor_rank_values=None` uses every feasible integer $r_\pi$ in this domain, and
`n_components_values="all"` uses every feasible integer $h$. Explicit component or predictor-rank
values restrict this domain after the same feasibility checks.

The Pulp dataset gives a compact example. Its exhaustive search has $q=8$ and
$r_{\pi,\mathrm{max}}=14$, giving 84 admissible pairs. Each colored cell below is one pair
evaluated under the maintained 10-times repeated five-fold CV protocol; color gives its actual
mean response-standardized CV-MSE. Gray cells violate $r_\pi\ge h$.

![Pulp exhaustive search domain](assets/generated/pulp/search_domain.svg)

These bounds describe feasibility. The EPV policy introduced below is instead a rule for choosing
one predictor rank inside the feasible domain.

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

Adaptive coverage first refines around the exact evaluated reference. If the smallest qualifying
evaluated rank and its immediately lower failing evaluated neighbor still bracket unevaluated
admissible ranks, the search bisects that tolerance boundary and evaluates the remaining interval
exhaustively once it contains at most five admissible ranks. Both adaptive refinement stages use
this same private five-rank switch. If boundary evaluation changes the exact evaluated reference,
reference refinement resumes before the tolerance boundary is resolved. Predictor-rank tolerance
can therefore add evaluated candidates under adaptive coverage without making the search globally
exhaustive.

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

The exact split reuse required for OOF reporting is documented under
[Selection and split provenance](oof_diagnostics.md#selection-and-split-provenance).

Π-PLS provides no dedicated leave-one-out mode or package-specific interpretation for singleton
validation folds. Foldwise $R^2$ is rejected when a validation split contains fewer than two
observations.

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

The search stores candidate evidence but no final fitted model.

## Computational consequences { #computational-consequences }

If $N_{\mathrm{split}}$ validation splits and $N_{\mathrm{pair}}$ candidate pairs are materialized,
candidate evaluation performs

\begin{equation}
N_{\mathrm{candidate\ fit}}
=
N_{\mathrm{pair}}N_{\mathrm{split}}
\end{equation}

fold-local fits, in addition to fold-level predictor-rank feasibility probes. A final `refit()` adds
one full-data fit. OOF-specific fitting cost is documented under
[OOF computation](oof_diagnostics.md#oof-computation).

Narrowing the rank or component domain changes the model-selection problem. Adaptive coverage
changes which interior ranks are evaluated. Fewer validation splits change the validation evidence.
These are not computationally neutral shortcuts.

`n_jobs` changes execution rather than the requested candidates or validation splits. Candidate
pairs are parallel tasks during search. More workers can increase peak memory. For symptoms and
practical controls, see
[Troubleshooting](troubleshooting.md#the-path-search-is-too-slow-or-uses-too-much-memory).

## Result objects and scoring functions

The search exposes immutable path and selection records rather than mutable analysis state. Their
fields are documented below from the public NumPy-style docstrings. `component_path_` contains the
component-count path, `select()` returns one `PiPLSSelection`, and `predictor_rank_profile()`
exposes the evaluated ranks and conditional evidence for one component count.

::: pipls.component_path.PiPLSComponentPath
    options:
      show_signature: false
      members: false

::: pipls.component_path.PiPLSSelection
    options:
      show_signature: false
      members: false

::: pipls.component_path.PiPLSPredictorRankProfile
    options:
      show_signature: false
      members: false

::: pipls.component_path.PiPLSPredictorRankEvidence
    options:
      show_signature: false
      members: false

The positive public loss and its scikit-learn-oriented negative scorer use the same fold-local
response-standardized MSE definition described under
[Candidate coverage and scoring](#scoring-and-conditioned-path-selection).

::: pipls.metrics.response_standardized_mse
    options:
      members: false

::: pipls.metrics.neg_response_standardized_mse
    options:
      members: false
