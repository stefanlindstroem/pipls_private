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

These bounds describe feasibility. Selection begins only after the admissible pairs have been
defined.

## Exhaustive search and selection { #exhaustive-search-and-selection }

With `predictor_rank_values=None` and `search_method="exhaustive"`, every pair in $\mathcal{D}$
is evaluated. Under the default scorer, each cell in the Pulp figure therefore contains one mean
response-standardized CV-MSE, denoted $M_{h,r_\pi}$ below. Lower values are better.

### Conditional predictor-rank selection

Predictor rank is selected separately at each component count. For fixed $h$, let

\begin{equation}
M_{h,\min}=\min_{r_\pi:(h,r_\pi)\in\mathcal{D}} M_{h,r_\pi}.
\end{equation}

A predictor rank qualifies when its mean CV-MSE satisfies both

\begin{equation}
M_{h,r_\pi} \le (1+\delta_{\pi,\mathrm{rel}})M_{h,\min},
\qquad
M_{h,r_\pi} \le M_{h,\min}+\delta_{\pi,\mathrm{abs}}.
\end{equation}

The smallest qualifying $r_\pi$ is retained. Thus the tolerance can trade a small increase in
validation error for a smaller predictor subspace without changing which candidates were evaluated.
`predictor_rank_relative_tolerance=None` resolves to `sqrt(np.finfo(np.float64).eps)`, while an
infinite `predictor_rank_absolute_tolerance` disables the absolute cap.

The next figure applies the same rule to the already evaluated Pulp surface. To make the mechanism
visible, it uses a deliberately generous 15% relative predictor-rank tolerance and no absolute cap.
Open circles mark the exact minimum at each $h$; diamonds mark the smallest qualifying rank. This
illustrative tolerance does not require another exhaustive CV run because candidate evaluation is
independent of the tolerance.

![Pulp conditional predictor-rank selection](assets/generated/pulp/conditioned_search_domain.svg)

Write the retained rank as $\hat r_\pi(h)$. The two-dimensional search has now been reduced to one
retained pair $(h,\hat r_\pi(h))$ for each component count.

### Component path and component selection

The retained pairs form the component path. Define its mean CV-MSE at component count $h$ as
$M_h=M_{h,\hat r_\pi(h)}$ and let

\begin{equation}
M_{\min}=\min_h M_h.
\end{equation}

For `rule="minimum_cv_mse"`, a component count qualifies when

\begin{equation}
M_h \le (1+\delta_{\mathrm{rel}})M_{\min},
\qquad
M_h \le M_{\min}+\delta_{\mathrm{abs}}.
\end{equation}

Because the path is ordered by increasing $h$, the first qualifying row is retained.
`relative_tolerance=None` again resolves to `sqrt(np.finfo(np.float64).eps)`, and an infinite
`absolute_tolerance` disables the absolute cap.

For visibility, the Pulp component-path figure below continues the 15% predictor-rank illustration
and then applies a deliberately generous 50% relative component tolerance. The exact path minimum
is at $h=3$, while the smaller $h=2$ model falls inside that illustrative tolerance and is retained.
The labels above the path show the predictor rank already selected conditionally at each $h$.

![Pulp conditioned component-path selection](assets/generated/pulp/conditioned_component_path.svg)

This ordering is central: predictor rank is resolved first at each $h$; component selection then
acts on the resulting one-dimensional path and does not revisit the predictor-rank profiles.

## Fixed predictor rank: EPV { #epv-policy }

The exhaustive case above optimizes predictor rank separately at every component count. The EPV
policy instead fixes one predictor rank before the component path is evaluated. With
`predictor_rank_values="epv"`, the nominal rank is

\begin{equation}
r_{\pi,\mathrm{epv,nominal}}
=
\min\left(p,\left\lceil\frac{n}{c}\right\rceil\right),
\end{equation}

where $n$ and $p$ are the full-data observation and predictor counts supplied to `fit()`, and $c$ is
`samples_per_predictor_rank`. The effective EPV rank is

\begin{equation}
r_{\pi,\mathrm{epv}}
=
\min(r_{\pi,\mathrm{epv,nominal}},r_{\pi,\mathrm{max}}).
\end{equation}

The default is $c=10$. For Pulp, $n=46$ and $p=14$, so the nominal rank is 5 and no feasibility
clipping is needed. The search therefore evaluates only the five compatible pairs
$(h,r_\pi)=(1,5),\ldots,(5,5)$. In the figure below, the full feasible domain is left neutral and
only those EPV pairs are colored by their actual mean response-standardized CV-MSE under the same
validation splits used in the exhaustive example.

![Pulp EPV search domain](assets/generated/pulp/epv_search_domain.svg)

There is no conditional predictor-rank selection in this case: $r_\pi$ is already fixed, and only
the component path remains to be selected. The default $c=10$ may be changed with
`samples_per_predictor_rank`; $c=5$ is a more permissive maintained setting. Values below 5 are
legal but emit `PredictorRankSupportWarning`. A nondefault `samples_per_predictor_rank` is invalid
outside the EPV policy.

### Other predictor-rank policies { #predictor-rank-policies }

A one-element explicit `predictor_rank_values` sequence also fixes one rank and therefore has no
conditional predictor-rank selection. A longer explicit sequence restricts the predictor-rank
domain and optimizes only over those supplied ranks, while `predictor_rank_values=None` uses the
full feasible integer domain described above. `n_components_values` restricts the component counts
and `max_predictor_rank` restricts the upper predictor-rank boundary.

## Scoring and adaptive coverage { #scoring-and-conditioned-path-selection }

The exhaustive example above uses the default scorer, `"neg_response_standardized_mse"`, which
resolves to `pipls.metrics.neg_response_standardized_mse`. Maximizing that score is equivalent to
minimizing the response-standardized CV-MSE shown in the figures. Ordinary scikit-learn scorer
names, scorer callables, and `scoring=None` are also accepted. With another scorer, the conditional
predictor-rank optimum and tolerance are defined on the configured-score scale; CV-MSE remains a
diagnostic and need not identify that optimum.

`search_method="adaptive"` uses the same admissible endpoints as exhaustive search but may leave
interior predictor ranks unevaluated. It first refines around the exact evaluated score optimum. If
the smallest qualifying evaluated rank and its immediately lower failing neighbor still bracket
unevaluated ranks, it refines that tolerance boundary as well. Each refinement switches to
exhaustive evaluation once at most five admissible ranks remain in the local interval. If a boundary
evaluation changes the exact evaluated optimum, optimum refinement resumes before the tolerance
boundary is finalized.

`search_is_exhaustive_` records whether every admissible pair was evaluated, and `cv_results_` is
the complete record of the pairs that were actually evaluated. `PiPLSPredictorRankEvidence` stores
the exact configured-score reference and resolved predictor-rank tolerances;
`predictor_rank_profile(h).reference_selection` exposes that reference, while `.selection` exposes
the retained tolerance-qualified rank. Under adaptive coverage these statements necessarily concern
the evaluated candidates rather than unevaluated ranks.

`rank_test_score` uses minimum ranks with private `rtol=1e-12` and `atol=1e-15` comparisons; tied
score groups are anchored to the leading score in each group rather than chained through adjacent
values. Public parsimony tolerances do not change these candidate-level ranks.

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
mutating the search. `rule="minimum_cv_mse"` applies the component-path tolerance described above
and stores the exact path minimum as `reference_minimum` together with the resolved tolerances and
`cv_mse_threshold`.

`rule="best_score"` instead chooses the maximum configured-score row on the conditioned component
path. Direct lookup by component count and `best_score` carry no CV-MSE tolerance provenance. In all
cases the predictor rank is the rank already retained conditionally for that component count;
`select()` does not revisit the predictor-rank profile.

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
[Scoring and adaptive coverage](#scoring-and-conditioned-path-selection).

::: pipls.metrics.response_standardized_mse
    options:
      members: false

::: pipls.metrics.neg_response_standardized_mse
    options:
      members: false
