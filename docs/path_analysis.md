# Path-selection details

The [synthetic tutorial](tutorials/synthetic.md#retrieve-selection-evidence) owns the routine
selection workflow. This page records configuration, cross-validation, and edge behavior needed
when the defaults are not enough. Exact signatures and fitted attributes are in the
[generated path API](api/path.md).

## Search bounds

For paired-mode count $h$ (`n_components`) and retained predictor-subspace dimension $r_\pi$
(`predictor_rank`), the ordinary automatic search domain is triangular. See
[Interpretation of the two rank controls](theory.md#interpretation-of-the-ranks) for their distinct
modeling roles. The predictor-rank ceiling
is determined first from fold-local feasibility and an optional explicit user cap; component counts
are then resolved from the ranks actually available under the chosen predictor-rank policy.

### Hard predictor-rank ceiling

Let $p_{\mathrm{min}}$ be the minimum predictor count after fold-local pipeline preprocessing,
$n_{\mathrm{train,min}}$ the smallest materialized training-fold size, and
$r_{\mathrm{num,min}}$ the minimum verified predictor numerical rank after terminal-estimator
centering and optional scaling. The hard ceiling is

\begin{equation}
r_{\pi,\mathrm{hard}}
=
\min\left[p_{\mathrm{min}},n_{\mathrm{train,min}}-1,r_{\mathrm{num,min}}\right].
\end{equation}

These are feasibility constraints, not statistical-support heuristics. With
`max_predictor_rank=None`, the effective automatic-search ceiling is
$r_{\pi,\mathrm{max}}=r_{\pi,\mathrm{hard}}$. An explicit positive integer
`max_predictor_rank=k` imposes the additional user restriction

\begin{equation}
r_{\pi,\mathrm{max}}
=
\min\left[r_{\pi,\mathrm{hard}},k\right].
\end{equation}

The events-per-variable parameter `samples_per_predictor_rank` does not enter this general ceiling.
It is used only by the explicit EPV policy described below.

With `predictor_rank_values=None`, the admissible ranks for a component count $h$ are

\begin{equation}
\mathcal{R}_h
=
\{h,h+1,\ldots,r_{\pi,\mathrm{max}}\}.
\end{equation}

If the response matrix has $q$ columns, the default `n_components_values="all"` resolves after the
predictor-rank policy and evaluates every $h$ from one through the smaller of $q$ and the largest
rank available under that policy. Explicit component or predictor-rank values outside the resolved
feasible domain are rejected before candidate evaluation. If an explicit predictor-rank sequence
leaves a requested component count without any rank satisfying $h\le r_\pi$, the request is
rejected rather than silently dropping that component count.

### Component-count requests

An explicit integer sequence requests a subset of the available paired-mode counts:

```python
search = PiPLSSearchCV(
    n_components_values=[1, 2, 3, 5],
).fit(X, Y)
```

Every requested value must have at least one admissible predictor rank.

## Predictor-rank policies

`predictor_rank_values` controls the rank domain or fixed-rank policy; `search_method` separately
controls how a multi-rank domain is covered. The supported cases are:

- `None`: optimize predictor rank over every integer from one through
  `max_predictor_rank_`, subject to $r_\pi\ge h$ for each component count;
- a one-element integer sequence: fix that rank across all compatible component counts;
- a longer integer sequence: optimize over exactly those supplied ranks after feasibility checks;
- `"epv"`: fix one rank using the events-per-variable-inspired rule.

For EPV, the nominal rank is computed from the full number of observations supplied to `fit()`:

\begin{equation}
r_{\pi,\mathrm{epv,nominal}}
=
\min\left[p,\left\lceil\frac{n}{c}\right\rceil\right],
\end{equation}

where $c$ is `samples_per_predictor_rank`. The effective EPV rank is then clipped only by the hard
ceiling and any explicit integer `max_predictor_rank`:

\begin{equation}
r_{\pi,\mathrm{epv}}
=
\min\left[r_{\pi,\mathrm{epv,nominal}},r_{\pi,\mathrm{max}}\right].
\end{equation}

The EPV default is $c=10$. A more permissive $c=5$ is also supported. Values below 5 are legal but
emit `PredictorRankSupportWarning`; for example, `predictor_rank_values="epv"` with
`samples_per_predictor_rank=1` deliberately pushes the nominal EPV rank to $\min(p,n)$ before
fold-local feasibility clipping. A nondefault `samples_per_predictor_rank` is invalid outside the
EPV policy so it cannot be mistaken for a hidden automatic-search bound.

`search_method="exhaustive"` is the default. It evaluates every admissible pair in a multi-rank
domain and therefore gives the complete configured-score reference optimum over that declared
domain. `search_method="adaptive"` is an explicit computational approximation: it uses the same
admissible endpoints but may leave interior ranks unevaluated. After fitting,
`search_is_exhaustive_` states whether every admissible pair was actually evaluated. Fixed and EPV
policies have one rank per compatible component count, so their candidate coverage is exhaustive
even when `search_method` is left at its default.

Adaptive refinement and `rank_test_score` continue to use private numerical tie handling around the
exact configured-score optimum among evaluated candidates. Final conditional rank retention then
applies the separate public predictor-rank tolerances.

For optimized policies, configure those tolerances on the search constructor:

```python
search = PiPLSSearchCV(
    predictor_rank_relative_tolerance=0.10,
    predictor_rank_absolute_tolerance=np.inf,
).fit(X, Y)
```

For each component count, the smallest evaluated rank satisfying both configured-score caps is
retained. Fixed and EPV policies accept only the default predictor-rank tolerances and have no
`PiPLSPredictorRankEvidence`. Under exhaustive coverage the reference optimum is taken over the
complete admissible rank domain; under adaptive coverage it is necessarily limited to the ranks
that were evaluated.

## Scoring and conditioned path selection { #scoring-and-conditioned-path-selection }

Candidate evaluation uses the configured mean test score. The default scoring parameter is
the stable package name `"neg_response_standardized_mse"`, which resolves to
`pipls.metrics.neg_response_standardized_mse`. Maximizing that score is equivalent
to minimizing mean response-standardized CV-MSE. With another scorer, the CV-MSE columns remain
diagnostics and need not identify the selected candidate.

Ordinary scikit-learn scorer names, scorer callables, and `scoring=None` are accepted. Candidate-level
`cv_results_` remains unchanged by parsimony tolerances. `rank_test_score` uses minimum ranks with
private `rtol=1e-12` and `atol=1e-15` numerical comparison. Every rank-1 candidate is tied directly
with the maximum score; lower rank groups are likewise anchored to their leading score rather than
formed through adjacent-score chaining. Adaptive search makes no claim about ranks it did not
evaluate.

For each component count, `PiPLSPredictorRankEvidence` records the exact configured-score reference
rank, its score and CV-MSE summary, the resolved tolerances, and the derived `score_threshold`.
`predictor_rank_profile(h).reference_selection` exposes the exact optimum, while `.selection`
exposes the retained tolerance-qualified rank. `search.select(rule="best_score")` then chooses the
maximum configured-score row on this conditioned component path, not an unretained global candidate.

## Pipelines and fold-local preprocessing

The supported estimator is either a direct `PiPLSRegression` or a scikit-learn `Pipeline` whose
final step is `PiPLSRegression`. The complete estimator is cloned and fitted inside every fold.
For the direct-template pattern and inherited settings, see
[Configure the candidate estimator](api/path.md#configure-the-candidate-estimator).

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from pipls import PiPLSRegression, PiPLSSearchCV

pipeline = Pipeline(
    [
        ("impute", SimpleImputer()),
        ("pipls", PiPLSRegression(n_components=1, predictor_rank=1)),
    ]
)
search = PiPLSSearchCV(estimator=pipeline).fit(X, Y)
```

The terminal estimator needs the valid construction seed pair `(1, 1)` because
`PiPLSRegression` always represents one explicit fixed pair. `PiPLSSearchCV` replaces both values
before fold-rank preflight, every candidate fit, explicit OOF reporting, and each full-data refit,
so the seed pair does not restrict or select the path. Other template settings, including `scale`,
`svd_solver`, and `random_state`, do affect candidate fitting.

Do not fit learned preprocessing on the complete dataset before path evaluation.

## Cross-validation protocols and metadata

`PiPLSSearchCV` accepts scikit-learn-compatible splitters and explicit split iterables for
grouped, repeated, predefined, temporal, or other protocols when their scientific assumptions match
the data. Suitable examples include `GroupKFold`, `RepeatedKFold`, `PredefinedSplit`, and
`TimeSeriesSplit`.

```python
from sklearn.model_selection import GroupKFold

search = PiPLSSearchCV(cv=GroupKFold(n_splits=5))
search.fit(X, Y, groups=sample_groups)
```

`groups` is an explicit `PiPLSSearchCV.fit()` parameter and participates in scikit-learn metadata
routing when routing is enabled and requested. Split metadata belongs to the search object;
`PiPLSRegression.fit(X, Y)` fits one explicit pair and accepts none.

## Ordered out-of-fold predictions

Request ordered OOF diagnostics for an existing selection before optional final refitting:

```python
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)
search = PiPLSSearchCV(cv=cv).fit(X, Y)
selection = search.select(rule="minimum_cv_mse")
report = search.oof_report(X, Y, selection=selection)
model = search.refit(X, Y, selection=selection)
```

A workflow whose purpose is only validation may stop after the report. Manual lookup uses the same
selection handoff:

```python
selection = search.select(n_components=4)
report = search.oof_report(X, Y, selection=selection)
```

The method validates that the supplied selection belongs to the fitted search, fits that fixed
parameterization once per stored training fold, and returns an immutable report:

- `selection` is the exact supplied component-count and predictor-rank selection;
- `oof_predictions` preserves input row order;
- repeated validation predictions are averaged and their counts are recorded;
- rows without validation coverage have count 0 and a NaN prediction;
- selection metrics remain available through `report.selection`;
- `pooled_oof_r2` uses only rows with OOF coverage.

`fit()` stores defensive read-only copies of the exact materialized validation indices. Therefore an
iterable splitter is not consumed a second time and a stochastic splitter is not asked to generate a
new partition. `oof_report()` requires the same sample count, feature count, and response-column
count as the fitted search, but it does not retain or compare original values. The caller is
responsible for passing the same observations in the same row order.

`has_complete_oof_coverage` records whether every row received at least one validation
prediction. These results are selection-conditioned because the same path search produced the
supplied selection. Use nested cross-validation or an external test set when an unbiased
post-selection estimate is required.

## Validation-protocol boundary

Π-PLS does not provide a dedicated leave-one-out mode, provenance flag, example, or compatibility
guarantee. Users may still intentionally supply any splitter or explicit split iterable accepted by
the generic `cv` interface, including protocols with singleton validation folds, but they own the
protocol choice and interpretation.

Foldwise $R^2$ is rejected whenever a validation split contains fewer than two observations. Use a
scorer defined for the realized validation-set sizes. Pooled OOF $R^2$, when available, remains a
single statistic over covered rows and is not mean foldwise $R^2$.

## Split variation and tolerance selection

`component_path_.cv_mse_std` is the population standard deviation of the realized split-specific
MSE values. It describes variation across the materialized validation splits. Maintained
component-path and predictor-rank-profile figures plot mean response-standardized CV-MSE with
symmetric $\pm 1$ SD bars from this stored quantity. The bars are descriptive split-to-split
variability; they are not confidence intervals and do not enter selection. The path and profile
objects expose no standard-error property.

### Search-owned selection rules

`PiPLSSearchCV.select()` returns complete immutable stored rows without fitting:

```python
minimum = search.select(rule="minimum_cv_mse")
tolerant = search.select(
    rule="minimum_cv_mse",
    relative_tolerance=0.02,
    absolute_tolerance=np.inf,
)
best = search.select(rule="best_score")
manual = search.select(n_components=3)
```

For the minimum-CV-MSE rule, let $M_{\min}$ be the exact minimum stored mean CV-MSE. A row qualifies
only when its mean is no larger than both $(1+\delta_{\mathrm{rel}})M_{\min}$ and
$M_{\min}+\delta_{\mathrm{abs}}$. The first qualifying row is returned because component counts
are stored in strictly ascending order. `relative_tolerance=None` resolves to
`sqrt(np.finfo(np.float64).eps)`, while positive-infinity `absolute_tolerance` disables the absolute
cap. The result retains the exact unruled minimum row as `reference_minimum`, the resolved
tolerances, and the derived `cv_mse_threshold`. Direct lookup by component count and `best_score`
selection carry no tolerance provenance.

The associated predictor rank is the rank already retained conditionally for that component count
under the configured scorer and constructor-level predictor-rank tolerances. Its
`predictor_rank_evidence` remains attached to direct, named-rule, refitted-model, and OOF selections.
`select()` does not revisit the predictor-rank profile, fit or refit an
estimator, mutate the search object, or attach selected state. With a nondefault scorer, the stored
predictor rank need not minimize CV-MSE within its component-count profile. Evidence-retaining
model-producing workflows create one selection and pass that same object to OOF reporting and final
refitting. After a successful fit, `model.selection_` confirms the fitted model's exact provenance.
`component_path_` remains the aligned numerical curve.

The [Tobacco](datasets.md#tobacco-spectral-integration) workflow demonstrates the hierarchy with
separate 10% relative tolerances. The search
constructor applies `predictor_rank_relative_tolerance=0.10` within every component count, and
`search.select(rule="minimum_cv_mse", relative_tolerance=0.10)` then acts on the resulting
conditioned component path. The predictor-rank profile shows its exact configured-score reference,
converted CV-MSE threshold, and
retained rank; the component path shows its own exact minimum, threshold, and retained component
count. Predictor-rank profile error bars use the stored split SD.

## Final-model refitting

After path evaluation, `refit()` fits one stored component-path row on the supplied full data. In
an evidence-retaining workflow, create the row first and pass the exact object forward:

```python
search = PiPLSSearchCV(search_method="adaptive").fit(X, Y)
selection = search.select(rule="minimum_cv_mse")
model = search.refit(X, Y, selection=selection)
Y_pred = model.predict(X_new)
```

Rule-based and manual component-count refitting remain compact alternatives:

```python
model_by_rule = search.refit(X, Y, rule="minimum_cv_mse")
model_by_count = search.refit(X, Y, n_components=4)
```

Exactly one of `selection`, `rule`, and `n_components` is required. The accepted named rules are:

- `rule="best_score"`, the maximum configured-score row on the conditioned component path;
- `rule="minimum_cv_mse"`, the smallest stored component-path row satisfying simultaneous relative
  and absolute tolerances around the exact minimum mean response-standardized CV-MSE.

Each rule retains the predictor rank already selected conditionally for the chosen component count.
With a nondefault scorer, that rank remains conditioned on the scorer even when the component rule
uses response-standardized CV-MSE. Relative tolerance must be finite and nonnegative; absolute
tolerance must be nonnegative and may be positive infinity. Nondefault tolerance arguments apply
only to `rule="minimum_cv_mse"`; a supplied selection has already resolved those controls.

`refit()` validates an existing selection against the fitted search or resolves one from the compact
rule/count arguments. It then clones the configured direct estimator or pipeline, replaces the
terminal Π-PLS rank pair, fits the clone, attaches the exact immutable row as `model.selection_`,
and returns the model.
It does not mutate the search, store the supplied matrices, or attach the model to search state.
Prediction, transformation, scoring, inverse transformation, and feature-name behavior belong to
the returned model.

`oof_report()` consumes an existing selection and does not alter search state. The search stores
candidate evidence and reusable split indices, but no report or fitted final model.

## Search and model diagnostics

Use `component_path_` for the concise component-count curve and `predictor_rank_profile(h)` for the
evaluated ranks at one count. `cv_results_` contains aligned `n_components` and `predictor_rank`
arrays, split and summary test scores, response-standardized MSE diagnostics, score ranks, and
fit/score timing summaries. The direct parameter columns are stable for both direct estimators and
pipelines, and the dictionary contains only evaluated candidates.

Use the returned estimator for fixed-model diagnostics such as `decomposition_`, coefficients,
latent scores, predictions, and inspection helpers. Retain the search variable when both search
evidence and the final fitted model are needed.
