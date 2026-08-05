# Decision 0146: CV-MSE tolerance selection and split-SD reporting

## Status

Accepted and implemented. All seven patches are complete. The active API uses tolerance-based
minimum-CV-MSE selection and descriptive split SD, and the complete Pulp workflow and Tutorial 3 use
ten repeated five-fold partitions.

## Context

Before this transition, the component-path API supported a fold-based SE rule. It derived a
standard error from the population standard deviation of the realized split losses and used the
minimum-CV-MSE row plus one such standard error as a component-count threshold. This is a familiar
heuristic for one ordinary cross-validation partition, but its uncertainty interpretation becomes
increasingly weak when a splitter materializes repeated folds. For example, repeated five-fold CV
with ten repetitions produces 50 correlated validation results rather than 50 independent
replicates. Treating the split count as independent evidence can make the threshold artificially
narrow.

The package must support several operating modes without making the API depend on how many
repetitions happen to be available. It also needs a computationally transparent route: repetitions
should improve the stability of mean candidate performance and OOF predictions, while selection
should not require a fragile estimate of the standard error of correlated CV results.

The existing `minimum_cv_mse` rule already provides the appropriate basis. The missing capability is
an explicit user-controlled tolerance that trades a bounded increase in observed mean CV-MSE for a
smaller component count. Split-to-split variability remains useful in plots, but it should be
reported descriptively as a standard deviation rather than used as an inferential selection
threshold.

## Decision

### Aggregate every materialized validation split equally

For component-path row $h$, let $M_{jh}$ denote the response-standardized validation MSE on
materialized split $j$, for $j=1,\ldots,J$. The path statistics are

\begin{equation}
M_h
=
\frac{1}{J}
\sum_{j=1}^{J} M_{jh}
\end{equation}

and

\begin{equation}
S_h
=
\sqrt{
\frac{1}{J}
\sum_{j=1}^{J}
\left(M_{jh}-M_h\right)^2
}.
\end{equation}

Thus `cv_mse_mean` is the arithmetic mean over all materialized split losses and `cv_mse_std` is
the population standard deviation over those same losses (`ddof=0`). When repeated $K$-fold CV is
used, all $K R$ split losses enter these quantities equally.

The SD is descriptive. Maintained figures label it as:

> Mean CV-MSE ± SD across validation splits.

It is not a standard error, confidence interval, or model-selection threshold.

### Select by minimum CV-MSE with two tolerances

Let

\begin{equation}
M_{\min}=\min_h M_h.
\end{equation}

A component-path row qualifies only when both conditions hold:

\begin{equation}
M_h \leq \left(1+\delta_{\mathrm{rel}}\right)M_{\min}
\end{equation}

and

\begin{equation}
M_h \leq M_{\min}+\delta_{\mathrm{abs}}.
\end{equation}

Equivalently, the effective threshold is

\begin{equation}
T
=
\min\left[
\left(1+\delta_{\mathrm{rel}}\right)M_{\min},
M_{\min}+\delta_{\mathrm{abs}}
\right].
\end{equation}

The rule selects the smallest evaluated component count satisfying `cv_mse_mean <= T`. The selected
predictor rank is the rank already associated with that component-path row. Decision 0148 defines
how optimized ranks are tolerance-conditioned before this component path is formed. The path remains
ordered by ascending component count, so the first qualifying row is the selected row.

The public arguments are:

```python
search.select(
    rule="minimum_cv_mse",
    relative_tolerance=None,
    absolute_tolerance=np.inf,
)

search.refit(
    X,
    Y,
    rule="minimum_cv_mse",
    relative_tolerance=None,
    absolute_tolerance=np.inf,
)
```

`relative_tolerance=None` resolves to

```python
sqrt(np.finfo(np.float64).eps)
```

which is approximately $1.49\times10^{-8}$. This default treats floating-point-scale differences as
numerically equivalent without introducing substantive parsimony. A user requests a meaningful
tradeoff explicitly, for example `relative_tolerance=0.10`.

`absolute_tolerance=np.inf` disables the absolute cap by default. A finite absolute tolerance may be
supplied when the user wants both a relative and an absolute restriction. Maintained examples need
not demonstrate the absolute option; public documentation must state its semantics.

Relative tolerance must be finite and nonnegative. Absolute tolerance must be nonnegative and may
be positive infinity. NaN and negative infinity are invalid. Nondefault tolerance requests apply
only to `rule="minimum_cv_mse"`.

### Retain complete selection provenance

A tolerance-based minimum-CV-MSE selection retains:

```python
selection.relative_tolerance
selection.absolute_tolerance
selection.reference_minimum
selection.cv_mse_threshold
```

The stored tolerance values are the resolved numeric values. `reference_minimum` is the exact
minimum-CV-MSE path row used to form the threshold and must not contain nested component-count
tolerance provenance. Under Decision 0148 it retains independent predictor-rank evidence when the
path policy is optimized. `cv_mse_threshold` is derived from the reference minimum and the two
component-count tolerances; it is not stored as independent state.

Manual component-count selections and `rule="best_score"` selections do not carry tolerance
provenance.

### Remove the one-standard-error surface

After maintained consumers migrate, remove:

```text
rule="one_standard_error"
cv_mse_standard_error
one_standard_error_threshold
_select_one_standard_error()
```

No compatibility alias or deprecation period is required before the first release. Historical
decisions and changelog entries may retain historically accurate wording.

### Demonstrate substantive relative tolerance in Tobacco

The Tobacco example becomes the maintained demonstration of deliberate parsimony:

```python
model = search.refit(
    X,
    Y,
    rule="minimum_cv_mse",
    relative_tolerance=0.10,
)
```

Its component-path figure shows mean CV-MSE ± split SD, the exact minimum row, the 10% threshold,
and the smallest qualifying component count. The example mentions the additional absolute cap in
prose only; it does not demonstrate a second tolerance configuration.

### Use repeated CV in Tutorial 3 and its Pulp experiment

The complete Pulp analysis and Tutorial 3 use:

```python
RepeatedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=0,
)
```

The search therefore materializes 50 splits. Candidate means and SDs use all 50 split losses. Each
Pulp observation normally receives ten OOF predictions, which are averaged in `oof_report()` and
recorded through `oof_prediction_counts == 10`.

The tutorial must state that this search is approximately ten times as expensive as the former
single five-fold partition. Repetition is introduced only in the complete Pulp workflow; the quick
start and other maintained examples remain computationally lighter unless separately decided.

### Keep outer evaluation out of scope

This transition changes inner path selection and descriptive reporting only. It does not introduce a
nested-CV or repeated outer-evaluation class. Independent evaluation of the complete
search-select-refit procedure remains a separate future design problem.

## Patch sequence

1. Establish this decision and synchronize the guide layer — complete.
2. Rename `cv_mse_fold_sd` to `cv_mse_std` and lock equal-split mean and population-SD numerics,
   temporarily retaining derived SE support as a migration bridge — complete.
3. Add relative and absolute tolerance arguments, validation, threshold selection, and immutable
   selection provenance — complete.
4. Replace maintained SE error bars and wording with SD across validation splits — complete.
5. Migrate maintained automatic workflows to `minimum_cv_mse`; use `relative_tolerance=0.10` in
   Tobacco and machine-scale defaults elsewhere — complete.
6. Use repeated five-fold CV with ten repetitions in the Pulp example, renderer, Tutorial 3, and
   corresponding tests and manifests — complete.
7. Remove the one-standard-error rule and all remaining active SE surface, synchronize public
   documentation and guide material, and run complete stale-surface audits — complete.

## Validation

Every implementation patch must pass:

```text
git diff --check
Ruff
mypy
complete pytest suite
strict MkDocs build
Python compilation
```

Patch 2 must protect exact split means and population SDs for ordinary, repeated, one-split, and
unequal-length splitter protocols. Patch 3 must protect default resolution, zero and finite
relative tolerance, finite and infinite absolute tolerance, both-conditions semantics, exact
threshold inclusion, zero minimum MSE, invalid inputs, custom scorers, pipelines, pickling, and OOF
selection compatibility. Patch 4 must execute affected renderers and verify SD labels and arrays.
Patch 5 must protect Tobacco's exact minimum, 10% threshold, first qualifying row, and selection
provenance. Patch 6 must protect 50 materialized splits, ten OOF predictions per Pulp observation,
semantic manifests, generated figures, and deliberate runtime documentation. Patch 7 must verify
that no active one-standard-error or standard-error selection surface remains.

## Consequences

Selection is defined by observed mean predictive error plus explicit user tolerances rather than an
uncertain SE heuristic. The same API works for one fold partition, repeated CV, leave-one-out, and
custom split iterables. SD error bars remain useful descriptive evidence without implying
independent resamples. The default remains practically minimum-CV-MSE selection, while users can
request transparent parsimony through relative tolerance and optionally impose an absolute cap.
