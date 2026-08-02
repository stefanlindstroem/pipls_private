# Decision 0139: three-stage user onboarding

## Status

Accepted; implementation in progress.

## Context

The implemented API now supports a particularly clear progression. A user can obtain a final model
with one chained search and refit expression, retain the search object when selection evidence
matters, and request selection-conditioned out-of-fold diagnostics explicitly when validation
matters. The package-owned Pulp loader also makes the shortest workflow available from an installed
package without a repository checkout.

The executable examples already demonstrate these operations, but the rendered documentation does
not present them in the order a new user naturally encounters them. The Pulp fitted-value quick
start is a source script and a short landing-page snippet rather than a rendered tutorial. The
served tutorial route still begins with the more detailed synthetic path-selection analysis, which
calls itself the first and shortest workflow. The path API reference likewise begins with manual
inspection rather than the automatic final-model route.

The package remains unreleased at version `0.0.0`. The onboarding surface can therefore be renamed
and reordered directly without compatibility filenames, duplicate examples, redirect pages, or
legacy tutorial navigation.

## Decision

Present the public workflow in three stages.

### 1. Automatic final model

The first rendered tutorial uses the package-owned Pulp dataset and the shortest automatic
workflow:

```python
from pipls import PiPLSSearchCV
from pipls.datasets import load_pulp

X, Y = load_pulp(return_X_y=True)

model = PiPLSSearchCV().fit(X, Y).refit(
    X,
    Y,
    rule="one_standard_error",
)
```

The tutorial predicts the same observations used for the final full-data fit, standardizes the
response columns for one combined observed-versus-fitted plot, and reports response-wise
standardized fitted RMSE. It must call these values **fitted values**, **full-data fitted
predictions**, or **calibration fit**. It must state that they are not out-of-fold validation.

The maintained source becomes:

```text
examples/01_pulp_quick_start.py
```

The former `examples/01_minimal_fit_and_plot.py` name is removed rather than retained as an alias.
A dedicated renderer generates one tutorial SVG and a small semantic manifest from the same public
workflow without importing or executing the artifact-writing example.

### 2. Inspect and decide

The synthetic tutorial becomes the second stage and is titled around its actual purpose: retaining
the fitted search object, inspecting `component_path_` and
`predictor_rank_profile(n_components=...)`, choosing a component count, and fitting the stored
pair through:

```python
model = search.refit(X, Y, n_components=chosen_n_components)
```

Its introduction must explain that it expands the quick start by retaining the search evidence. It
must not describe itself as the first Pi-PLS model or the shortest complete path.

### 3. Validate and interpret

The complete Pulp tutorial remains the third stage. It owns real-data selection qualification,
explicit final refitting, selection-conditioned OOF reporting through `validation_report()`, and
representative model-interpretation figures. Pulp OOF predictions must not be described as
fixed-parameter OOF because the selected row is chosen using the same observations.

### Navigation and reference order

The served tutorial navigation is:

```text
1. Quick start with Pulp
2. Inspect and select with synthetic data
3. Complete Pulp analysis
```

The documentation landing page presents the same conceptual order before detailed navigation:

1. discard the temporary search when only the automatic final model matters;
2. retain the search when path evidence and manual selection matter;
3. call `validation_report()` when selection-conditioned OOF diagnostics matter.

The path API reference begins with the shortest automatic workflow, then manual path and profile
inspection, then selected-row validation. Its wording must make clear that `refit()` resolves and
transfers the complete stored pair $(h,r_\pi^*(h))$ internally; ordinary users do not manually copy
`predictor_rank` into another estimator.

### Documentation ownership

The quick-start tutorial owns the first successful fitted-value workflow and its single generated
figure. The synthetic tutorial owns inspect-decide-refit mechanics on controlled data. The complete
Pulp tutorial owns selection-conditioned validation and interpretation. The README and home page
summarize these routes but do not duplicate the tutorials. API pages own exact signatures and
result contracts.

Implement the transition in three reviewable patches:

1. establish this decision and the guide-layer target;
2. rename the first example, add the rendered quick-start tutorial and asset pipeline, update
   navigation, and protect its distribution and rendering contracts;
3. reframe the landing page, synthetic tutorial, path reference, catalogues, and active terminology
   around the three-stage route, then close the transition with stale-surface audits.

Patch 1 changed no executable example, renderer, navigation, or living user documentation. Patch 2
implements the renamed quick-start example, rendered tutorial, generated asset pipeline, and first
navigation position. Patch 3 remains responsible for reframing the broader landing page, path
reference, catalogues, and active terminology around the complete three-stage route.

This decision refines Decisions 0044, 0064, 0075, 0076, and 0104 where they assign the first or
shortest pedagogical route. Their broader self-contained-example, tutorial ownership, documentation
layering, and user-orientation contracts remain in force.

## Consequences

- The first rendered page shows the most attractive installed-package workflow and an immediate
  result.
- Users encounter automatic fitting before optional search inspection and validation machinery.
- The search object's lifetime becomes intuitive: temporary for automatic use, retained for
  evidence and OOF reporting.
- Fitted training-data diagnostics remain visibly distinct from predictive validation.
- The synthetic tutorial gains a precise pedagogical role rather than competing with the quick
  start for first-entry ownership.
- The complete Pulp tutorial remains the scientific endpoint of the onboarding route.
- No duplicate example, compatibility filename, redirect tutorial, or stale OOF terminology is
  retained.
