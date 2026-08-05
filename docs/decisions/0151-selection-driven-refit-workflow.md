# Decision 0151: selection-driven refit workflow

## Status

Accepted. Patch 1 of 6 is implemented: this target contract and the active maintainer guidance are
synchronized. The runtime API, maintained examples, served tutorials, Mermaid configuration, and
final documentation audit remain for Patches 2--6.

## Context

`PiPLSSearchCV.select()` already returns the complete immutable path row needed for later work, and
`oof_report(selection=...)` already consumes such an existing compatible selection. In contrast,
`refit()` currently asks the caller to repeat a rule or component count. A detailed workflow can
therefore inspect one selection, validate it through OOF reporting, and then independently repeat
the request that determines the final model.

That repetition weakens provenance. It also encourages the maintained examples to fit the final
model before they inspect or report the search evidence supporting the fit. The compact automatic
route remains useful, but the evidence-retaining route should pass one selection object through
inspection, validation, and final full-data fitting.

The three served tutorials also describe distinct workflows whose operation order is easier to
understand graphically. The documentation stack should represent those workflows directly in
Markdown without adding generated diagram assets or a separate publication toolchain.

## Decision

### Let `refit()` consume an existing selection

Add the keyword-only argument:

```python
selection: PiPLSSelection | None = None
```

The full public selection surface becomes conceptually:

```python
search.refit(
    X,
    y,
    *,
    selection=None,
    rule=None,
    n_components=None,
    relative_tolerance=None,
    absolute_tolerance=np.inf,
)
```

Exactly one of `selection`, `rule`, and `n_components` must determine the fitted row.

When `selection` is supplied:

- it must be a `PiPLSSelection` exactly compatible with the fitted search;
- compatibility uses the same definition as `oof_report(selection=...)`, including rule,
  component-count, predictor-rank, score, CV-MSE, split-count, and tolerance provenance;
- nondefault component-count tolerance arguments are invalid because the selection has already
  resolved those controls;
- validation occurs before cloning or full-data fitting;
- the returned estimator receives the exact supplied immutable object as `model.selection_` after
  fitting succeeds.

The method does not add search-owned fitted-model state, retain supplied matrices, or weaken direct
`PiPLSRegression` ownership of exact manually specified `(n_components, predictor_rank)` pairs.
Rule-based and component-count refitting remain supported for compact workflows.

The implementation must use one shared private selection-compatibility validator for `refit()` and
`oof_report()`. It must not maintain two definitions of compatible provenance.

### Make one selection the workflow handoff

The canonical evidence-retaining route is:

```python
search = PiPLSSearchCV(cv=cv).fit(X, y)

path = search.component_path_
selection = search.select(
    rule="minimum_cv_mse",
    relative_tolerance=relative_tolerance,
)
rank_profile = search.predictor_rank_profile(selection.n_components)
report = search.oof_report(X, y, selection=selection)

model = search.refit(X, y, selection=selection)
predictions = model.predict(X_new)
```

The conceptual order is:

```text
search
-> inspect search evidence
-> create one selection
-> qualify that selection when validation is part of the task
-> refit the same selection on all development data
-> inspect the fitted model
-> predict
-> report
```

OOF reporting remains optional and selection-conditioned. It is not outer validation. A workflow
whose purpose is only selection inspection, OOF reporting, or path comparison need not fit a final
model.

The compact automatic route remains deliberately shorter:

```python
model = PiPLSSearchCV().fit(X, y).refit(
    X,
    y,
    rule="minimum_cv_mse",
)
```

### Align maintained examples by task

The maintained routes are:

| Route | Examples | Required order |
|---|---|---|
| Automatic selection and final refit | 01 | search and rule-based refit, then fitted-value diagnostics |
| Inspect, select, qualify, and refit | 02 and 05--07 | search evidence, one selection, optional OOF report, selection-driven refit, fitted-model analysis |
| Selection-conditioned validation without a final model | 03 | search, selection, OOF report |
| Path comparison without a final model | 04 | evaluate and compare paths |

Examples 02 and 05--07 must not recover their working selection from `model.selection_`. They create
the selection before final fitting and pass that same named object to every selection-conditioned
operation. Numerical outputs and selected pairs are intended to remain unchanged.

### Add one source-level flowchart to each served tutorial

Use Mermaid flowcharts in the three served tutorials:

1. quick start with Pulp;
2. inspect and select with synthetic data;
3. complete Pulp analysis.

The diagrams describe the scientific workflow rather than every source statement. Each uses a
vertical layout suitable for narrow screens, stable operation labels shared across tutorials, and
approximately five to eight nodes. A nearby prose sentence states the same sequence so the page
remains understandable when client-side diagram rendering is unavailable.

Mermaid source remains in the tutorial Markdown. Do not commit generated diagram SVG or PNG files,
add TikZ or Graphviz build steps, or make a diagram the sole statement of a public contract. The
MkDocs configuration and Pages overlay must preserve strict local and source-distribution builds.

### Refine earlier lifecycle decisions

This decision refines the workflow-order and duplicated-selection portions of Decisions 0137,
0139, 0141, 0143, and 0145. Their durable boundaries remain:

- search fitting produces evidence rather than a final model;
- `select()`, `refit()`, and `oof_report()` remain explicit operations;
- refitted models retain exact immutable selection provenance;
- OOF reporting reuses the fitted search splits and remains selection-conditioned;
- the three-stage onboarding route begins with the compact automatic workflow.

The paused presentation increment in Decision 0139 is superseded by the integrated tutorial and
workflow work in this decision.

## Patch sequence

1. Establish this decision, refine the affected lifecycle records, and synchronize active
   maintainer contracts without changing runtime or served documentation -- complete.
2. Add `selection=` to `PiPLSSearchCV.refit()`, share exact compatibility validation with
   `oof_report()`, and add API, pipeline, failure, cloning, and pickle coverage.
3. Reorder the synthetic inspect-and-select example, renderer, tutorial, and tests around one
   pre-refit selection while preserving numerical artifacts.
4. Reorder the complete Pulp, Sugarcane, and Tobacco workflows and replace the global refit-first
   source contract with route-specific lifecycle tests.
5. Add Mermaid support, one flowchart per served tutorial, a safe Pages configuration overlay, and
   strict rendering and source-distribution checks.
6. Update remaining public routes, examples catalogue, changelog, maintainer records, and stale-
   surface audits; complete this decision.

## Validation obligations

The completed sequence must verify that:

- `selection`, `rule`, and `n_components` are mutually exclusive selection sources;
- supplied selections use the same exact compatibility contract in `refit()` and `oof_report()`;
- nondefault component-count tolerances are rejected with `selection`;
- the exact supplied selection object is attached only after successful full-data fitting;
- direct estimators and terminal pipelines behave identically at the public boundary;
- incompatible types and provenance fail before model fitting and do not mutate the search;
- rule-based and manual component-count refitting remain compatible;
- Examples 02 and 05--07 create one selection before OOF reporting and final refitting;
- Examples 03 and 04 retain their no-final-model roles;
- selected pairs, predictions, OOF results, and generated numerical artifacts remain unchanged;
- every served tutorial contains one vertical Mermaid flowchart and equivalent prose;
- local, Pages-overlay, and isolated source-distribution documentation builds render the diagrams
  strictly without committed generated diagram assets.

## Consequences

The automatic route stays concise, while analytical workflows gain an explicit immutable handoff
from evidence to validation and final fitting. The same selection cannot be silently re-resolved
with different arguments between OOF reporting and refitting. Tutorial order reflects the actual
scientific decisions, and compact source-level diagrams make the distinct routes easier to scan
without expanding the runtime package or numerical surface.
