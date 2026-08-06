# Decision 0152: selection-review feedback workflow

## Status

Accepted and implemented. The manual-selection tutorials now inspect an unselected component path
before creating a selection, present the selected path and conditional evidence as a review stage,
and show one explicit return from that review to the selection decision when the evidence is
unsatisfactory. Selection-conditioned OOF reporting from the same search is described as inspection,
not independent qualification or validation.

## Context

Decision 0151 established one immutable selection as the handoff to OOF reporting and final
refitting. Its first tutorial diagrams compressed all search evidence into one inspection step and
used the phrase "qualify the selection" for optional OOF reporting.

That presentation hides two distinct questions in manual component-count workflows:

1. Which component count should be selected from the unconditional component path?
2. After that choice, is the selected row satisfactory when viewed together with its conditional
   predictor-rank profile and, where relevant, its selection-conditioned OOF predictions?

The second question can cause the user to revise the component count and create another selection.
It is therefore a review loop. It is not independent post-selection validation when the OOF report
reuses the same search splits that informed the selection.

The static examples also place a reproducible value in `CHOSEN_N_COMPONENTS`. That assignment and
`search.select(n_components=...)` represent one scientific operation: choose the component count and
create the immutable selection. Retrieving arrays and drawing a figure are implementation details of
inspection rather than separate workflow stages.

## Decision

### Separate unconditional and selection-conditioned inspection

For a manual component-count workflow, the conceptual order is:

```text
fit search
-> inspect component path
-> choose component count and create selection
-> inspect selected path and conditional evidence
-> refit the accepted selection
-> inspect the fitted model or predict
```

The first inspection uses `search.component_path_` without a selected marker. It supplies the
evidence from which the component count is chosen.

The selection operation combines the recorded component-count value and the call to
`search.select(...)`. The diagram must not represent those as separate scientific decisions.

The second inspection may include:

- the component path with the selected row marked;
- the predictor-rank profile conditional on the selected component count;
- selection-conditioned OOF predictions and diagnostics when that task is part of the workflow.

If this evidence is unsatisfactory, one dashed feedback edge returns to the selection operation.
The diagram does not imply an unbounded automated optimization loop; it records the possible manual
revision of one analytical choice.

### Do not call same-search OOF reporting independent qualification

`search.oof_report(X, y, selection=selection)` refits the selected pair over the validation splits
materialized by the same fitted search. Its predictions are useful descriptive evidence for the
selected row, but they are not an independent post-selection estimate.

Active tutorials and workflow descriptions therefore use terms such as "inspect", "review", or
"selection-conditioned OOF evidence". They do not say that the same-search report qualifies or
validates the selection. Independent post-selection assessment requires an outer resampling design
or untouched external data.

This terminology change does not alter mathematical uses of words such as "qualifying rank" when a
candidate satisfies an explicit tolerance inequality.

### Show the component path before and after selection

Tutorials 2 and 3 expose two component-path figures:

1. `component_path`, without a selected marker, before the component count is chosen;
2. `selected_component_path`, with the selected row marked, during review of the conditional
   evidence.

The same immutable `PiPLSComponentPath` data may be reused for both figures. No search is repeated.
The additional figure is a presentation artifact only.

The synthetic tutorial then inspects the conditional predictor-rank profile, refits the accepted
selection, and evaluates untouched external-test predictions.

The Pulp tutorial reviews the selected path, conditional predictor-rank profile, and
selection-conditioned OOF diagnostics before final refitting. Numerical fitted-model inspection
continues only after `search.refit(..., selection=selection)`.

### Keep scripts reproducible while presenting the decision order honestly

In the checked tutorial snippets, `CHOSEN_N_COMPONENTS` is introduced in the same source stage as
`search.select(...)`, after the initial path has been retrieved. The tutorials explain that an
interactive user would inspect the initial path, set the value, and rerun from the selection stage.
The fixed value remains in the complete script so examples, documentation assets, and tests are
deterministic.

Renderers may create the initial path figure before final refitting because the required immutable
search result already exists. This is consistent with data-first rendering: numerical calculations
remain explicit and rendering remains caller-owned. It does not add package plotting APIs or hide
selection inside a rendering helper.

### Refine earlier workflow decisions

This decision refines the pedagogical ordering and terminology in Decisions 0139 and 0151. Their
remaining contracts are unchanged:

- search fitting creates evidence rather than a final model;
- one immutable selection is passed to every selection-conditioned operation;
- the same accepted selection is passed to final refitting;
- `model.selection_` remains fitted-model provenance;
- automatic and comparison-only routes remain distinct from analytical refit workflows.

## Implementation outcome

The synthetic and Pulp examples, tutorial renderers, and served tutorials now distinguish the
unselected and selected path views. Their Mermaid diagrams combine setting the component count and
creating the selection in one node, combine retrieval and plotting within inspection nodes, and
include a dashed return edge from selected-evidence review to selection.

The Pulp, Sugarcane, and Tobacco workflows calculate selection-conditioned OOF diagnostics before
final refitting. Fitted-model inspection remains after refitting. Public catalogues, maintainer
records, and the changelog use the same terminology. Decision 0153 subsequently removes the
dedicated protocol-specific validation example and renumbers the complete real-data examples to
04--06 without changing this review sequence.

## Validation obligations

The implemented workflow must verify that:

- manual tutorial examples retrieve the component path before defining their chosen component count;
- the chosen value and `search.select(...)` occur in one checked source stage;
- predictor-rank-profile and OOF inspection follow selection and precede final refitting;
- the same named selection is passed to OOF reporting and final refitting;
- fitted-model inspection follows final refitting;
- each affected Mermaid diagram contains one feedback edge from selected-evidence review to
  selection;
- the initial path artifact contains no selected marker and the selected path artifact does;
- generated manifests include both path artifacts;
- selected pairs, CV-MSE values, OOF predictions, fitted predictions, and model results are
  numerically unchanged;
- active workflow prose does not describe same-search OOF reporting as qualification or independent
  validation;
- no generated Mermaid asset is committed.

## Consequences

The tutorials now match the actual reasoning required by a manual component-count choice. Readers
first see the evidence needed to make the choice, then see evidence conditional on that choice, and
can revise it before final refitting. The change adds two presentation artifacts but no public API,
search computation, model fit, or numerical result.
