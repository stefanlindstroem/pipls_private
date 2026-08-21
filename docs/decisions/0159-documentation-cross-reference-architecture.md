# Decision 0159: documentation cross-reference architecture

## Status

Accepted; implementation is in progress.

## Context

The served documentation has mature tutorial, guide, reference, dataset, theory, citation, and
reproducibility pages, but cross-references between those layers are uneven. Readers can encounter a
reference dataset, the companion publication, or a Pi-PLS mathematical concept without an immediate
route to the page that owns its detailed explanation.

Adding links mechanically to every repeated noun would create visual noise and would not preserve
semantic meaning. A reference to the Pulp dataset, a Pulp tutorial, and `load_pulp()` are different
navigation intents. Likewise, a reference to predictor rank should lead to rank interpretation,
while a reference to the least-squares response policy should lead to response-subspace theory.

The documentation therefore needs a small set of stable canonical destinations and a semantic rule
for choosing among them.

## Decision

### Cross-reference by semantic destination

Cross-references are added where they shorten the reader's route to maintained detail. The link
text and target must reflect the meaning of the reference in context:

- a prose reference to a package-owned reference dataset links to that dataset's detailed section;
- a reference to a tutorial links to the tutorial rather than to the dataset page;
- a loader or programming symbol links to its programming reference where that is the intended
  subject;
- a reference to the companion paper, manuscript, or peer-reviewed publication links to the
  canonical companion-paper section;
- a mathematical concept links to the most specific maintained theory section that owns that
  concept rather than generically to the top of the theory page.

The goal is semantic navigation, not maximum hyperlink density. A substantive introduction or
renewed reference in a new section should normally be linked. Immediate repetitions may remain
plain text when the destination is already clear. Headings, code, citations or quotations, figure
alt text, and prose already on the canonical destination page need not self-link.

### Canonical dataset destinations

The canonical real-dataset destinations are:

- Pulp: `datasets.md#pulp-real-data-integration`;
- Sugarcane: `datasets.md#sugarcane-spectral-integration`;
- Tobacco: `datasets.md#tobacco-spectral-integration`.

The dataset-name link answers the question "what is this dataset?" Other links may coexist when the
reader instead needs a tutorial, example, or loader API.

### Canonical publication destination

The canonical destination for references to the companion paper, companion manuscript, or
peer-reviewed publication is:

- `citation.md#companion-paper`.

This destination owns the publication identity and citation information. A page may additionally
link to a theory or reproduction section when the surrounding statement is about those subjects.

### Canonical theory destinations

Frequently referenced mathematical concepts use stable anchors in `theory.md`:

- problem setting and the two rank controls:
  `theory.md#problem-setting-and-two-rank-controls`;
- canonical Pi-PLS terminology: `theory.md#canonical-terminology`;
- rank-controlled predictor projection: `theory.md#rank-controlled-predictor-projection`;
- response-subspace selection: `theory.md#response-subspace-selection`;
- diagonal latent coupling: `theory.md#diagonal-latent-coupling`;
- panoramic interpretation: `theory.md#why-the-method-is-panoramic`;
- interpretation of $r_\pi$ and $h$: `theory.md#interpretation-of-the-ranks`;
- relationships to established methods: `theory.md#relationships-to-established-methods`, with
  stable method-specific anchors for ordinary least squares, reduced-rank regression, canonical
  correlation analysis, and PLS/PLS-SVD;
- selection, validation, and synthetic-data boundaries:
  `theory.md#selection-validation-and-synthetic-data-boundaries`.

Existing stable anchors for response-subspace selection, diagonal latent coupling, and rank
interpretation remain unchanged. This decision makes the other high-value destinations explicit so
heading wording can evolve without silently breaking incoming links.

## Patch sequence

### 0159A — contract and canonical anchors

Implemented in this patch. Record the semantic cross-reference policy and make the canonical
reference-dataset, companion-paper, and high-value theory anchors explicit. No broad prose-linking
pass is performed yet.

### 0159B — reference-dataset cross-references

Implemented. Substantive references to Pulp, Sugarcane, and Tobacco across the served documentation
now link to the corresponding dataset-detail sections when the dataset itself is the navigation
subject. The pass covers Home, the example catalogue and comparison guide, API overview and dataset
reference, path-selection guidance, reproducibility, theory context, the Quick Start, the complete
Pulp tutorial, and same-page navigation in the dataset guide.

Tutorial names, workflow references, loader symbols, headings, code, figure alt text, and immediate
repetitions are not mechanically redirected to the dataset guide when another destination is more
semantic or a nearby dataset link already supplies the route.

### 0159C — publication and theory cross-references

Link substantive companion-publication references to the canonical citation section and connect
mathematical terminology and method comparisons to the most specific maintained theory sections.
Avoid generic theory-page links when a stable concept-specific anchor is available.

### 0159D — final navigation audit and regression protection

Review the served documentation for remaining isolated pages and missing high-value contextual
links, add lightweight regression tests for the canonical cross-reference architecture, run strict
documentation and source-distribution validation, and close Decision 0159.

## Relationship to earlier decisions

- Decision 0065 continues to own tutorial, guide, and reference-layer separation.
- Decision 0117 continues to own authorship, license, and citation policy.
- Decision 0120 continues to own companion-manuscript theory alignment.
- Decision 0121 continues to own canonical Pi-PLS terminology.
- Decision 0142 continues to own the three package-owned reference datasets.
- Decision 0158 continues to own the Home-page Pulp/Tobacco parsimony presentation.
- This decision changes documentation navigation only. It introduces no numerical, dataset,
  estimator, search, inspection, or public-API behavior.

## Consequences

- Dataset names can lead directly to provenance, dimensions, preparation, and loader context.
- Publication references have one stable citation destination.
- Mathematical references can lead to concept-specific theory rather than requiring readers to
  search the theory page manually.
- Explicit anchors provide stable incoming-link contracts even if visible heading text is edited.
- Cross-referencing remains readable because repeated nearby mentions and destination-page
  self-links are not required.
