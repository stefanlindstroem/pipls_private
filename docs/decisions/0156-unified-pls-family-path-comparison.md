# Decision 0156: unify the maintained PLS-family path comparison

## Status

Accepted, implemented, and closed.

## Context

Decision 0155 introduced `response_subspace="least_squares"` as a programming-user software
extension while retaining `"cross_covariance"` as the peer-reviewed default. Its initial user-facing
comparison was a separate Pulp-only Example 07. The repository already had Example 03 for matched
Pi-PLS-versus-ordinary-PLS component paths on all three package-owned reference datasets.

Maintaining two overlapping comparison workflows split the PLS-family evidence unnecessarily. It
also made the least-squares response policy look like a special Pulp-only side analysis rather than
one alternative fixed-estimator construction that can be compared under the same validation
protocol as the publication-default Pi-PLS model and ordinary PLS.

## Decision

Example 03 is the sole maintained PLS-family component-path comparison. For each of Pulp,
Sugarcane, and Tobacco it compares three alternatives on exactly the same materialized shuffled
five-fold partitions:

1. Pi-PLS with `response_subspace="cross_covariance"`, the peer-reviewed package default;
2. Pi-PLS with `response_subspace="least_squares"`, the software extension outside the companion
   publication;
3. ordinary scikit-learn PLS evaluated with the existing response-standardized CV-MSE helper.

The two Pi-PLS alternatives remain separate `PiPLSSearchCV` objects. `response_subspace` remains
fixed estimator configuration and is not added as a third search dimension. Apart from the response
policy, both Pi-PLS searches use the same dataset-specific search configuration and the same
materialized CV split object.

The ordinary-PLS example helper accepts either its existing `KFold` input or a reusable materialized
split sequence so that all three paths can use the exact same partitions rather than merely
reconstructing equivalent splitters.

Example 03 does not select or refit a final model. Its CV-MSE paths are model-development evidence,
not independent post-selection validation. The least-squares policy remains explicitly labeled as
a software extension and no general predictive-superiority claim is made.

## Presentation and execution

The three existing Example-03 PDFs are retained:

- `examples/results/pls_path_comparison/pulp_component_path_comparison.pdf`;
- `examples/results/pls_path_comparison/sugarcane_component_path_comparison.pdf`;
- `examples/results/pls_path_comparison/tobacco_component_path_comparison.pdf`.

Each figure contains the two Pi-PLS response-policy paths and the ordinary-PLS path. Predictor-rank
annotations are omitted from the figure because two separately optimized rank sequences would
obscure the comparison; both Pi-PLS rank sequences are printed instead.

The script accepts `--dataset {pulp,sugarcane,tobacco}` for bounded execution while its default
continues to run all three datasets. Complete example qualification runs all three. Source-
distribution qualification executes only the Pulp branch so that artifact validation covers both
response policies without duplicating the high-dimensional example cost.

The former `examples/07_response_subspace_comparison.py`, its dedicated test, and
`response_subspace_comparison.pdf` are retired. The maintained numbered catalogue therefore returns
to Examples 01 through 06.

## Relationship to earlier decisions

- Decision 0045 continues to own the boundary between PLS-family comparison and fitted Pi-PLS model
  interpretation. The unified Example 03 remains comparison-only.
- Decision 0155 continues to own the two response-subspace policies and their publication boundary.
  Decision 0156 changes only where their maintained comparative evidence is presented.
- Decision 0152 remains historical owner of the selection-review workflow. Its original six-example
  catalogue was later extended by Decision 0155 and is now consolidated back to six examples here.

## Consequences

- Users see one PLS-family comparison rather than two overlapping workflows.
- The least-squares software extension is compared on all three reference datasets rather than only
  Pulp.
- All three plotted methods use identical materialized validation partitions within each dataset.
- The package search surface is unchanged.
- `make examples` becomes more computationally expensive because Sugarcane and Tobacco now each run
  a second Pi-PLS search; artifact qualification remains bounded to the Pulp comparison branch.
