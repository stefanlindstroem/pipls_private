# Decision 0158: add a concise Home-page parsimony comparison

## Status

Accepted; implementation in progress.

## Context

Decision 0156 made Example 03 the maintained PLS-family comparison workflow and Decision 0157
added one deterministic synthetic stress case. Example 03 intentionally exposes the full technical
comparison: cross-covariance Pi-PLS, least-squares Pi-PLS, and ordinary PLS on matched materialized
validation folds.

The Home page has a different purpose. Its `Why use Π-PLS?` section should communicate the practical
motivation for the method without introducing the optional least-squares response-subspace extension
before the reader needs it. Pulp and Tobacco provide complementary real-data examples in which the
publication-default Pi-PLS path reaches its useful low-error region with fewer shared components
than ordinary PLS.

## Decision

The Home page will show two simplified component-path figures side by side under `Why use Π-PLS?`:

- Pulp;
- Tobacco.

Each Home figure will contain only:

- Pi-PLS with `response_subspace="cross_covariance"`, labelled `Π-PLS`;
- ordinary PLS, labelled `PLS`.

The least-squares response-subspace software extension remains available in Example 03 and the API
and theory documentation, but it is intentionally omitted from the Home-page motivation figure.
Figure labels will stay concise: the x-axis is `Nr of components`, and the y-axis remains
`Mean response-standardized CV-MSE (±1 SD)`.

The Home figures must use the same numerical protocol as Example 03 rather than reimplementing that
analysis independently. In particular, Pulp retains exhaustive predictor-rank coverage, Tobacco
retains adaptive coverage with `svd_solver="full"` and `n_jobs=1`, and all methods use the same
seeded five-fold materialized validation protocol.

The Home-page prose may describe parsimony only in the shared component count $h$. It must not call
the displayed Pi-PLS model unconditionally smaller or simpler, because Pi-PLS also selects a
predictor rank $r_\pi$. The full Example-03 comparison remains the technical destination for readers
who want both response-subspace policies and the complete matched-CV context.

## Patch sequence

### 0158A — shared numerical protocol

Implemented in this patch. The numerical evaluation used by Example 03 is factored into
`examples/_support/pls_family_path_comparison.py`. The helper owns materialization of the maintained
five-fold protocol, case-specific Pi-PLS search construction, requested response-subspace path
fitting, component-domain consistency checks, and ordinary-PLS path evaluation. Example 03 requests
both Pi-PLS response-subspace policies through that helper, preserving its existing plotting and
console presentation. The helper can also evaluate only the cross-covariance policy, which is the
contract needed by the future Home-page renderer.

### 0158B — simplified generated Home assets

Pending. Add a documentation renderer that evaluates only the publication-default Pi-PLS path and
ordinary PLS for Pulp and Tobacco and writes two concise generated SVG assets plus reproducibility
metadata.

### 0158C — Home-page placement and closure

Pending. Place the Pulp and Tobacco figures side by side under `Why use Π-PLS?`, state the bounded
shared-component parsimony interpretation, link to Example 03 for the complete comparison, update
documentation reproducibility records, and close this decision.

## Relationship to earlier decisions

- Decision 0045 continues to own the comparison-only role of ordinary PLS.
- Decision 0155 continues to own the response-subspace policies and publication boundary.
- Decision 0156 continues to own Example 03 as the complete maintained PLS-family comparison.
- Decision 0157 continues to own the separate deterministic synthetic stress case.
- This decision changes documentation presentation only; it introduces no public estimator or
  search API.

## Consequences

- Home gets a concise real-data motivation figure without duplicating the full Example-03 story.
- The numerical comparison protocol has one maintained implementation shared by Example 03 and the
  future Home renderer.
- The Home figures cannot silently drift to different folds or case-specific search settings.
- Documentation-build cost avoids the unnecessary least-squares searches for the Home assets.
