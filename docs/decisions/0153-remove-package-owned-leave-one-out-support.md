# Decision 0153: remove package-owned leave-one-out support

## Status

Accepted. Implementation is planned in five patches. Patches 1--3 are implemented. The public
leave-one-out provenance field and private detector are removed, singleton-validation scorer safety
is protocol-neutral, the dedicated example and test are deleted, and the maintained examples are
renumbered contiguously. Support documentation and historical decision consolidation remain for
Patches 4 and 5.

## Context

The package currently has two different kinds of validation functionality:

1. generic cross-validation interoperability through `PiPLSSearchCV(cv=...)` and explicit split
   iterables;
2. package-owned leave-one-out support consisting of a dedicated example, automatic recognition of
   leave-one-out split structure, the public `PiPLSOOFReport.is_leave_one_out` provenance field,
   leave-one-out-specific documentation, and tests that promise this protocol as a maintained
   workflow.

The first capability is part of ordinary scikit-learn estimator interoperability and remains useful.
The second adds protocol-specific code and documentation without changing the Pi-PLS model itself.
A user can already request singleton validation folds intentionally by supplying a compatible
splitter or explicit splits. A user can also write a direct validation loop around
`PiPLSRegression`. The package does not need a dedicated leave-one-out mode, detector, result flag,
example, or compatibility promise to permit those uses.

The removal must not be implemented by deleting generic OOF reporting. Ordered OOF predictions,
per-observation prediction counts, repeated-prediction averaging, partial coverage, and pooled OOF
$R^2$ are protocol-neutral search results. Similarly, validation scoring must still reject any split
whose validation size makes the configured scorer undefined. That safety rule applies to arbitrary
splitters, not only leave-one-out.

## Decision

### Remove the package-owned leave-one-out surface

The package will no longer provide, document, or test leave-one-out as a named supported workflow.
The removal includes:

- `PiPLSOOFReport.is_leave_one_out`;
- private recognition of whether stored splits form a leave-one-out partition;
- the dedicated numbered leave-one-out example and its executable test;
- public or maintainer documentation that teaches, recommends, or guarantees leave-one-out;
- leave-one-out-specific test cases whose sole purpose is to establish package support;
- active decision claims that leave-one-out is part of onboarding or OOF provenance.

This is an intentional pre-release removal. Do not add a deprecated property, compatibility alias,
warning period, replacement detector, or hidden protocol tag.

### Preserve generic splitter interoperability

`PiPLSSearchCV(cv=...)` continues to accept scikit-learn-compatible splitters and explicit split
iterables under the existing validation-metadata and split-materialization contracts. The package
will not special-case, identify, advertise, or guarantee `sklearn.model_selection.LeaveOneOut`.

A user may still intentionally supply singleton validation folds, including a scikit-learn splitter
that produces them, provided the configured scoring operation is defined for those folds. Such use
is ordinary splitter interoperability, not package-owned leave-one-out functionality. The package
also does not prevent a user from implementing a direct leave-one-out procedure around the fixed
estimator.

No new convenience parameter such as `leave_one_out=True`, no package splitter, and no dedicated
helper will replace the removed surface.

### Keep generic OOF reporting

`PiPLSSearchCV.oof_report(...)` remains selection-conditioned and protocol-neutral. Its durable
result surface after this decision is implemented is:

```python
report.selection
report.oof_predictions
report.oof_prediction_counts
report.pooled_oof_r2
report.has_complete_oof_coverage
```

The report continues to:

- reuse every split materialized by the fitted search;
- preserve observation order;
- average repeated validation predictions per observation;
- expose prediction counts and partial coverage;
- calculate pooled OOF $R^2$ over covered observations when mathematically defined;
- avoid candidate rescoring, search mutation, implicit full-data fitting, or retained input data.

Removing leave-one-out provenance must not alter candidate evaluation, selection, OOF arrays,
repeated-fold behavior, or ordinary prediction behavior.

### Retain protocol-neutral singleton-validation scoring safety

A scorer such as foldwise $R^2$ is undefined when a validation split contains fewer than two
observations. The package must retain an early, protocol-neutral check for this condition. The check
and its message must refer to singleton validation sets or insufficient validation observations,
not to leave-one-out.

Tests for that safety rule should use an explicit split iterable or a small custom splitter. They
must not import `LeaveOneOut` merely to exercise a generic invalid-fold condition.

### Remove and renumber the maintained example

Delete the dedicated leave-one-out example and its example-level test. Renumber the remaining
examples so the catalogue remains contiguous:

```text
04_pls_path_comparison.py -> 03_pls_path_comparison.py
05_pulp_real_data.py      -> 04_pulp_real_data.py
06_sugarcane_real_data.py -> 05_sugarcane_real_data.py
07_tobacco_real_data.py   -> 06_tobacco_real_data.py
```

Update tutorial snippet paths, renderers, source-distribution checks, workflow tests, catalogues,
and maintained decision references in the same increment. The renumbering is documentary and
organizational; numerical calculations and generated artifact names remain unchanged unless a path
contains the example filename itself.

### Consolidate active decisions after implementation

Decision 0143 remains the canonical generic OOF and selection-provenance decision, but its
leave-one-out detector and report-field claims must be removed. Decisions 0139, 0151, and 0152 must
use the renumbered example catalogue and must not retain a dedicated leave-one-out route.

The retirement map entries for the already retired leave-one-out decisions must point to this
decision for the removal boundary and to Decision 0143 only for generic OOF behavior. They must not
claim that the leave-one-out example remains in onboarding.

Historical Git content remains available through repository history. Do not create a new archived
copy of the removed example, detector, field, or decisions.

## Patch sequence

1. Establish this decision and synchronize the decision registry, current-state handoff, strategy,
   public-API contract, and testing contract without changing runtime behavior or served user
   documentation.
2. Remove `PiPLSOOFReport.is_leave_one_out` and private split recognition, keep generic OOF
   reporting and protocol-neutral singleton-fold scoring safety, and update focused API and unit
   tests.
3. Delete the dedicated example and test, renumber the remaining examples, and update all source,
   renderer, distribution, and structural references without changing their numerical behavior.
4. Remove leave-one-out-specific public and maintainer documentation, replace repeated support
   claims with one concise non-support boundary, and keep generic CV and OOF guidance.
5. Consolidate Decisions 0139, 0143, 0151, and 0152, correct the retirement map, complete stale-
   surface audits, and return maintainer state to a completed current-state description.

## Validation obligations

The completed sequence must verify that:

- no runtime source imports or names `LeaveOneOut`;
- no public or private runtime field detects leave-one-out structure;
- `PiPLSOOFReport` remains immutable, validated, pickle-safe, and protocol-neutral;
- ordered OOF predictions, repeated-prediction averaging, counts, partial coverage, and pooled OOF
  $R^2$ remain unchanged for maintained generic split protocols;
- singleton-validation scorer safety remains and uses protocol-neutral terminology;
- grouped, temporal, predefined, repeated, and ordinary $K$-fold splitters retain coverage;
- no dedicated leave-one-out example, test, tutorial section, or active support claim remains;
- the numbered examples and every path reference are contiguous after renumbering;
- no compatibility alias or deprecation scaffold replaces `is_leave_one_out`;
- retired leave-one-out decision filenames have accurate current replacements;
- model fitting, candidate scoring, selection, final refitting, and ordinary prediction numerics are
  unchanged.

## Consequences

The public OOF result becomes smaller and describes only information that is useful across validation
protocols. The example catalogue loses a protocol-specific workflow, and the package stops promising
behavior that users can already express through ordinary splitter interoperability.

Users remain free to supply singleton folds intentionally or to use compatible scikit-learn
functionality, but they own that protocol choice. Pi-PLS will not identify it, attach leave-one-out
provenance, provide a dedicated example, or treat it as a separately maintained package feature.
