# Decision 0150: computational-performance guidance

## Status

Accepted. Patch 4 of the authorized six-patch sequence is complete. The central served guide,
its navigation position, scope, section structure, and maintainer contracts are established.
Patch 5 writes the complete practical guidance. Patch 6 integrates links, troubleshooting,
documentation tests, distribution checks, and final current-state wording.

## Context

Training cost is currently explained only in isolated places. The Pulp tutorial notes that repeated
validation increases runtime, while the path and regression API pages separately document
`search_method`, `predictor_rank_values`, `n_components_values`, `n_jobs`, and `svd_solver`. A
programming user must infer how these controls interact and which changes alter statistical
evidence, candidate coverage, numerical approximation, or only execution.

The predictor-rank search terminology is now final under Decision 0149. The package therefore needs
one central guide written against `search_method="adaptive"` and
`search_method="exhaustive"`, rather than more scattered performance remarks.

## Decision

### Add one central served reference page

The canonical guide is:

```text
docs/computational_performance.md
```

It appears under **Reference**, immediately after **Path-selection details** and before
**Troubleshooting**. Tutorials and API pages may link to it, but they must not duplicate its full
explanation.

The guide covers computational performance during:

- `PiPLSSearchCV.fit()` path evaluation;
- explicit full-data `refit()`;
- selection-conditioned `oof_report()`;
- fixed `PiPLSRegression` fitting when SVD policy is relevant.

It does not define benchmark results, promise machine-independent speedups, or add runtime API.

### Distinguish four kinds of performance control

The guide must keep these categories separate.

1. **Validation evidence:** fewer folds or repetitions reduce fold fits but also change the evidence
   available for selection and split-to-split stability.
2. **Candidate policy:** adaptive rather than exhaustive coverage, fixed or maximum predictor rank,
   restricted rank sets, and restricted component paths reduce evaluated candidates but change the
   evaluated model-selection problem.
3. **Numerical approximation:** randomized predictor SVD may reduce suitable large-matrix costs but
   uses an approximate decomposition and is not guaranteed to be faster near full rank.
4. **Execution and reuse:** `n_jobs` may reduce wall time without intended statistical change, while
   retaining and reusing an OOF report avoids repeated selected-pair fits.

The guide must not present any control from the first three categories as a computationally neutral
acceleration.

### Preserve fold-local preprocessing and reproducibility

No performance advice may move learned centering, scaling, or other estimator preprocessing outside
cross-validation. Every candidate remains fitted independently on each training fold.

Examples using shuffled CV or randomized predictor SVD must use explicit integer seeds. The guide
must explain that reproducibility and computational cost are separate concerns.

### Describe current implementation behavior, not a generic search abstraction

The guide must accurately state that:

- path search evaluates fixed `(n_components, predictor_rank)` candidates over the materialized
  validation splits;
- `search_method="adaptive"` may omit admissible predictor ranks;
- `search_method="exhaustive"` evaluates every admissible predictor rank;
- maximum and one-element fixed-rank policies provide one rank candidate per component count and
  should omit `search_method` in examples;
- path search parallelizes candidate pairs within an evaluation batch, while folds for one
  candidate are evaluated serially;
- `oof_report()` performs a new selected-pair fit on every stored split and parallelizes those split
  fits;
- timing arrays in `cv_results_` are per-candidate split summaries, not total parallel wall time.

### Use a stable guide structure

The guide uses these sections:

1. A cost model for path training.
2. Develop with a smaller validation protocol.
3. Choose predictor-rank coverage deliberately.
4. Use fixed or restricted predictor-rank policies when justified.
5. Restrict the component path when justified.
6. Use randomized predictor SVD for large problems.
7. Use parallelism deliberately.
8. Avoid repeated OOF computation.
9. Inspect the work performed.
10. Separate development and final-analysis workflows.
11. Summary of trade-offs.

Patch 5 fills these sections with the complete formulas, examples, caveats, and decision table.
Patch 6 adds concise routes from existing documentation and a troubleshooting entry.

### Keep advice task-oriented and evidence-based

The guide should lead with the largest common multiplicative cost: the number of materialized
validation splits. It should then discuss candidate coverage and rank policy, component-path scope,
SVD policy, parallel execution, and OOF reuse.

Published examples should show current parameter names and executable constructor combinations.
Fixed and maximum-rank examples must omit `search_method`; randomized-SVD and shuffled-CV examples
must be seeded. The text should recommend measurement on the user's matrices and machine rather
than universal worker-count or speed claims.

## Authorized remaining sequence

5. Write the complete computational-performance guide using only the final predictor-rank search
   terminology.
6. Add concise cross-links, troubleshooting, changelog, documentation and distribution tests, then
   mark this decision implemented and return `.llm` to current-state wording.

## Validation obligations

The completed documentation increment must verify that:

- `computational_performance.md` is served in the required navigation position;
- its stable section structure is present;
- every shown public constructor parameter exists and every shown value is accepted;
- current guides use only `"adaptive"` and `"exhaustive"` for predictor-rank search;
- maximum and one-element fixed-rank examples omit `search_method`;
- shuffled CV and randomized-SVD examples use integer seeds;
- learned preprocessing remains fold-local;
- candidate-fit counts are distinguished from parallel wall time;
- maximum-rank, restricted-rank, restricted-component, and reduced-repetition advice is described
  as changing the evaluated evidence or policy;
- randomized SVD is described as an approximate numerical route;
- local links and anchors resolve;
- wheel and source distributions contain the new page.

## Consequences

Programming users gain one canonical route for understanding and controlling training cost. The
page can explain practical reductions without obscuring their statistical or numerical
consequences. Existing APIs and numerical behavior remain unchanged.
