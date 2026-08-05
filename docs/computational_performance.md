# Computational performance

Pi-PLS training cost is determined by the validation protocol, the number of evaluated component
and predictor-rank candidates, the matrix dimensions and retained ranks, the predictor SVD policy,
and the available execution resources. These controls do not all have the same meaning: some alter
the statistical evidence or candidate set, one introduces a numerical approximation, and others
primarily affect wall time or repeated work.

This guide organizes those choices for path training, fixed refitting, and selection-conditioned
OOF diagnostics. It assumes that learned preprocessing remains inside every estimator fit and every
training fold.

## A cost model for path training

Path evaluation repeats each fixed candidate over the materialized validation splits. Candidate-fit
counts provide a useful first accounting unit, but they are not a complete wall-clock model because
matrix sizes, retained ranks, preprocessing, memory traffic, solver choice, and parallel scheduling
also matter.

## Develop with a smaller validation protocol

The number of folds and repetitions multiplies the candidate-fit count. Development and final
analysis may therefore use different validation effort, provided the resulting difference in
selection evidence and partition-sensitivity information is stated explicitly.

## Choose predictor-rank coverage deliberately

`search_method="adaptive"` and `search_method="exhaustive"` control which admissible predictor ranks
are evaluated. Adaptive coverage is the routine default; exhaustive coverage is a deliberate request
for complete candidate evaluation.

## Use fixed or restricted predictor-rank policies when justified

Maximum rank, one fixed rank, or a declared rank subset can reduce the predictor-rank dimension.
These are modeling or candidate-set decisions, not equivalent accelerations of the unrestricted
search.

## Restrict the component path when justified

A declared `n_components_values` sequence can reduce the number of evaluated component counts when
scientific knowledge supports that restriction. It changes the path being evaluated.

## Use randomized predictor SVD for large problems

`svd_solver="randomized"` can be useful when predictor matrices are large and retained rank is well
below the matrix dimensions. It is an approximate predictor-decomposition route and is not
necessarily faster near full rank.

## Use parallelism deliberately

`n_jobs` can reduce wall time by evaluating independent work concurrently, but additional workers
can increase memory pressure and scheduling overhead. Worker choice should be measured on the
actual matrices and machine.

## Avoid repeated OOF computation

`oof_report()` refits the selected pair on every stored validation split. Compute the report only
when its diagnostics are needed, retain it, and reuse it across tables and figures.

## Inspect the work performed

The fitted search records the evaluated candidate rows, split count, achieved exhaustiveness, and
per-candidate timing summaries. These quantities help explain the work performed but should not be
confused with total parallel wall time.

## Separate development and final-analysis workflows

A development workflow can use less validation effort and a narrower justified search while code,
data handling, and reporting are being checked. A final workflow should restore the validation,
candidate, and numerical policies required by the scientific analysis.

## Summary of trade-offs

Performance controls should be classified by what they change: validation evidence, candidate
coverage or model policy, numerical approximation, parallel execution, or repeated diagnostic work.
That classification is more reliable than treating every faster configuration as equivalent.
