# Decision 0097: focused leave-one-out example

## Context

Decision 0048 removed a context-free advanced-validation script that combined grouped,
leave-one-out, and temporal splitters on an unexplained random matrix. Leave-one-out validation
remained supported and documented, but users had no maintained example showing how singleton-safe
scoring, ordered OOF predictions, and pooled OOF $R^2$ fit together for a small multivariate
regression problem.

A leave-one-out example is useful only when it represents a recognizable sampling problem rather
than serving as a catalogue of splitter classes.

## Decision

1. Add `examples/03_leave_one_out_validation.py` as a focused synthetic small-calibration workflow
   with twelve observations and two responses.
2. Use one explicit compact triangular candidate set, `LeaveOneOut()`, the default
   response-standardized scorer, `return_oof_predictions=True`, and `refit=False`.
3. Report the selected rank pair, split count, complete OOF coverage, prediction shape, mean
   response-standardized CV-MSE, pooled OOF $R^2$, and the selection-conditioned estimate kind.
4. State explicitly that pooled OOF $R^2$ is not mean foldwise $R^2$; the latter is undefined for
   singleton validation folds.
5. Link the leave-one-out reference to the example and test the small script directly. Do not add
   figures, generated files, broad splitter comparisons, or publication-oriented analysis.

## Consequences

Leave-one-out validation now has one concrete, inexpensive, self-contained example. Grouped and
temporal validation remain reference-only until a distinct application-specific example is
justified. The earlier prohibition on context-free splitter catalogues remains intact.
