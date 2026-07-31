# Decision 0127: artifact-based rendering validation

## Status

Accepted and implemented.

## Context

Decision 0118 established a consistent visual policy for maintained numbered-example PDFs and
repository-generated tutorial SVGs. Its implementation added source-level tests for exact title
strings, label spellings, subplot-title absence, axis-limit expressions, tick-label expressions,
and categorical-label rotation. Those tests verified the migration, but they also made ordinary
rendering edits depend on the exact arrangement and wording of Matplotlib calls.

The repository already has stronger durable boundaries. The runtime package is tested without
rendering dependencies or a plotting API, numbered examples keep chart construction caller-owned,
tutorial renderers execute in isolated processes, generated SVG files are parsed, and manifests
record the declared assets. Numerical meaning is tested separately, including the use of
fold-based standard errors for maintained CV-MSE error bars.

## Decision

Retain the visual conventions of Decision 0118 as the maintained rendering policy, but do not
encode every convention as an exact source-string test.

Automated rendering tests protect:

- the absence of a runtime plotting API and runtime rendering dependencies;
- successful runtime imports without Matplotlib or `adjustText`;
- direct caller-owned chart construction in numbered examples and tutorial renderers;
- the absence of a hidden rendering layer under `examples/_support/`;
- direct rendering from named immutable result arrays;
- fold-based standard errors, rather than fold standard deviations, in maintained CV-MSE error
  bars;
- successful generation, declared filenames, manifest integrity, and parseability of generated
  tutorial assets.

Do not retain tests whose sole contract is an exact rendered phrase, title count, source expression,
axis-limit call, tick-label expression, label rotation syntax, or the absence of a particular
Matplotlib call inside a source-code range. Review regenerated artifacts through `make docs-figures`
and `make examples` when rendering behavior changes.

This decision supersedes Decision 0118 only where its consequences require source-level tests to
protect the complete renderer set. It does not reverse the visual policy, add a plotting API, alter
numerical results, or change the ownership of rendering.

## Consequences

- Rendering tests protect executable artifacts, architectural ownership, and numerical meaning
  rather than one implementation spelling.
- Maintainers can revise labels and layout without rewriting source-string tests when the visible
  result remains appropriate.
- Changes to rendered output still require regenerating and reviewing the affected SVG or PDF
  artifacts through the maintained Make targets.
- Tutorial renderer failures, missing declared assets, malformed SVG output, hidden plotting
  abstractions, runtime dependency regressions, and misuse of fold dispersion remain detectable.
