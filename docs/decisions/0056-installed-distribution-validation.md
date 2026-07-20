# Decision 0056: installed-distribution validation

## Status

Accepted and implemented.

## Context

The compatibility matrix validates the repository checkout under the supported interpreter and
runtime dependency boundaries. The documentation workflow also validates that the unpacked source
distribution contains enough material to build the public site. Neither check proves that the wheel
and source distribution users install provide the intended runtime package independently of the
repository checkout.

A packaging check must exercise the artifacts themselves without turning CI into a second broad
test matrix or duplicating separate wheel and source-distribution smoke-test implementations.

## Decision

Add `tools/check_distributions.py` and expose it as `make dist-check`. The helper:

- builds one wheel and one source distribution in a temporary artifact directory;
- creates a separate clean virtual environment for each artifact;
- installs each exact artifact path while running outside the repository checkout and with
  `PYTHONPATH` removed;
- runs the same smoke-test program in both environments;
- checks installed distribution metadata against `pipls.__version__`;
- imports the supported top-level estimators and the public datasets, inspection, metrics, and
  plotting modules without invoking optional plotting functions;
- fits and predicts with a representative fixed `PiPLSRegression` model; and
- verifies that `pipls` resolves from the clean virtual environment rather than repository `src/`.

The build workflow runs `make dist-check`. Because the helper performs the standard build itself,
the workflow does not run a redundant preceding `make build` command. The helper is included in the
source distribution so the public Make target remains valid from an unpacked release artifact.

## Consequences

CI now checks the installed runtime behavior of both standard Python distribution formats. Wheel
and source-distribution failures use the same public smoke test, so their results are directly
comparable and do not depend on private package internals.

This is an installation and packaging check, not a replacement for the ordinary test suite,
compatibility matrix, documentation-distribution build, or future publication rehearsal. It does
not upload artifacts, test TestPyPI, or define the first release version.
