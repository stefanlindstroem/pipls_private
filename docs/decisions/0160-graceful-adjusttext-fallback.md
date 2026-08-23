# Decision 0160: graceful optional `adjustText` fallback

## Status

Accepted, implemented, and closed. The allocator choice is superseded by Decision 0161; the
graceful-fallback and runtime-dependency boundary remains in force.

## Context

Pi-PLS owns numerical results rather than plotting. Decision 0083 therefore kept Matplotlib and
`adjustText` outside the runtime package and used `adjustText` only for heuristic label placement in
annotated Pulp biplots. The dependency was listed in the `examples`, `docs`, and `dev` extras, but
the maintained Pulp example and tutorial renderer imported it unconditionally. A user with
Matplotlib but without `adjustText` could therefore fit Pi-PLS normally yet could not execute those
otherwise valid rendering workflows.

## Decision

Keep `adjustText` in the `examples`, `docs`, and `dev` extras so maintained documentation and example
environments continue to receive the preferred collision-reducing label layout. Do not make it a
runtime dependency and do not remove it from those extras.

Example 04 and the Pulp tutorial renderer must treat absence of the `adjustText` package as a
non-fatal rendering condition. They create the ordinary Matplotlib text artists first and call
`adjust_text()` only when the optional import succeeds. If the package itself is unavailable, the
labels remain at their original predictor endpoints and all numerical results, arrows, scores,
loadings, diagnostics, and output workflows remain unchanged.

Catch only absence of the `adjustText` package itself. A `ModuleNotFoundError` raised for another
module while importing an installed `adjustText` must propagate rather than being hidden as an
optional-dependency fallback.

## Consequences

- The runtime package continues to import and operate without Matplotlib or `adjustText`.
- Users with Matplotlib alone can run the complete Pulp example; only automatic label repositioning
  is lost when `adjustText` is absent.
- The Pulp documentation renderer has the same fallback, so direct renderer execution no longer
  requires a stub module when `adjustText` is unavailable.
- Installing `pipls[examples]`, `pipls[docs]`, or `pipls[dev]` still installs `adjustText`, preserving
  the preferred maintained figure layout.
- Regression coverage blocks `adjustText` during import of both Pulp rendering workflows and verifies
  that the fallback is selected without affecting the package-owned numerical boundary.


## Supersession

Decision 0161 replaces `adjustText` with optional `textalloc` for maintained Pulp label allocation.
This record remains authoritative for the earlier decision that absence of an annotation-layout
extra must not prevent the Pulp example or tutorial renderer from running.
