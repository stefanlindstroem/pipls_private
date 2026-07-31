# Decision 0126: historical-removal test policy

## Status

Accepted and implemented.

## Context

Several pre-release cleanup increments added tests whose only purpose was to prove that a former
name, helper, dataset integration, documentation page, or repository directory remained absent.
Those checks were useful while each migration was being completed, but permanent tombstones make
the test suite retain every intermediate repository shape. The accepted decision records already
preserve why those objects were removed.

A negative assertion remains useful when absence is itself a current product boundary. Examples
include the lack of a runtime plotting API, optional rendering dependencies staying outside runtime
requirements, and repository datasets not becoming top-level package exports. Those checks describe
the present architecture rather than a past migration.

## Decision

Do not retain a test solely because it proves that a former internal symbol, helper, dataset,
documentation wrapper, or repository path is still absent. Historical removals are recorded in the
decision history and do not require one permanent test per deleted object.

Retain negative tests when the absence is an explicit current public or architectural contract and
the test verifies that boundary directly. Prefer positive tests of the supported public surface and
generic structural invariants over lists of retired names.

Remove the migration-specific tombstones for:

- former fixed-estimator aliases and search bookkeeping attributes;
- former result-table attributes and obsolete example helpers;
- the retired Linnerud integration and paper-reproduction placeholders;
- the duplicate API-documentation check for the removed plotting module.

The rendering-policy test remains the authoritative check that the runtime package exposes no
plotting API. Existing public-behavior, search-result, example-structure, repository-layout, and
decision-index tests remain unchanged.

This decision supersedes Decision 0071 only where it requires permanent absence tests for removed
result attributes and obsolete helper functions. It does not restore any removed object or alter
any public behavior.

## Consequences

- The test suite protects the current supported architecture rather than every pre-release shape.
- Historical records remain available without being duplicated as executable name lists.
- Reintroducing an unsupported public surface is still caught by current API, rendering, packaging,
  and structural tests where absence is a maintained contract.
- Implementation, numerical behavior, examples, documentation content, and distribution contents
  are unchanged.
