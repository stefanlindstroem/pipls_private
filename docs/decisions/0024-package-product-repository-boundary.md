# Decision 0024: package-product repository boundary

## Status

Accepted after completion of the initial Phase E3 reference-dataset suite.

## Context

The repository began with a roadmap that combined an installable Pi-PLS package with future
manuscript-reproduction infrastructure. Pi-PLS is expected to evolve beyond one publication,
including possible future standardization pipelines and block-scaling functionality. Keeping
paper-specific figures, complete comparison grids, and manuscript orchestration in the package
repository would blur the public software contract and burden programming users with assets that
are not part of ordinary package use.

The package must remain useful to several future publications without being organized around any
one of them.

## Decision

- Treat `pipls` as the long-lived repository for the installable package, public API, user
  documentation, concise examples, transparent datasets, lightweight validation benchmarks, tests,
  packaging, and releases.
- Keep manuscript figures, complete publication experiments, paper-only OLS/CCA comparisons,
  cached paper results, and publication-specific environments in separate downstream reproduction
  repositories.
- Let downstream reproduction repositories depend on tagged `pipls` releases.
- Permit ordinary PLS in lightweight package benchmarks because it is the nearest user-relevant
  baseline.
- Include OLS or CCA only when they test a package-level mathematical identity, limiting case, or
  public behavior; do not add them merely to reproduce a paper comparison.
- Remove paper-oriented placeholders and public navigation from `pipls` in a subsequent cleanup
  patch.
- Follow that cleanup with a synthetic benchmark-contract patch.
- Preserve future preprocessing, standardization, and block scaling as valid product directions,
  but explicitly defer all API design, naming, scheduling, and implementation until the project
  owner starts a dedicated future phase.

## Consequences

- The `pipls` roadmap becomes product-oriented and can support multiple future papers.
- Publication reproduction remains possible without becoming part of the package's supported
  surface.
- Lightweight benchmarks focus on software validation and user-relevant comparisons rather than
  publication claims.
- No preprocessing or block-scaling API commitment is created by this decision.
- Existing paper-oriented directories and public prose are transitional repository state and should
  be removed or rewritten in the next patch, without changing runtime behavior.
