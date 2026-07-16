# Decision: 0007-predictor-rank-search-policies

Status: accepted; implementation pending Phase C2c.

## Context

For fixed `n_components`, the current estimator evaluates every admissible predictor rank under
the name `predictor_rank="auto"`. This is statistically clear but can be prohibitively expensive
when the fold-safe upper bound is large. The public API needs both an exhaustive reference mode
and a scalable default mode without conflating rank-search approximation with numerical SVD
approximation.

The CV objective is discrete and may be noisy or non-unimodal. Strict bisection, ternary search,
or golden-section search would assume more structure than the objective guarantees and could
irreversibly discard the region containing the global minimum.

## Decision

The target predictor-rank modes are:

- a positive integer: fit the supplied rank directly;
- `"max"`: fit the rule-derived upper rank without CV search;
- `"optimal"`: exhaustively evaluate every admissible integer rank and select the best CV score;
- `"auto"`: use deterministic adaptive logarithmic coarse-to-fine search and accept that the
  selected rank is approximate relative to exhaustive search.

Because the package is pre-alpha, the current exhaustive `"auto"` behavior will be renamed to
`"optimal"` without a deprecated compatibility alias.

## Adaptive-search contract

For the interval

\begin{equation}
\{h, h+1, \ldots, r_{\pi,\max}\},
\end{equation}

adaptive search must:

1. materialize one CV split set and reuse it for every candidate;
2. fit all learned preprocessing inside each training fold;
3. begin with a deterministic logarithmically spaced set that includes both endpoints;
4. cache every evaluated rank and never refit a cached candidate unnecessarily;
5. identify the best evaluated rank using the same scorer orientation and low-rank tie rule as
   exhaustive search;
6. refine the integer interval bounded by the neighboring evaluated ranks around the current best
   rank;
7. switch to exhaustive evaluation when the remaining interval contains no more than a small
   fixed implementation threshold, initially targeted at 10 ranks;
8. return the best rank among all evaluated candidates;
9. expose diagnostics sufficient to reconstruct the search path.

The first implementation may use private deterministic constants for the number of logarithmic
points and the exhaustive-switch threshold. These should not become public constructor parameters
until benchmark evidence demonstrates a stable need.

## Required diagnostics

At minimum, adaptive fitting should record:

- the evaluated predictor ranks in deterministic order;
- mean and per-split scores for evaluated ranks;
- the number of admissible and evaluated candidates;
- the final refinement interval;
- whether every admissible rank was evaluated;
- the selected rank and standard best-score diagnostics.

## Limitations

`"auto"` does not guarantee the exhaustive optimum for an arbitrary non-unimodal CV curve. This
limitation must be explicit in API documentation. Tests should show deterministic behavior,
correct caching, exhaustive equivalence on small intervals, and substantial candidate reduction
on large intervals. Comparative benchmarks should quantify how often adaptive and exhaustive
selection agree on representative synthetic and publication datasets.

## Separate numerical policy

Randomized SVD is not implicit in `predictor_rank="auto"`. A later increment will add an explicit
`svd_solver` policy such as `"full"`, `"randomized"`, and `"auto"`, together with `random_state`
and fitted solver diagnostics. This separation allows users to distinguish approximate search
coverage from approximate linear algebra.

## Consequences for path analysis

`PiPLSPathCV` must reuse the same search-policy vocabulary. Its exhaustive policy searches the
complete admissible triangular $(h,r_\pi)$ grid; its adaptive policy may reduce that grid while
reporting every evaluated pair and retaining identical fold-local preprocessing and scoring
contracts.
