# Decision: 0007-predictor-rank-search-policies

Status: accepted and implemented in `PiPLSSearchCV`; Decision 0039 removes these search modes from
`PiPLSRegression`.

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
- `"optimal"`: exhaustively evaluate every admissible integer rank;
- `"auto"`: use deterministic adaptive logarithmic coarse-to-fine search and accept that the
  evaluated candidate set is approximate relative to exhaustive search.

Decision 0148 refines final retained-rank selection: after either policy finishes candidate
evaluation, separate public relative and absolute tolerances retain the smallest qualifying
evaluated rank.

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
5. identify the exact evaluated reference rank using the configured scorer and private low-rank
   numerical tie rule;
6. refine the integer interval bounded by the neighboring evaluated ranks around that exact
   reference rank;
7. switch to exhaustive evaluation when the remaining interval contains no more than a small
   fixed implementation threshold, initially targeted at 10 ranks;
8. after evaluation, apply Decision 0148 and retain the smallest evaluated rank satisfying both
   public tolerance caps;
9. expose diagnostics sufficient to reconstruct both candidate coverage and final retention.

The implementation uses private deterministic constants of seven logarithmic points and an
exhaustive-switch threshold of 10 ranks. These are not public constructor parameters and should
remain private until benchmark evidence demonstrates a stable need.

## Required diagnostics

At minimum, adaptive fitting should record:

- the evaluated predictor ranks in deterministic order;
- mean and per-split scores for evaluated ranks;
- the number of admissible and evaluated candidates;
- the final refinement interval;
- whether every admissible rank was evaluated;
- the exact reference rank, tolerance-qualified retained rank, and standard score diagnostics.

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

`PiPLSSearchCV` must reuse the same search-policy vocabulary. Its exhaustive policy searches the
complete admissible triangular $(h,r_\pi)$ grid; its adaptive policy may reduce that grid while
reporting every evaluated pair and retaining identical fold-local preprocessing and scoring
contracts.
