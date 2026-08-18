# Decision: 0007-predictor-rank-search-policies

Status: accepted and implemented in `PiPLSSearchCV`; Decision 0039 removes these search modes from
`PiPLSRegression`. Decision 0154 keeps both coverage algorithms but makes exhaustive coverage the
accepted default and removes the separate `predictor_rank_values="max"` policy; implementation is
pending.

## Context

For fixed `n_components`, exhaustive evaluation of every admissible predictor rank can be
prohibitively expensive when the fold-safe upper bound is large. The public API therefore needs
both an exhaustive reference mode and a scalable default mode without conflating candidate-coverage
approximation with numerical SVD approximation.

The CV objective is discrete and may be noisy or non-unimodal. Strict bisection, ternary search,
or golden-section search would assume more structure than the objective guarantees and could
irreversibly discard the region containing the global minimum.

## Decision

`PiPLSSearchCV.search_method` exposes two candidate-coverage policies:

- `"exhaustive"`: evaluate every admissible integer predictor rank;
- `"adaptive"`: use deterministic logarithmic coarse-to-fine coverage and accept that the
  evaluated candidate set may be a subset of the exhaustive set.

`predictor_rank_values="max"` and one-element explicit sequences are separate fixed policies with
one rank candidate per component count. They do not require a candidate-coverage choice. Decision
0148 refines final retained-rank selection: after either optimized coverage policy finishes candidate
evaluation, separate public relative and absolute tolerances retain the smallest qualifying
evaluated rank. The public values are `"adaptive"` and `"exhaustive"`; no aliases are retained
for earlier pre-release spellings.

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

`"adaptive"` does not guarantee the exhaustive optimum for an arbitrary non-unimodal CV curve. This
limitation must be explicit in API documentation. Tests should show deterministic behavior,
correct caching, exhaustive equivalence on small intervals, and substantial candidate reduction
on large intervals. Comparative benchmarks should quantify how often adaptive and exhaustive
selection agree on representative synthetic and publication datasets.

## Separate numerical policy

Randomized SVD is not implicit in `search_method="adaptive"`. `search_method` controls predictor-
rank candidate coverage, while `svd_solver` independently controls the predictor decomposition
through `"full"`, `"randomized"`, or `"auto"`. This separation lets users distinguish incomplete
candidate coverage from approximate linear algebra.

## Consequences for path analysis

`PiPLSSearchCV` must reuse the same search-policy vocabulary. Its exhaustive policy searches the
complete admissible triangular $(h,r_\pi)$ grid; its adaptive policy may reduce that grid while
reporting every evaluated pair and retaining identical fold-local preprocessing and scoring
contracts.
