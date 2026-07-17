# Decision 0028: Synthetic CI benchmark runner

Status: accepted and implemented.

## Context

Decision 0027 defined a versioned synthetic benchmark contract but intentionally added no runner or
frozen results. The first executable layer needed to remain small enough for ordinary package CI,
consume the manifest directly, preserve fold-local model standardization, and avoid turning observed
benchmark values into scientific or performance claims.

The adaptive `PiPLSPathCV` method evaluates independent candidate models. Uncontrolled nested native
linear-algebra threads made that workload unsuitable for a small CI budget, even though the declared
problem sizes were modest.

## Decision

- Add a repository-local `benchmarks/run_synthetic.py` command rather than a public `pipls` API.
- Implement only the manifest's `ci` tier in this increment.
- Resolve constant and logarithmic scale specifications deterministically and generate each problem
  with `make_pipls_train_test`.
- Execute fixed Pi-PLS, adaptive `PiPLSPathCV`, and ordinary `PLSRegression` exactly as declared by
  the manifest.
- Parallelize independent path candidates with the joblib threading backend while limiting native
  BLAS/LAPACK work to one thread per candidate.
- Produce one schema-valid JSON Lines record per scenario, seed, and method.
- Record prediction, selection, subspace-capture, and resource metrics without committing generated
  result files by default.
- Test executable CLI behavior, numerical repeatability excluding resource timings, finite metrics,
  and subspace bounds.
- Do not freeze predictive rankings, exact selected ranks, timing limits, or broad result fixtures.

## Consequences

`make benchmark-ci` now creates the ignored file
`benchmarks/results/synthetic-ci.jsonl`. The runner is a package-maintenance asset, not an importable
user API and not a publication experiment. Standard and performance tiers remain unimplemented and
require separate review.
