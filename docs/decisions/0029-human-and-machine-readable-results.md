# Decision 0029: Human- and machine-readable benchmark results
> Status: superseded in its universal manifest/runner/schema consequences by Decision 0030.

Historical status: accepted and implemented before Decision 0030.

## Context

The first synthetic CI runner wrote nested JSON Lines records. That representation was appendable
and schema-valid, but the results are fundamentally one table. Routine inspection required custom
normalization code, nested field names, and knowledge of the record structure. This was unnecessary
friction for programming users and also made common tools such as spreadsheet software less useful.

The package is read and used by humans and machines. Neither audience should be treated as an
afterthought.

## Decision

- Use UTF-8, comma-delimited CSV as the primary and only generated result format for the synthetic
  package benchmark.
- Write one flat row per suite, tier, scenario, seed, and method.
- Flatten software versions, resolved method/CV parameters, metrics, status, and message into named
  columns.
- Use an empty cell for a value that is unavailable or does not apply to a method.
- Define exact column order, types, null representation, and result-schema version in a
  machine-readable JSON schema.
- Increment the result schema to version 2 because the output contract changes incompatibly. The
  synthetic scenario suite itself remains version 1.
- Keep generated results ignored by Git and remove a legacy `synthetic-ci.jsonl` file when
  `make benchmark-ci` runs.
- Prefer similarly direct formats elsewhere in the repository: when data are naturally tabular,
  nested serialization requires a demonstrated need.

## Consequences

`make benchmark-ci` writes `benchmarks/results/synthetic-ci.csv`. It can be opened directly with
pandas, R, spreadsheet software, or a text editor, while automated consumers can validate and parse
it using `benchmarks/schema/result-v2.schema.json`. The benchmark runner remains repository-local
and the scientific metric and tolerance policies are unchanged.