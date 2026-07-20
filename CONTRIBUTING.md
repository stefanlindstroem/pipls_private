# Contributing

Use a clean Git worktree and make one reviewable change at a time.

The repository is the long-lived home of the installable `pipls` package. Contributions should
serve package users through the public API, documentation, concise examples, transparent datasets,
lightweight validation benchmarks, tests, packaging, or release maintenance. Paper-specific figure
pipelines, complete publication grids, manuscript tables, and paper-only comparator workflows
belong in downstream reproduction repositories.

Run `make help` to see the maintained repository commands. Before submitting a patch, run:

```bash
make check
```

Run `make examples` when changing executable example behavior, example artifacts, or the
application-facing workflow. This target intentionally runs every numbered example, including the
slower Tobacco analysis. Run `make docs` after installing `.[docs]` when changing public
documentation, navigation, or docstrings. Use `make docs-serve` for a live local preview; it
serves `http://127.0.0.1:8000/` until stopped with `Ctrl+C`. Run `make docs-dist` when changing
documentation packaging, documentation dependencies, `MANIFEST.in`, or the source-distribution
boundary. Run
`make build` when changing packaging, dependencies, included data, or public modules.

Mathematical changes must update the relevant contracts in `.llm/` and include focused tests.
Benchmark changes must follow `.llm/benchmarking.md`: one question, one readable script, and one
minimal CSV output. Do not introduce a universal manifest or result schema, and do not commit
generated result files without an explicit fixture decision. Generated files, datasets without
verified redistribution terms, and archive clutter must not be committed.
