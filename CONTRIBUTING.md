# Contributing

Use a clean Git worktree and make one reviewable change at a time.

The repository is the long-lived home of the installable `pipls` package. Contributions should
serve package users through the public API, documentation, concise examples, transparent datasets,
lightweight validation benchmarks, tests, packaging, or release maintenance. Paper-specific figure
pipelines, complete publication grids, manuscript tables, and paper-only comparator workflows
belong in downstream reproduction repositories.

Before submitting a patch, run:

```bash
make check
```

Run `make build` when changing packaging, dependencies, included data, or public modules.

Mathematical changes must update the relevant contracts in `.llm/` and include focused tests.
Generated files, datasets without verified redistribution terms, and archive clutter must not be
committed.
