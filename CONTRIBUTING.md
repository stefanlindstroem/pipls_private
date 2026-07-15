# Contributing

Use a clean Git worktree and make one reviewable change at a time.

Before submitting a patch, run:

```bash
make check
```

Mathematical changes must update the relevant contracts in `.llm/` and include focused tests.
Generated files, datasets without verified redistribution terms, and archive clutter must not be committed.
