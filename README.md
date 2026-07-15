# Pi-PLS

Development repository for Pi-PLS, a PLS-family method for multivariate regression.

This initial repository increment establishes packaging, validation commands, Git patch
exchange, and tarball snapshots. The numerical implementation is intentionally deferred
to the next testable increment.

## Bootstrap

```bash
git init
git add .
git commit -m "Initialize Pi-PLS repository"
python -m pip install -e ".[dev]"
make check
make snapshot
```

Read `.llm/README.md` before preparing an LLM-assisted change.
