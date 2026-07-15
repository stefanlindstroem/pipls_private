# LLM-assisted development layer

This directory contains repository communication contracts and helper scripts. It is
tracked in Git but is not installed with `pipls`.

For each change, read in this order:

1. `project.md`
2. `mathematics.md` for mathematical work
3. `numerical_contracts.md` for numerical work
4. `public_api.md` for API work
5. `development.md`
6. the relevant source and test files

Normal exchange:

```bash
make snapshot
# upload the generated tarball and request one root-relative unified Git patch
.llm/apply_patch.sh proposed-change.patch
make check
```

Git and tests remain authoritative. `.llm` standardizes communication; it does not replace review.
