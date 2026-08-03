# Review checklist

- Scope is one coherent increment and follows `.llm/state.md` or an explicit owner reordering.
- Patch paths are repository-relative, apply cleanly to the uploaded snapshot, and are accompanied
  by a SHA-256 checksum.
- Source/tests agree with the claimed implemented state.
- Mathematical dimensions and invariants remain consistent.
- Numerical tolerances are scale-relative and tested.
- Public API names, defaults, supported composition, and array orientations match the contract.
- Learned preprocessing remains inside the correct CV boundary.
- Weighted fitting or unsupported metadata/composite behavior was not introduced implicitly.
- Real-data examples read and form `X` and `Y` visibly; no registry, generic loader,
  metadata-driven runtime path, or hidden I/O helper was introduced.
- Package-owned reference datasets keep one canonical `X.csv`/`Y.csv` pair with valid
  `metadata.json`, README, license, integrity, distribution, and raw-access contracts.
- Phase/default/roadmap changes update `.llm/state.md` and `.llm/strategy.md`.
- New accepted decisions have a numbered record and `.llm/decisions.md` entry.
- No generated files, caches, archive clutter, or unverified data are included.
- Validation report is complete and distinguishes passed, failed, and not-run targets.
