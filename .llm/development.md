# Development contract

## Startup and scope

- Inspect `.llm/SNAPSHOT_INFO` before work and note whether the uploaded snapshot was clean.
- Read `.llm/README.md`, `.llm/state.md`, `.llm/strategy.md`, `.llm/project.md`, and the relevant
  decision records and contracts.
- Treat source and tests as evidence of implemented behavior; do not rely on prior chat history or
  an old patch description.
- Work on the current increment unless the project owner explicitly changes the order.
- Ask for a scientific or public-API decision only when the repository leaves a material choice
  unresolved. Do not ask the user to repeat information already captured in the repository.

## Implementation

- Follow scikit-learn estimator conventions for constructor parameters, cloning, validation,
  fitted attributes, feature names, conditional delegation, and scalar `score()`.
- Keep constructor arguments unchanged; resolve data-dependent values in `fit()`.
- Keep the fixed numerical core independent from preprocessing, CV, datasets, and paper policy.
- Reuse the shared private evaluation/search machinery rather than adding a second fold loop.
- Fit every learned preprocessing operation inside its matching training fold.
- Add dependencies only when a short, stable NumPy/scikit-learn implementation is insufficient.
- Do not broaden supported estimator composition or metadata routing implicitly.
- Weighted fitting and `sample_weight` propagation during fitting are out of scope unless the
  project owner explicitly reverses that decision.

## Tests and documentation

- Every behavioral change requires focused tests at the most public relevant boundary.
- Mathematical changes update `.llm/mathematics.md`, `.llm/theory.md`, and user-facing theory
  documentation when applicable.
- Numerical changes update `.llm/numerical_contracts.md` and include deterministic boundary tests.
- Public API changes update `.llm/public_api.md`, user documentation, and applicable estimator/API
  tests.
- Phase, roadmap, ownership, default, or supported-scope changes update `.llm/state.md` and
  `.llm/strategy.md` in the same patch.
- New accepted architectural or public-API decisions receive a numbered record under
  `docs/decisions/` and an entry in `.llm/decisions.md`.
- Do not edit generated files or commit caches, build products, archive clutter, or unverified
  datasets.

## Delivery and validation

- Return one unified Git patch relative to repository root.
- Report each applicable target as passed, failed, or not run; never describe inspection alone as
  validation.
- Run focused tests while developing, then `make check` before delivery.
- Run `make build` for packaging, dependency, public-module, or included-data changes.
- Provide exact direct Git commands for `git apply --check`, application, inspection, validation,
  staging, committing, and `make snapshot`.
