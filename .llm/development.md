# Development contract

## Startup and scope

- Inspect `.llm/SNAPSHOT_INFO` before work and note whether the uploaded snapshot was clean.
- Read `.llm/README.md`, `.llm/state.md`, `.llm/product_scope.md`, `.llm/strategy.md`,
  `.llm/project.md`, and the relevant decision records and contracts.
- Treat source and tests as evidence of implemented behavior; do not rely on prior chat history or
  an old patch description.
- Work on the current increment unless the project owner explicitly changes the order.
- Ask for a scientific or public-API decision only when the repository leaves a material choice
  unresolved. Do not ask the user to repeat information already captured in the repository.

## Implementation

- Follow scikit-learn estimator conventions for constructor parameters, cloning, validation,
  fitted attributes, feature names, and scalar `score()` on fitted model estimators. Search objects
  own evidence and do not delegate fitted-model methods.
- Keep constructor arguments unchanged unless an accepted decision explicitly changes the public
  lifecycle. Decision 0137 removed constructor-time selection, refit, and OOF controls; do not add
  aliases, ignored arguments, deprecation paths, or fallback state.
- Keep the fixed numerical core independent from preprocessing, CV, datasets,
  and publication-specific workflows.
- Reuse the shared private evaluation/search machinery rather than adding a second fold loop.
- Use one private selected-row resolver for rule/component-count selection and one shared private
  exact-compatibility validator for existing selections consumed by `refit(selection=...)` and
  `oof_report(selection=...)`. `refit()` returns a fitted estimator clone and does not mutate search
  state or retain training data. `oof_report(selection=...)` reuses the exact splits materialized by
  the search and does not perform a full-data fit.
- Preserve current estimator-internal centering/scaling: fit its statistics inside every
  candidate training fold and refit them on the complete training set after selection.
- Fit every additional learned preprocessing operation inside its matching training fold.
- Add dependencies only when a short, stable NumPy/scikit-learn implementation is insufficient.
- Keep executable example dependencies under the `examples` extra; the `dev` extra must include
  them so repository validation does not skip example artifacts.
- Keep optional dependency groups tied to maintained workflows: `examples` for numbered examples,
  `docs` for strict documentation construction, and `dev` for repository validation. Do not expose
  a dataset-access extra when datasets are ordinary repository files, and do not retain tools that
  no maintained target invokes.
- Public source-checkout instructions use ordinary noneditable installation. Editable installation
  belongs to `CONTRIBUTING.md` and maintainer workflows.
- Keep rendering data-first: the package computes immutable numerical results, while maintained
  examples and tutorial renderers use ordinary Matplotlib directly. Retain
  `biplot_coordinates()` as numerical preparation and use optional `adjustText` only after final
  axis configuration. Private rendering functions may clarify a complete numbered example when
  they remain in that same script, accept completed results explicitly, and perform no data loading,
  fitting, cross-validation, selection, or inspection calculation. Do not add public `plot_*`
  functions, a plotting submodule, or a shared support helper that hides chart construction.
- Do not broaden supported estimator composition or metadata routing implicitly.
- Weighted fitting and `sample_weight` propagation during fitting are out of scope unless the
  project owner explicitly reverses that decision.
- Every numbered example must present a recognizable user task, explicit comparison, or focused
  comparison. It must explain its data and label its output without relying on publication context or
  earlier project history. Do not ship context-free API demonstrations as numbered examples.
- In maintained evidence-retaining workflows, inspect path evidence, create one selection through
  `search.select(...)`, pass that same object to any OOF report, and fit the final full-data model
  through `search.refit(..., selection=selection)`. Do not recover the working selection from the
  fitted model or manually transfer the selected predictor rank into a new estimator. Compact
  automatic workflows may continue to refit directly from a rule or component count.
- For real-data examples, form `X` and `Y` visibly in the script. Package-owned Pulp, Sugarcane,
  and Tobacco use their named loaders; ordinary user data retain explicit user-owned reading. Do not
  introduce a registry, generic loader, metadata-driven runtime path, or helper that obscures data
  acquisition.
- Package-owned Pulp, Sugarcane, and Tobacco follow `.llm/dataset_layout.md`: one canonical
  comma-delimited `X.csv`/`Y.csv` pair plus `metadata.json`, `README.md`, and `LICENSE.txt` under
  `src/pipls/_data/<dataset>/`, with no duplicate top-level dataset copy.
- Treat every committed dataset asset as public-facing. Cite only public or included sources;
  do not commit private archive paths, inaccessible source checksums, or preparation-only scripts.
- Add public reconstruction or preprocessing code only when it operates on included or publicly
  obtainable raw data and exposes analysis-relevant choices that users should follow.
- Do not add paper-figure, manuscript-table, publication-grid, or paper-only comparator workflows
  to this repository. Downstream reproduction repositories should pin tagged `pipls` releases.
- Do not introduce provisional block-aware scaling classes, public names, constructor parameters,
  or internal abstractions until the owner starts a dedicated future design phase. This restriction
  does not defer or weaken the existing `PiPLSRegression` centering/scaling contract.

## Tests and documentation

- Follow `.llm/testing.md`: test executable behavior and durable machine contracts, not the
  current prose or individual field values of living guidance and documentary metadata.
- For shipped Markdown and YAML, prefer existence, UTF-8 decoding, parsability, and generic
  structural consistency over phrase matching or copied field values.
- Keep public Markdown under `docs/` self-contained. Do not link to `.llm`, example source files,
  dataset README files, or other paths outside `docs/`; summarize those public workflows in dedicated
  documentation pages instead.
- User guides describe implemented theory and behavior. Do not preserve rejected or unused options
  merely because they appeared in design discussions.
- During a staged public-API transition, living user documentation must describe the implemented
  boundary, not a later authorized stage. Guide-layer planning may describe the target explicitly as
  pending. Update public documentation in the patch that makes the new boundary true.
- Keep one task-oriented troubleshooting page and one API result-object map. Validate local
  documentation links and anchors generically; do not duplicate explanatory sentences in tests.
- Present estimator- and helper-returned immutable records as fields to inspect rather than
  constructors to call. Keep constructor signatures visible only for records whose direct user
  construction is part of the supported workflow.
- Do not expose internal phase or patch labels in served user guides. Maintainer chronology belongs
  in `.llm` and excluded decision records.
- Keep the root README focused on package users: restrained application-oriented motivation,
  installation, compact fixed and selected-model workflows, tutorial routes, and public reference
  links. Do not make general performance claims. Development environments, Make targets,
  distribution checks, repository layout, and snapshot instructions belong in `CONTRIBUTING.md`.
- Keep tutorial openings focused on purpose, coverage, setup or data, and the modeling workflow.
  The three served tutorials each include one compact vertical Mermaid flowchart plus equivalent
  prose; diagrams remain supplementary source-level documentation and produce no committed image
  assets. Put maintained source paths, renderer ownership, and figure-generation commands in a
  terminal reproduction section.
- Keep repository commands discoverable through the grouped self-documenting Makefile. `make`
  and `make help` show first setup and routine validation before the maintained task groups; each
  public target carries one `##` description and each group one `##@` heading.
- Build public documentation with the dedicated `docs` dependency extra and `make docs`. Use
  `make docs-serve` for a live local preview at `http://127.0.0.1:8000/`. The build
  is strict: navigation, internal links, anchors, mathematics support, and generated API targets must
  remain warning-free. Core generated pages use explicit public objects and source docstrings; do not
  expose private modules or inherited implementation machinery by broad module expansion. Never commit
  generated `site/` output or `docs/assets/generated/` tutorial assets. `make docs` and
  `make docs-serve` regenerate the synthetic and Pulp assets through `make docs-figures`. Tutorial
  code excerpts use checked `pymdownx.snippets` sections from repository source; do not copy the
  maintained example analyses into Markdown. Use `make docs-dist`
  when changing documentation packaging or the source-distribution documentation boundary; it
  performs a clean install and strict build from
  the unpacked sdist and therefore remains outside the ordinary fast test target. Use
  `make dist-check` for packaging, dependency, public-module, or included-data changes; it builds
  once and checks separate clean wheel and sdist installations outside the checkout. The sdist
  check also installs the `examples` extra, runs example 01 from the extracted source tree, and
  verifies that its repository-owned output directory is present and writable.
- The dedicated documentation workflow runs the strict checkout and source-distribution builds on
  pushes and pull requests. Pushes to `master` additionally derive the default GitHub Pages and
  repository URLs from the Actions context, rebuild the site with those canonical values, and
  deploy `site/`. Keep Pages write and identity-token permissions confined to the deployment job;
  never commit the temporary `.mkdocs-pages.yml` file or generated site.
- Exact dataset values, shapes, columns, or hashes require an explicit fixture decision; do not
  freeze them accidentally in repository-layout tests.
- Preserve fold-local standardization and keep generated outputs out of Git unless a decision
  explicitly freezes a narrow fixture.
- Design files and command outputs for both humans and machines. When results are naturally tabular,
  prefer flat UTF-8 CSV with explicit self-explanatory columns over nested serialization.
- Numbered examples migrated under Phase F4 operate directly on immutable path and inspection results
  in memory. They must not write generated CSV files as analytical or plotting intermediates.
- Preserve `cv_mse_std` as descriptive population split dispersion. Maintained CV-MSE figures
  use it directly for symmetric $\pm 1$ SD bars. No standard-error result or selection rule is
  public. Selection lookup must not fit, refit, or mutate search state. Decision 0140 assigns that
  ownership to `search.select(...)`, and maintained examples, tutorial snippets, and living API
  pages use that operation. Under Decision 0151, Example 07 shows the exact minimum row, horizontal
  10% relative-tolerance threshold, and recommended pre-refit selection, then passes that same
  object to OOF reporting and final fitting. Keep generated pages and cross-links synchronized with
  the implemented stage.
- Every behavioral change requires focused tests at the most public relevant boundary.
- Mathematical changes update `.llm/mathematics.md`, `.llm/theory.md`, and user-facing theory
  documentation when applicable.
- Numerical changes update `.llm/numerical_contracts.md` and include deterministic boundary tests.
- Public API changes update `.llm/public_api.md`, user documentation, and applicable estimator/API
  tests.
- Dataset or example-I/O changes update `.llm/data_io.md` and include a review of whether `X` and
  `Y` construction remains transparent.
- Product-scope or publication-boundary changes update `.llm/product_scope.md`.
- Phase, roadmap, ownership, default, or supported-scope changes update `.llm/state.md` and
  `.llm/strategy.md` in the same patch.
- New accepted architectural or public-API decisions receive a numbered record under
  `docs/decisions/` and an entry in `.llm/decisions.md`.
- Do not edit generated files or commit caches, build products, archive clutter, or unverified
  datasets.
- Create handoff snapshots only from a clean committed Git tree. `make snapshot` must refuse
  tracked, staged, or nonignored untracked changes and must archive `HEAD`, so ignored generated
  files cannot enter the handoff. It must also refuse a clean `HEAD` that contains tracked caches, bytecode, coverage output,
  generated documentation, build products, or files below `examples/results/` other than
  `.gitkeep` placeholders.

## Delivery and validation

- Return one downloadable unified Git patch relative to repository root and a separate SHA-256
  checksum file.
- Report each applicable target as passed, failed, or not run; never describe inspection alone as
  validation.
- Run focused tests while developing, then `make check` before delivery.
- Run `make docs-figures` for changes to either tutorial example, renderer, display subset, or plot
  behavior used by generated assets.
- Run `make docs` for changes to public guides, navigation, documentation configuration, public
  docstrings, or generated tutorial assets.
- Run `make examples` for changes to numbered examples, example-generated final PDF
  artifacts, or the
  application-facing workflow. This target runs all examples, including Tobacco.
- Run `make build` for a quick artifact build and `make dist-check` for packaging, dependency,
  public-module, or included-data changes before delivery.
- Verify `git apply --check` against a clean extraction before delivery. Show the owner the concise
  routine sequence `git apply`, `make check`, `git add -A`, `git commit`, and `make snapshot`; add
  inspection or repair commands only when they are specifically needed.
