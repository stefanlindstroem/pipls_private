# LLM-assisted development layer

This directory is the navigation and communication layer for Pi-PLS development. It is tracked
in Git, included in project snapshots, and excluded from the installable `pipls` package.

## Ownership and authority

The project owner controls scientific decisions, publication policy, and public API commitments.
The LLM maintainer is responsible for keeping the operational strategy and `.llm` contracts
consistent with each other and with the repository.

The principal navigation documents are:

1. `state.md` — concise fresh-chat handoff: implemented boundary, accepted defaults, exclusions,
   next increment, and revised roadmap. Update it whenever phase or public-scope state changes.
2. `product_scope.md` — normative boundary between the long-lived package repository and
   downstream publication-reproduction repositories.
3. `strategy.md` — current development principles, unresolved work, active acceptance
   conditions, and maintenance ownership. Completed increment history belongs in decisions and
   Git history.
4. `project.md` — concise repository and source-ownership map.
5. `decisions.md` — index of accepted decision records and implemented clarifications that
   supersede broader historical proposals.
6. `theory.md` — persistent conceptual derivation, interpretation, limiting cases, and
   theory-to-implementation consequences.
7. `mathematics.md` — normative equations, notation, dimensions, and mathematical invariants.
8. `numerical_contracts.md` — numerical algorithms, tolerances, degeneracy, and comparison rules.
9. `public_api.md` — constructor parameters, fitted attributes, shapes, supported composition,
   dataset interfaces, and explicit exclusions.
10. `data_io.md` — transparent real-data input and example contract.
11. `dataset_layout.md` — normative package-resource, integrity, raw-access, and single-copy
    contract for named reference datasets.
12. `analysis.md` — fitted-model interpretation, prediction-provenance, plotting, and analysis-
    artifact contracts.
13. `testing.md` — durable test boundaries for behavior, shipped-file structure, and living
    documents.
14. `development.md` — coding, testing, documentation, dependency, and patch requirements.

Git history, tests, and implemented behavior are evidence of what currently exists. Strategy and
plans describe intended work. A conflict must be surfaced and resolved explicitly.

## Fresh-chat bootstrap

A new chat must be able to proceed from the uploaded snapshot without prior conversation history.
First inspect `.llm/SNAPSHOT_INFO`, then read:

1. this file;
2. `state.md`;
3. `product_scope.md`;
4. `strategy.md`;
5. `project.md`;
6. `decisions.md` and the relevant full decision records;
7. `theory.md` for mathematical or model-selection work;
8. `data_io.md` and `dataset_layout.md` for real-data, example, or dataset work;
9. `analysis.md` for fitted-model interpretation, plotting, prediction diagnostics, or analysis
    artifacts;
10. `testing.md` before changing repository-document, metadata, or fixture tests;
11. the normative contracts relevant to the requested change;
12. the affected source, tests, and user-facing documentation.

Confirm that source and tests agree with the claimed state. Do not silently rely on remembered
chat context or old patch descriptions. Surface any conflict before implementation.

Do not implement work assigned to a later strategy phase unless the project owner explicitly
changes the sequence. Paper-reproduction work belongs in downstream repositories that pin released
versions of `pipls`. Routine package work should not require a manuscript because `theory.md` and
the normative contracts preserve the accepted scientific construction.

## Expected patch response

For a bounded implementation request, return:

- one downloadable unified Git patch relative to repository root;
- a separate SHA-256 checksum for that patch;
- a concise description of changed contracts and observable behavior;
- validation results for every applicable Makefile target, marked passed, failed, or not run;
- the concise routine command sequence for application, validation, staging, committing, and
  snapshot creation. Add troubleshooting or inspection commands only when the patch needs them.

Do not promise later/background work, bundle unrelated future phases, or require the user to
restate decisions already captured in the repository.

## User workflow

Start from a clean committed worktree. Snapshot creation fails when tracked, staged, or
nonignored untracked changes remain. Ignored local outputs do not enter the archive. Create and
upload a snapshot:

```bash
make snapshot
# inspect if desired:
tar -tzf ../pipls-snapshot.tar.gz | head
```

The snapshot contains the contents of the repository root, without an enclosing project-name
directory. Archive paths therefore match root-relative Git patch paths.

Request one small, testable increment. The response should provide one downloadable root-relative
unified Git patch, its SHA-256 checksum, and a validation report. Save both files, normally under
`~/Downloads`.

The routine command sequence shown with a patch is intentionally short:

```bash
git apply ~/Downloads/proposed-change.patch
make check
git add -A
git commit -m "Describe the completed increment"
make snapshot
```

The maintainer producing the patch must already have verified `git apply --check` against a clean
extraction. The owner may additionally inspect `git status`, `git diff`, or the staged diff at any
point; include those commands in the response only when a special migration or troubleshooting step
requires them. The supplied SHA-256 file supports optional patch-integrity verification.

To discard an uncommitted applied patch, use Git rather than a helper script:

```bash
git restore --staged .
git restore .
git clean -nd   # preview untracked files that would be removed
git clean -fd   # remove them only after reviewing the preview
```

The complete exchange cycle is therefore:

```text
commit clean state -> make snapshot -> upload -> receive patch and checksum -> git apply
-> make check -> git add -A -> git commit -> make next snapshot
```

## Helper scripts

- `snapshot.sh [OUTPUT]`: create an upload tarball from a clean committed Git tree. The helper
  refuses tracked, staged, or nonignored untracked changes; ignored generated files are excluded
  because only `HEAD` is archived. It also rejects committed files below `examples/results/` other
  than the directory-preserving `.gitkeep` placeholders.
- `create_patch.sh [OUTPUT]`: optionally export current unstaged changes as a root-relative patch.

Patch application and committing deliberately use ordinary Git commands. This keeps behavior
visible, avoids hidden staging or commit actions, and makes troubleshooting independent of project
shell wrappers.

Git and tests remain authoritative. The `.llm` layer standardizes communication and maintenance;
it does not replace scientific review, code review, release notes, or ordinary Git inspection.
