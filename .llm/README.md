# LLM-assisted development layer

This directory is the navigation and communication layer for Pi-PLS development. It is tracked
in Git, included in project snapshots, and excluded from the installable `pipls` package.

## Ownership and authority

The project owner controls scientific decisions, publication policy, and public API commitments.
The LLM maintainer is responsible for keeping the operational strategy and `.llm` contracts
consistent with each other and with the repository.

The principal navigation documents are:

1. `strategy.md` — current implementation phase, next increment, acceptance conditions, and
   maintenance ownership. The LLM maintainer updates this file when strategy state changes.
2. `project.md` — concise map of the repository and current implemented scope.
3. `mathematics.md` — defining equations, notation, dimensions, and mathematical invariants.
4. `numerical_contracts.md` — numerical algorithms, tolerances, degeneracy, and comparison rules.
5. `public_api.md` — intended constructor parameters, fitted attributes, shapes, and exclusions.
6. `development.md` — coding, testing, documentation, dependency, and patch requirements.
7. `docs/publication_repository_plan.md` — detailed design record and publication architecture.

Git history, tests, and implemented behavior are evidence of what currently exists. Strategy and
plans describe intended work. A conflict must be surfaced and resolved explicitly.

## Reading order for an LLM-assisted change

Read:

1. this file;
2. `strategy.md`;
3. `project.md`;
4. the contracts relevant to the requested change;
5. the affected source, tests, and user-facing documentation.

Do not implement work assigned to a later strategy phase unless the project owner explicitly
changes the sequence.

## User workflow

Start from a clean committed worktree. Create and upload a snapshot:

```bash
make snapshot
# inspect if desired:
tar -tzf ../pipls-snapshot.tar.gz | head
```

The snapshot contains the contents of the repository root, without an enclosing project-name
directory. Archive paths therefore match root-relative Git patch paths.

Request one small, testable increment. The response should provide one root-relative unified Git
patch and a validation report. Save the patch, normally under `~/Downloads`.

From the repository root, check and apply it directly with Git:

```bash
git status --short
git apply --check ~/Downloads/proposed-change.patch
git apply ~/Downloads/proposed-change.patch
```

The first command should normally print nothing. The applicability check and application are
silent on success. Then inspect and validate:

```bash
git status
git diff --check
git diff
make check
```

When satisfied, stage and inspect exactly what will be committed:

```bash
git add -A
git diff --cached --check
git diff --cached --stat
git diff --cached
```

Commit only after reviewing the staged diff:

```bash
git commit -m "Describe the completed increment"
git status
make snapshot
```

To discard an uncommitted applied patch, use Git rather than a helper script:

```bash
git restore --staged .
git restore .
git clean -nd   # preview untracked files that would be removed
git clean -fd   # remove them only after reviewing the preview
```

The complete exchange cycle is therefore:

```text
commit clean state -> make snapshot -> upload -> receive patch -> git apply --check
-> git apply -> inspect and validate -> git add -> inspect staged diff -> git commit
-> make next snapshot
```

## Helper scripts

- `snapshot.sh [OUTPUT]`: create a clean deterministic upload tarball.
- `create_patch.sh [OUTPUT]`: optionally export current unstaged changes as a root-relative patch.

Patch application and committing deliberately use ordinary Git commands. This keeps behavior
visible, avoids hidden staging or commit actions, and makes troubleshooting independent of project
shell wrappers.

Git and tests remain authoritative. The `.llm` layer standardizes communication and maintenance;
it does not replace scientific review, code review, release notes, or ordinary Git inspection.
