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
patch and a validation report. Save the patch, normally under `~/Downloads`, and apply it from the
repository root:

```bash
.llm/apply_patch.sh ~/Downloads/proposed-change.patch
```

Then validate and inspect:

```bash
make check
git diff
git status
```

When satisfied, commit explicitly:

```bash
.llm/commit.sh "Describe the completed increment"
```

`commit.sh` runs `make check`, rejects whitespace errors, stages all repository changes, displays
the staged summary, and commits with the supplied message. It does not create a snapshot. After a
successful commit, create the next snapshot with `make snapshot`.

The complete exchange cycle is therefore:

```text
commit clean state -> make snapshot -> upload -> receive patch -> apply patch
-> inspect and validate -> commit -> make next snapshot
```

## Helper scripts

- `snapshot.sh [OUTPUT]`: create a clean deterministic upload tarball.
- `apply_patch.sh [--allow-dirty] PATCH`: validate and apply a root-relative patch; never commits.
- `create_patch.sh [OUTPUT]`: export current unstaged changes as a root-relative patch.
- `commit.sh COMMIT_MESSAGE`: validate, stage, and commit the accepted increment.

Git and tests remain authoritative. The `.llm` layer standardizes communication and maintenance;
it does not replace scientific review, code review, release notes, or ordinary Git inspection.
