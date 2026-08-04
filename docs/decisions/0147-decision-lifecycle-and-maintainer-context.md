# Decision 0147: decision lifecycle and maintainer-context consolidation

## Status

Accepted. Patches 1 and 2 of seven are complete. The decision lifecycle is established and the
active `.llm` layer now describes current contracts rather than completed transition history. No
existing decision record is retired before Patch 3 supplies an explicit supersession map.

## Context

The repository now contains 146 numbered decision records and more than five thousand lines of
active `.llm` guidance. These records document valuable architectural reasoning, but many describe
superseded APIs, one-off migrations, completed patch sequences, or cleanup work whose durable
outcome is already captured by a later canonical decision. Keeping every intermediate record in the
active decision index makes current policy harder to identify and encourages stale wording to remain
in the maintainer context.

The same problem appears in the active `.llm` layer. Current contracts, completed transition logs,
and historical implementation narratives are mixed together. This increases maintenance cost and
can cause a future maintainer to follow an obsolete intermediate state instead of the implemented
surface.

Git already preserves the complete repository history. The maintained tree therefore needs a clear
lifecycle for decision records and a smaller current-state guide layer, without discarding durable
scientific, numerical, API, distribution, or repository contracts.

## Decision

### Classify maintained decision material

Decision material has three maintained classes.

**Current decisions** remain as numbered records under `docs/decisions/`. A current decision still
defines at least one of:

- observable public or numerical behavior;
- a repository, distribution, documentation, or testing policy;
- a scientific or data-governance boundary;
- an unresolved accepted roadmap that still constrains implementation.

**Historical summaries** compact completed development eras into `docs/decisions/history.md`. A
summary records the durable outcome, names the current canonical decisions, and gives enough context
to understand why the repository has its present shape. It does not reproduce deleted decisions or
become a second patch log.

**Retired decisions** are removed from the maintained tree and recovered from Git history when
needed. Their decision numbers are never reused.

### Retire by relevance, not age

A numbered decision is eligible for retirement when it:

- describes a public API or repository shape that no longer exists;
- is explicitly superseded by a later canonical decision;
- records only patch ordering, migration progress, or a one-time cleanup;
- duplicates a later decision that fully defines the durable contract;
- documents an intermediate naming or compatibility state with no remaining active effect.

A decision must not be retired merely because it is old. Foundational mathematical, numerical,
scientific, data-provenance, compatibility, distribution, and repository-policy records remain
current while they still constrain the package.

### Require an explicit retirement map

Before deleting numbered records, the retirement patch must provide an explicit map from each
retired decision to one of:

- a current canonical decision;
- a section of `docs/decisions/history.md`;
- Git history alone, when the record describes only a one-off migration or cleanup.

The same patch must remove or redirect active links from `.llm`, tests, documentation, and retained
decisions. No copied archive, tarball, or hidden duplicate of the retired Markdown files is added to
the repository.

### Keep the active indexes current-focused

After consolidation:

- `docs/decisions/index.md` lists current numbered decisions and links to the historical summary;
- `.llm/decisions.md` is a concise registry of current decisions and implemented clarifications;
- `docs/decisions/history.md` summarizes completed eras without attempting exhaustive
  reconstruction;
- Git remains the authoritative archive for retired records.

The target of approximately 35--50 current decisions is a review goal, not a test-pinned invariant.
The retained count follows the relevance criteria above.

### Compact the active maintainer layer

The active `.llm` files describe the implemented package and current work only. Completed patch
narratives and superseded intermediate contracts move to the historical summary or disappear when
Git history is sufficient. In particular:

- `state.md` records current implementation state;
- `strategy.md` records unresolved work and the current roadmap;
- `testing.md` records current validation obligations;
- `analysis.md` records current architectural and interpretation reasoning;
- `public_api.md` and `numerical_contracts.md` describe only the implemented surface.

### Preserve behavior during documentation cleanup

Decision retirement and `.llm` compaction do not authorize numerical, public-API, dataset,
distribution, or rendering changes. Later source refactoring remains separately reviewable. The
public `pipls.datasets` import surface also remains unchanged when the oversized dataset module is
split internally.

## Patch sequence

1. Establish this decision and synchronize the guide-layer phase state -- complete.
2. Compact and correct the active `.llm` layer around current contracts -- complete.
3. Retire explicitly superseded decisions using a reviewed retirement map.
4. Add `docs/decisions/history.md` and consolidate completed micro-decisions, targeting roughly
   35--50 current records.
5. Simplify brittle repository and workflow-structure tests and harden snapshot creation against
   tracked caches and generated artifacts.
6. Split `datasets.py` into private type, resource-loading, and synthetic-generation modules while
   preserving the public façade.
7. Normalize links and indexes, complete stale-surface and repository-artifact audits, and mark this
   decision implemented.

## Validation

Every patch must pass `git diff --check`, Python compilation, Ruff, mypy, the complete pytest suite,
and strict MkDocs validation where available.

Decision-retirement patches must additionally verify that every shipped numbered decision is
indexed, every index link resolves, every active decision reference resolves, retired numbers are
not reused, and the retirement map covers every deleted record. Tests must not enforce a fixed
number of decisions or preserve one tombstone assertion per retired record.

The snapshot-hardening patch must demonstrate that untracked ignored caches remain excluded and
that a deliberately tracked cache or generated artifact causes snapshot creation to fail.

The dataset split must preserve public imports, packaged resource contents, deterministic synthetic
outputs, immutable result behavior, pickling, and distribution contents exactly.

## Consequences

Maintainers get a smaller current-policy surface while Git retains the complete development record.
Decision numbers remain stable and meaningful, but intermediate migration documents no longer
compete with current contracts. Future decisions enter the current registry when accepted and may
later be summarized or retired through the same explicit process.
