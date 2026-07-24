# Decision 0098: grouped maintainer command index

## Context

The repository has sixteen maintained Make targets covering setup, routine checks, documentation,
examples, distribution validation, snapshots, and cleanup. The former `make help` output presented
all targets in one flat list. Although every target had a description, a new contributor could not
see which commands formed the normal entry route or which commands belonged to a particular kind
of work.

The target names and recipes are already used by documentation, tests, and CI. The problem is
therefore command orientation, not missing functionality or excessive target count.

## Decision

1. Retain every public Make target and its existing behavior.
2. Make `make` and `make help` identify `make install` as first setup and `make check` as routine
   validation before listing task-specific commands.
3. Group the maintained targets under four headings: Start here, Development, Documentation and
   examples, and Distribution and maintenance.
4. Keep the Makefile self-documenting: each public target has one `##` description and each help
   section has one `##@` heading.
5. Present the same hierarchy in `CONTRIBUTING.md`, with `make install` and `make check` as the
   primary contributor route.
6. Test the section order, target membership, and primary commands without freezing complete help
   prose or target descriptions.

## Consequences

Contributors now see the normal setup and validation route before specialized commands. Existing
commands, CI jobs, documentation builds, distribution checks, and snapshot behavior are unchanged.
Future targets must be placed in one maintained help group rather than appended to an
undifferentiated list.
