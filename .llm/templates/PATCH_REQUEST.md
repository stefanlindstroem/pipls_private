# Patch request

## Objective

[One bounded change.]

## Snapshot and current state

- Snapshot cleanliness/commit from `.llm/SNAPSHOT_INFO`: [value]
- Current increment from `.llm/state.md`: [phase]
- Explicit owner-directed reordering, if any: [none / describe]

## Required reading

- `.llm/README.md`
- `.llm/state.md`
- `.llm/strategy.md`
- `.llm/project.md`
- `.llm/decisions.md` and relevant full decision records
- [relevant mathematical, numerical, API, and development contracts]
- [relevant source, tests, and user-facing documentation]

## Acceptance criteria

- [observable behavior]
- [tests]
- [contract/documentation updates]
- [scope explicitly excluded from this patch]

## Output

Return one unified Git patch relative to repository root, a concise behavioral summary, exact
direct Git commands, and a validation report:

```text
Validation:
- focused tests: passed / failed / not run
- `make test`: passed / failed / not run
- `make lint`: passed / failed / not run
- `make typecheck`: passed / failed / not run
- `make build`: passed / failed / not run
```
