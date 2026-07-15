# Patch request

## Objective

[One bounded change.]

## Required reading

- `.llm/project.md`
- [relevant mathematical, numerical, API, and development contracts]
- [relevant source and test files]

## Acceptance criteria

- [observable behavior]
- [tests]
- [documentation]

## Output

Return one unified Git patch relative to repository root and a validation report:

```text
Validation:
- `make test`: passed / failed / not run
- `make lint`: passed / failed / not run
- `make typecheck`: passed / failed / not run
- `make build`: passed / failed / not run
```
