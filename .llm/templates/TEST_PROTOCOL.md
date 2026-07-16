# Test protocol

1. Run focused tests while developing.
2. Add tests at the most public boundary affected by the change.
3. Run `make check` before delivery.
4. Run `make build` for packaging, dependency, public-module, or included-data changes.
5. Record every command as passed, failed, or not run.
6. Never describe inspection, import success, or type checking alone as full validation.
7. For `.llm` changes, run the repository-seed tests that protect navigation and snapshot startup.
