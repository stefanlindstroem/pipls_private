PYTHON ?= python3
EXAMPLE_SCRIPTS := $(sort $(wildcard examples/[0-9][0-9]_*.py))
EXAMPLE_ENV := PYTHONPATH=src MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

.DEFAULT_GOAL := help

.PHONY: help install test lint format typecheck docs-figures docs docs-serve docs-dist build dist-check check examples snapshot clean

help: ## Show the available Make targets.
	@printf 'Usage: make <target>\n\nAvailable targets:\n'
	@awk 'BEGIN {FS = ":.*## "} /^[A-Za-z0-9_.-]+:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the editable package with development dependencies.
	$(PYTHON) -m pip install -e ".[dev]"

test: ## Run the test suite.
	PYTHONPATH=src $(PYTHON) -m pytest -q

lint: ## Run Ruff lint checks.
	$(PYTHON) -m ruff check src tests benchmarks examples tools

format: ## Format Python files with Ruff.
	$(PYTHON) -m ruff format src tests benchmarks examples tools

typecheck: ## Run strict mypy checks.
	$(PYTHON) -m mypy src

docs-figures: ## Generate tutorial figures.
	$(EXAMPLE_ENV) $(PYTHON) tools/render_synthetic_tutorial.py
	$(EXAMPLE_ENV) $(PYTHON) tools/render_pulp_tutorial.py

docs: docs-figures ## Build the strict documentation site.
	$(PYTHON) -m mkdocs build --strict

docs-serve: docs-figures ## Preview documentation at http://127.0.0.1:8000/.
	@printf 'Documentation preview: http://127.0.0.1:8000/ (stop with Ctrl+C)\n'
	$(PYTHON) -m mkdocs serve --dev-addr=127.0.0.1:8000

docs-dist: ## Verify documentation from a clean source distribution.
	$(PYTHON) tools/check_sdist_docs.py

build: ## Build the wheel and source distribution.
	$(PYTHON) -m build

dist-check: ## Verify clean wheel and source-distribution installations.
	$(PYTHON) tools/check_distributions.py

check: test lint typecheck ## Run tests, lint, and type checks.

examples: ## Run all numbered examples.
	@set -e; for example in $(EXAMPLE_SCRIPTS); do \
		printf '==> %s\n' "$$example"; \
		$(EXAMPLE_ENV) $(PYTHON) "$$example"; \
	done

snapshot: ## Create an uploadable repository snapshot.
	./.llm/snapshot.sh

clean: ## Remove generated files and caches.
	rm -rf build dist site docs/assets/generated benchmarks/results .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find examples/results -type f ! -name .gitkeep -delete
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
