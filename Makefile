PYTHON ?= python3
EXAMPLE_SCRIPTS := $(sort $(wildcard examples/[0-9][0-9]_*.py))
EXAMPLE_ENV := PYTHONPATH=src MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

.PHONY: install test lint format typecheck docs build check examples snapshot clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests benchmarks examples

format:
	$(PYTHON) -m ruff format src tests benchmarks examples

typecheck:
	$(PYTHON) -m mypy src

docs:
	@printf 'Documentation build is introduced in a later increment.\n'

build:
	$(PYTHON) -m build

check: test lint typecheck

examples:
	@set -e; for example in $(EXAMPLE_SCRIPTS); do \
		printf '==> %s\n' "$$example"; \
		$(EXAMPLE_ENV) $(PYTHON) "$$example"; \
	done

snapshot:
	./.llm/snapshot.sh

clean:
	rm -rf build dist benchmarks/results .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find examples/results -type f ! -name .gitkeep -delete
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
