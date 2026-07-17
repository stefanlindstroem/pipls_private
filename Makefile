PYTHON ?= python3

.PHONY: install test lint format typecheck docs build check benchmark-ci snapshot clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests benchmarks

format:
	$(PYTHON) -m ruff format src tests benchmarks

typecheck:
	$(PYTHON) -m mypy src

docs:
	@printf 'Documentation build is introduced in a later increment.\n'

build:
	$(PYTHON) -m build

check: test lint typecheck

benchmark-ci:
	rm -f benchmarks/results/synthetic-ci.jsonl
	PYTHONPATH=src $(PYTHON) benchmarks/run_synthetic.py --tier ci --output benchmarks/results/synthetic-ci.csv
	@printf 'Wrote benchmarks/results/synthetic-ci.csv\n'

snapshot:
	./.llm/snapshot.sh

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
