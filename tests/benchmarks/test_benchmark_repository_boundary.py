from __future__ import annotations

from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_retired_universal_benchmark_architecture_is_absent() -> None:
    root = _repository_root()
    retired = [
        root / "benchmarks" / "run_synthetic.py",
        root / "benchmarks" / "manifests" / "synthetic-v1.yaml",
        root / "benchmarks" / "schema" / "result-v2.schema.json",
    ]
    assert not any(path.exists() for path in retired)


def test_benchmark_results_remain_generated_and_ignored() -> None:
    root = _repository_root()
    ignored = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "benchmarks/results/" in ignored
