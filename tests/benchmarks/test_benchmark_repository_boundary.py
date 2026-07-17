from __future__ import annotations

import subprocess
import tarfile
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


def test_benchmark_results_are_excluded_from_snapshots(tmp_path: Path) -> None:
    root = _repository_root()
    result = root / "benchmarks" / "results" / "snapshot-test.csv"
    archive = tmp_path / "snapshot.tar.gz"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text("generated\n", encoding="utf-8")
    try:
        subprocess.run(
            [str(root / ".llm" / "snapshot.sh"), str(archive)],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        result.unlink(missing_ok=True)

    with tarfile.open(archive, "r:gz") as handle:
        names = {name.removeprefix("./") for name in handle.getnames()}

    assert not any(name.startswith("benchmarks/results/") for name in names)
