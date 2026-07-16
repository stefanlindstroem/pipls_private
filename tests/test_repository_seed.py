from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

import pipls


def test_package_imports() -> None:
    assert pipls.__version__ == "0.0.0"


def test_required_llm_contracts_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    required = {
        ".llm/README.md",
        ".llm/project.md",
        ".llm/mathematics.md",
        ".llm/numerical_contracts.md",
        ".llm/development.md",
        ".llm/public_api.md",
        ".llm/snapshot.sh",
        ".llm/create_patch.sh",
        ".llm/strategy.md",
    }
    missing = sorted(path for path in required if not (root / path).is_file())
    assert not missing, f"Missing repository contracts: {missing}"


def test_llm_layer_is_outside_installable_package() -> None:
    package_root = Path(pipls.__file__).resolve().parent
    assert ".llm" not in {part.name for part in package_root.parents}


def test_snapshot_has_repository_contents_at_archive_root(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    archive = tmp_path / "snapshot.tar.gz"

    subprocess.run(
        [str(root / ".llm" / "snapshot.sh"), str(archive)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    with tarfile.open(archive, "r:gz") as handle:
        names = {name.removeprefix("./") for name in handle.getnames()}

    assert "README.md" in names
    assert ".llm/SNAPSHOT_INFO" in names
    assert not any(name.startswith(f"{root.name}/") for name in names)


def test_llm_workflow_scripts_are_executable() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / ".llm" / "snapshot.sh",
        root / ".llm" / "create_patch.sh",
    ]
    assert all(path.stat().st_mode & 0o111 for path in scripts)


def test_strategy_declares_ownership_and_next_increment() -> None:
    root = Path(__file__).resolve().parents[1]
    strategy = (root / ".llm" / "strategy.md").read_text(encoding="utf-8")
    readme = (root / ".llm" / "README.md").read_text(encoding="utf-8")

    assert "The LLM maintainer owns" in strategy
    assert 'Phase C1: rank-bound helper and `predictor_rank="max"`' in strategy
    assert "The LLM maintainer updates this file" in readme
    assert 'git apply --check ~/Downloads/proposed-change.patch' in readme
    assert 'git commit -m "Describe the completed increment"' in readme
    assert not (root / '.llm' / 'apply_patch.sh').exists()
    assert not (root / '.llm' / 'commit.sh').exists()
