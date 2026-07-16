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
        ".llm/theory.md",
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
    assert 'Phase C2b: `predictor_rank="auto"`' in strategy
    assert "Phase C2c: split exhaustive and adaptive rank-search semantics" in strategy
    assert "Current status: **complete**. Exhaustive search" in strategy
    assert "Phase C2d: scalable linear-algebra policy" in strategy
    assert 'Current status: **complete**. The estimator now exposes `svd_solver=' in strategy
    assert "Phase C2e: public parameter-validation hardening" in strategy
    assert "Current status: **complete**. Constructor validation" in strategy
    assert "Phase D1: complete path analysis" in strategy
    assert "Phase D1a: shared private selection engine" in strategy
    assert "Phase D1b: scikit-learn and PLS-style API alignment" in strategy
    assert "Current status: **complete**. `PiPLSPathCV` is public" in strategy
    assert "The next implementation patch should be **Phase E1" in strategy
    assert (root / "docs" / "decisions" / "0005-leave-one-out-protocol.md").is_file()
    assert (root / "docs" / "decisions" / "0007-predictor-rank-search-policies.md").is_file()
    assert (root / "docs" / "decisions" / "0008-predictor-svd-policy.md").is_file()
    assert (root / "docs" / "decisions" / "0009-public-parameter-validation.md").is_file()
    assert (root / "docs" / "decisions" / "0010-path-analysis-api.md").is_file()
    assert (root / "docs" / "decisions" / "0011-shared-selection-engine.md").is_file()
    assert (root / "docs" / "decisions" / "0012-sklearn-api-alignment.md").is_file()
    assert (root / "docs" / "path_analysis.md").is_file()
    assert (root / "docs" / "cross_validation.md").is_file()
    assert "The LLM maintainer updates this file" in readme
    assert "git apply --check ~/Downloads/proposed-change.patch" in readme
    assert 'git commit -m "Describe the completed increment"' in readme
    assert not (root / ".llm" / "apply_patch.sh").exists()
    assert not (root / ".llm" / "commit.sh").exists()


def test_theory_reference_is_navigable_and_contains_core_identities() -> None:
    root = Path(__file__).resolve().parents[1]
    llm_readme = (root / ".llm" / "README.md").read_text(encoding="utf-8")
    theory = (root / ".llm" / "theory.md").read_text(encoding="utf-8")
    mathematics = (root / ".llm" / "mathematics.md").read_text(encoding="utf-8")
    docs_index = (root / "docs" / "index.md").read_text(encoding="utf-8")
    docs_theory = (root / "docs" / "theory.md").read_text(encoding="utf-8")

    assert "`theory.md` — persistent conceptual derivation" in llm_readme
    assert "`.llm/theory.md`" in mathematics
    assert "theory.md" in docs_index
    assert "../.llm/theory.md" in docs_theory

    required_theory_fragments = {
        "\\mathbf{Z}=\\mathbf{X}\\mathbf{\\Pi}",
        "\\mathbf{W}=\\mathbf{Z}^{+}\\mathbf{Y}\\mathbf{C}",
        "\\mathbf{P}=\\mathbf{\\Pi}\\mathbf{M}",
        "\\mathbf{Q}=\\mathbf{C}\\mathbf{N}",
        "\\mathbf{B}_{\\mathrm{cs}}",
        'predictor_rank="optimal"',
        'predictor_rank="auto"',
    }
    missing = sorted(fragment for fragment in required_theory_fragments if fragment not in theory)
    assert not missing, f"Theory reference is missing core fragments: {missing}"
