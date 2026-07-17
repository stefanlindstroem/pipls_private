from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

import pipls


def test_package_imports() -> None:
    assert pipls.__version__ == "0.0.0"


def test_packaging_uses_pep639_license_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires = ["setuptools>=77.0.3", "wheel"]' in pyproject
    assert 'license = "BSD-3-Clause"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert 'license = {file = "LICENSE"}' not in pyproject


def test_required_llm_contracts_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    required = {
        ".llm/README.md",
        ".llm/project.md",
        ".llm/state.md",
        ".llm/decisions.md",
        ".llm/theory.md",
        ".llm/mathematics.md",
        ".llm/numerical_contracts.md",
        ".llm/development.md",
        ".llm/public_api.md",
        ".llm/data_io.md",
        ".llm/dataset_layout.md",
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
    assert "Phase D1c: final scikit-learn cleanup boundary" in strategy
    assert "Current status: **complete**. `PiPLSPathCV` is public" in strategy
    assert "Phase E1: dataset schema and deterministic synthetic generator" in strategy
    assert "Current status: **complete**. `PiPLSDataset` validates" in strategy
    assert "Phase E2: transparent real-data input contract" in strategy
    assert "Current status: **complete**. `.llm/data_io.md`" in strategy
    assert "Current status: **underway**. Linnerud establishes" in strategy
    assert "Pulp is the second" in strategy
    assert "private-source references" in strategy
    assert "The next implementation patch should remain in **Phase E3**" in strategy
    assert (root / "docs" / "decisions" / "0005-leave-one-out-protocol.md").is_file()
    assert (root / "docs" / "decisions" / "0007-predictor-rank-search-policies.md").is_file()
    assert (root / "docs" / "decisions" / "0008-predictor-svd-policy.md").is_file()
    assert (root / "docs" / "decisions" / "0009-public-parameter-validation.md").is_file()
    assert (root / "docs" / "decisions" / "0010-path-analysis-api.md").is_file()
    assert (root / "docs" / "decisions" / "0011-shared-selection-engine.md").is_file()
    assert (root / "docs" / "decisions" / "0012-sklearn-api-alignment.md").is_file()
    assert (root / "docs" / "decisions" / "0014-validation-metadata-scope.md").is_file()
    assert (root / "docs" / "decisions" / "0015-dataset-and-synthetic-api.md").is_file()
    assert (root / "docs" / "decisions" / "0016-transparent-data-ingestion.md").is_file()
    assert (root / "docs" / "decisions" / "0017-first-real-dataset.md").is_file()
    assert (root / "docs" / "decisions" / "0018-repository-dataset-layout.md").is_file()
    assert (root / "docs" / "decisions" / "0019-pulp-dataset-integration.md").is_file()
    assert (root / "docs" / "decisions" / "0020-public-dataset-provenance-boundary.md").is_file()
    assert (root / "docs" / "path_analysis.md").is_file()
    assert (root / "docs" / "cross_validation.md").is_file()
    assert "The LLM maintainer updates this file" in readme
    assert "## Fresh-chat bootstrap" in readme
    assert "`.llm/SNAPSHOT_INFO`" in readme
    assert "`state.md`" in readme
    assert "`decisions.md`" in readme
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


def test_llm_fresh_chat_handoff_is_current_and_navigable() -> None:
    root = Path(__file__).resolve().parents[1]
    state = (root / ".llm" / "state.md").read_text(encoding="utf-8")
    project = (root / ".llm" / "project.md").read_text(encoding="utf-8")
    public_api = (root / ".llm" / "public_api.md").read_text(encoding="utf-8")
    decisions = (root / ".llm" / "decisions.md").read_text(encoding="utf-8")

    assert "Phases A through E2 are complete" in state
    assert "Phase E3 is underway" in state
    assert "Linnerud" in state
    assert "pulp" in state
    assert "no metadata, registry, or package-owned loader required for fitting" in state
    assert "comma-delimited `X.csv`, `Y.csv`, and documentary" in state
    assert "deterministic synthetic" in state
    assert (
        "weighted fitting and general sample-weight routing are intentionally out of scope"
        in state
    )
    assert "direct `PiPLSRegression` or `Pipeline`" in state
    assert "Routine work should not require re-uploading the manuscript" in state
    assert "adapted from public" in state

    assert "The next increment remains Phase E3" in project
    assert "transparent" in project
    assert "`.llm/state.md`: current handoff" in project

    assert "Arbitrary nested meta-estimators are rejected" in public_api
    assert "Weighted fitting" in public_api
    assert "composite estimator containing one" not in public_api
    assert "Plain arrays and data frames" in public_api

    decision_files = sorted(path.name for path in (root / "docs" / "decisions").glob("*.md"))
    missing = [name for name in decision_files if f"`{name}`" not in decisions]
    assert not missing, f"Decision index is missing records: {missing}"


def test_llm_data_io_contract_is_transparent() -> None:
    root = Path(__file__).resolve().parents[1]
    data_io = (root / ".llm" / "data_io.md").read_text(encoding="utf-8")
    dataset_layout = (root / ".llm" / "dataset_layout.md").read_text(encoding="utf-8")
    strategy = (root / ".llm" / "strategy.md").read_text(encoding="utf-8")

    assert "model = PiPLSRegression().fit(X, Y)" in data_io
    assert "must not require a registry, metadata file" in data_io
    assert "Do not hide these steps behind a package utility" in data_io
    assert "Every committed real dataset directory" in dataset_layout
    assert "`X.csv`: predictor matrix" in dataset_layout
    assert "`Y.csv`: response matrix" in dataset_layout
    assert "`metadata.yaml`: public repository description" in dataset_layout
    assert "use a comma as delimiter" in dataset_layout
    assert "not a runtime input" in dataset_layout
    assert "### Phase E3: real dataset integrations" in strategy
    assert "Current status: **underway**" in strategy
    assert not (root / "datasets" / "registry.yaml").exists()
    assert "private archive" in data_io
    assert "Corn is the planned special case" in data_io
    assert "Do not include:" in dataset_layout



def test_linnerud_is_repository_data_not_runtime_api() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")
    data_readme = (root / "datasets" / "linnerud" / "README.md").read_text(
        encoding="utf-8"
    )
    example = (root / "examples" / "09_linnerud_real_data.py").read_text(
        encoding="utf-8"
    )

    assert "recursive-include datasets" in manifest
    assert "recursive-include examples" in manifest
    assert "`X.csv` and `Y.csv`" in data_readme
    assert "pd.read_csv(DATA_DIR / \"X.csv\")" in example
    assert "pd.read_csv(DATA_DIR / \"Y.csv\")" in example
    assert "metadata.yaml" in example
    assert "import yaml" not in example
    assert "yaml.safe_load" not in example
    assert "load_dataset" not in example
    assert "linnerud" not in pipls.__all__



def test_pulp_is_repository_data_not_runtime_api() -> None:
    root = Path(__file__).resolve().parents[1]
    data_readme = (root / "datasets" / "pulp" / "README.md").read_text(encoding="utf-8")
    example = (root / "examples" / "10_pulp_real_data.py").read_text(encoding="utf-8")
    assert "14 fiber-description predictors" in data_readme
    assert 'pd.read_csv(DATA_DIR / "X.csv")' in example
    assert 'pd.read_csv(DATA_DIR / "Y.csv")' in example
    assert "metadata.yaml" in example
    assert "import yaml" not in example
    assert "load_dataset" not in example
    metadata = (root / "datasets" / "pulp" / "metadata.yaml").read_text(encoding="utf-8")
    assert "10.1016/j.compchemeng.2025.109143" in metadata
    assert "PiPLSR_v0.1" not in metadata
    assert "scripts/prepare_data" not in metadata
    assert not (root / "scripts" / "prepare_data" / "prepare_pulp.py").exists()
    assert "pulp" not in pipls.__all__

def test_llm_prompts_bootstrap_from_repository_state() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / ".llm" / "prompts").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        assert ".llm/" in text
        assert "root-relative" in normalized
        assert "unified Git patch" in normalized
        assert "validation results" in normalized
