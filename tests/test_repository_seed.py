from __future__ import annotations

import re
import subprocess
import tarfile
from pathlib import Path

import pipls

_REQUIRED_LLM_CONTRACTS = {
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
    ".llm/testing.md",
    ".llm/snapshot.sh",
    ".llm/create_patch.sh",
    ".llm/strategy.md",
}
_DECISION_ROW = re.compile(r"^\| `(?P<filename>\d{4}-[a-z0-9-]+\.md)` \|", re.MULTILINE)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _assert_markdown_format(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# "), f"{path} must start with a level-one heading"
    assert "\x00" not in text, f"{path} contains a NUL byte"


def test_package_imports() -> None:
    assert pipls.__version__ == "0.0.0"


def test_packaging_uses_pep639_license_metadata() -> None:
    pyproject = (_repository_root() / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires = ["setuptools>=77.0.3", "wheel"]' in pyproject
    assert 'license = "BSD-3-Clause"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert 'license = {file = "LICENSE"}' not in pyproject


def test_required_llm_contracts_exist_and_are_formatted() -> None:
    root = _repository_root()
    missing = sorted(path for path in _REQUIRED_LLM_CONTRACTS if not (root / path).is_file())
    assert not missing, f"Missing repository contracts: {missing}"

    for relative_path in sorted(_REQUIRED_LLM_CONTRACTS):
        path = root / relative_path
        if path.suffix == ".md":
            _assert_markdown_format(path)


def test_llm_layer_is_outside_installable_package() -> None:
    package_root = Path(pipls.__file__).resolve().parent
    assert ".llm" not in {part.name for part in package_root.parents}


def test_snapshot_has_repository_contents_at_archive_root(tmp_path: Path) -> None:
    root = _repository_root()
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
    root = _repository_root()
    scripts = [
        root / ".llm" / "snapshot.sh",
        root / ".llm" / "create_patch.sh",
    ]
    assert all(path.stat().st_mode & 0o111 for path in scripts)


def test_decision_index_and_records_are_structurally_consistent() -> None:
    root = _repository_root()
    index = (root / ".llm" / "decisions.md").read_text(encoding="utf-8")
    indexed_files = set(_DECISION_ROW.findall(index))
    decision_paths = sorted((root / "docs" / "decisions").glob("*.md"))
    shipped_files = {path.name for path in decision_paths}

    assert indexed_files == shipped_files
    assert decision_paths

    for path in decision_paths:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# Decision"), path
        assert "\x00" not in text, path


def test_llm_prompt_templates_are_nonempty_utf8_files() -> None:
    prompt_files = sorted((_repository_root() / ".llm" / "prompts").glob("*.md"))
    assert prompt_files
    for path in prompt_files:
        text = path.read_text(encoding="utf-8")
        assert text.strip()
        assert "\x00" not in text


def test_repository_datasets_are_not_top_level_runtime_exports() -> None:
    datasets_root = _repository_root() / "datasets"
    dataset_names = {path.name for path in datasets_root.iterdir() if path.is_dir()}
    assert dataset_names.isdisjoint(pipls.__all__)
