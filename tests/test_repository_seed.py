from __future__ import annotations

import re
import subprocess
import tarfile
from pathlib import Path

import yaml

import pipls

_REQUIRED_LLM_CONTRACTS = {
    ".llm/README.md",
    ".llm/project.md",
    ".llm/product_scope.md",
    ".llm/state.md",
    ".llm/decisions.md",
    ".llm/theory.md",
    ".llm/mathematics.md",
    ".llm/numerical_contracts.md",
    ".llm/development.md",
    ".llm/public_api.md",
    ".llm/data_io.md",
    ".llm/dataset_layout.md",
    ".llm/benchmarking.md",
    ".llm/analysis.md",
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
    controls = sorted(
        {ord(character) for character in text if ord(character) < 32 and character not in "\n\r"}
    )
    assert not controls, f"{path} contains ASCII control characters: {controls}"


def test_package_imports() -> None:
    assert pipls.__version__ == "0.0.0"


def test_packaging_uses_pep639_license_metadata() -> None:
    pyproject = (_repository_root() / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires = ["setuptools>=77.0.3", "wheel"]' in pyproject
    assert 'license = "BSD-3-Clause"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert 'license = {file = "LICENSE"}' not in pyproject


def test_source_distribution_manifest_includes_documentation_build_inputs() -> None:
    manifest_lines = {
        line.strip()
        for line in (_repository_root() / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    assert {
        "include Makefile",
        "include mkdocs.yml",
        "include tools/check_sdist_docs.py",
        "include tools/check_distributions.py",
        "include tools/render_synthetic_tutorial.py",
        "include tools/render_pulp_tutorial.py",
        "recursive-include docs *.md *.js",
    } <= manifest_lines


def test_docs_extra_declares_the_build_toolchain() -> None:
    pyproject = (_repository_root() / "pyproject.toml").read_text(encoding="utf-8")
    assert "docs = [" in pyproject
    assert '"build>=1.2,<2"' in pyproject
    assert '"mkdocs>=1.6,<2"' in pyproject
    assert '"mkdocs-material>=9.5,<9.7"' in pyproject
    assert '"mkdocstrings-python>=2,<3"' in pyproject
    assert '"ruff>=0.6"' in pyproject.split("docs = [", 1)[1].split("]", 1)[0]


def test_mkdocs_configuration_has_valid_user_navigation() -> None:
    root = _repository_root()
    config = yaml.safe_load((root / "mkdocs.yml").read_text(encoding="utf-8"))

    def targets(items: list[object]) -> set[str]:
        found: set[str] = set()
        for item in items:
            if isinstance(item, str):
                found.add(item)
                continue
            assert isinstance(item, dict)
            for value in item.values():
                if isinstance(value, str):
                    found.add(value)
                else:
                    assert isinstance(value, list)
                    found.update(targets(value))
        return found

    nav_targets = targets(config["nav"])
    public_pages = {
        path.relative_to(root / "docs").as_posix()
        for path in (root / "docs").rglob("*.md")
        if "decisions" not in path.relative_to(root / "docs").parts
    }

    assert config["strict"] is True
    assert config["theme"]["name"] == "material"
    assert nav_targets == public_pages
    assert all((root / "docs" / target).is_file() for target in nav_targets)
    assert config["exclude_docs"].splitlines() == ["decisions/**"]
    assert "not_in_nav" not in config
    assert not any(target.startswith("decisions/") for target in nav_targets)
    assert "javascripts/mathjax.js" in config["extra_javascript"]
    mkdocstrings = next(
        plugin["mkdocstrings"]
        for plugin in config["plugins"]
        if isinstance(plugin, dict) and "mkdocstrings" in plugin
    )
    python_handler = mkdocstrings["handlers"]["python"]
    assert python_handler["paths"] == ["src"]
    assert python_handler["options"]["docstring_style"] == "numpy"
    assert python_handler["options"]["show_source"] is False


def test_make_docs_is_strict_and_generated_site_is_ignored() -> None:
    root = _repository_root()
    completed = subprocess.run(
        ["make", "-n", "docs"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    clean = subprocess.run(
        ["make", "-n", "clean"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "-m mkdocs build --strict" in completed.stdout
    assert "site" in clean.stdout.split()
    assert "site/" in (root / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_make_help_and_documentation_preview_are_discoverable() -> None:
    root = _repository_root()
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    help_output = subprocess.run(
        ["make", "help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    default_output = subprocess.run(
        ["make"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    preview = subprocess.run(
        ["make", "-n", "docs-serve"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    public_targets = {
        "help",
        "install",
        "test",
        "lint",
        "format",
        "typecheck",
        "docs",
        "docs-serve",
        "docs-dist",
        "build",
        "dist-check",
        "check",
        "examples",
        "snapshot",
        "clean",
    }

    assert ".DEFAULT_GOAL := help" in makefile
    assert "Usage: make <target>" in help_output
    assert "Usage: make <target>" in default_output
    assert public_targets <= set(re.findall(r"^([A-Za-z0-9_.-]+):.*## .+$", makefile, re.MULTILINE))
    assert all(target in help_output for target in public_targets)
    assert "-m mkdocs serve --dev-addr=127.0.0.1:8000" in preview
    assert "http://127.0.0.1:8000/" in preview


def test_documentation_distribution_target_uses_the_validation_helper() -> None:
    root = _repository_root()
    completed = subprocess.run(
        ["make", "-n", "docs-dist"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "tools/check_sdist_docs.py" in completed.stdout
    helper = (root / "tools" / "check_sdist_docs.py").read_text(encoding="utf-8")
    assert '"--sdist"' in helper
    assert 'f"{source}[docs]"' in helper
    assert '"docs"' in helper
    assert "PYTHON={python}" in helper


def test_documentation_ci_builds_checkout_and_source_distribution() -> None:
    workflow = yaml.safe_load(
        (_repository_root() / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
    )
    docs_steps = workflow["jobs"]["docs"]["steps"]
    commands = [step["run"] for step in docs_steps if "run" in step]

    assert 'python -m pip install -e ".[docs]"' in commands
    assert "make docs" in commands
    assert "make docs-dist" in commands


def test_examples_extra_declares_data_and_plotting_dependencies() -> None:
    pyproject = (_repository_root() / "pyproject.toml").read_text(encoding="utf-8")

    assert 'examples = ["pandas>=2.0", "matplotlib>=3.8"]' in pyproject
    assert '"pandas>=2.0"' in pyproject.split("dev = [", 1)[1].split("]", 1)[0]
    assert '"matplotlib>=3.8"' in pyproject.split("dev = [", 1)[1].split("]", 1)[0]


def test_make_examples_runs_every_numbered_example() -> None:
    root = _repository_root()
    completed = subprocess.run(
        ["make", "-n", "examples"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    expected = [
        "examples/01_minimal_fit_and_plot.py",
        "examples/02_synthetic_path_selection.py",
        "examples/09_pls_path_comparison.py",
        "examples/10_pulp_real_data.py",
        "examples/11_sugarcane_real_data.py",
        "examples/12_tobacco_real_data.py",
    ]
    positions = [completed.stdout.index(filename) for filename in expected]

    assert positions == sorted(positions)
    assert completed.stdout.count("MPLBACKEND=Agg") == 1


def test_required_llm_contracts_exist_and_are_formatted() -> None:
    root = _repository_root()
    missing = sorted(path for path in _REQUIRED_LLM_CONTRACTS if not (root / path).is_file())
    assert not missing, f"Missing repository contracts: {missing}"

    for relative_path in sorted(_REQUIRED_LLM_CONTRACTS):
        path = root / relative_path
        if path.suffix == ".md":
            _assert_markdown_format(path)


def test_public_markdown_has_no_ascii_control_characters() -> None:
    root = _repository_root()
    markdown_paths = sorted(
        path
        for directory in (root / "docs", root / "examples", root / "benchmarks", root / "datasets")
        for path in directory.rglob("*.md")
    )
    markdown_paths.extend([root / "README.md", root / "CONTRIBUTING.md", root / "CHANGELOG.md"])

    for markdown_path in markdown_paths:
        _assert_markdown_format(markdown_path)


def test_public_documentation_is_self_contained() -> None:
    root = _repository_root()
    docs_root = (root / "docs").resolve()
    markdown_target = re.compile(r"\]\((?P<target>[^)]+)\)")
    abandoned_selection_terms = (
        "one-standard-error",
        "one-standard-deviation",
        "1sd",
    )

    for path in sorted((root / "docs").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for match in markdown_target.finditer(text):
            target = match.group("target").strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative_path = target.split("#", 1)[0]
            resolved = (path.parent / relative_path).resolve()
            assert resolved.is_relative_to(docs_root), f"{path} links outside docs/: {target}"
        lowered = text.lower()
        assert not any(term in lowered for term in abandoned_selection_terms), path

    theory = (root / "docs" / "theory.md").read_text(encoding="utf-8")
    assert ".llm" not in theory


def test_maintainer_decision_index_links_every_record() -> None:
    root = _repository_root()
    index = (root / "docs" / "decisions" / "index.md").read_text(encoding="utf-8")
    linked_files = set(re.findall(r"\((\d{4}-[a-z0-9-]+\.md)\)", index))
    decision_files = {
        path.name for path in (root / "docs" / "decisions").glob("[0-9][0-9][0-9][0-9]-*.md")
    }

    assert linked_files == decision_files


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
        members = handle.getmembers()
        names = {member.name.removeprefix("./") for member in members}

    assert "README.md" in names
    assert ".llm/SNAPSHOT_INFO" in names
    assert not any(name.startswith(f"{root.name}/") for name in names)
    assert not any(name == "site" or name.startswith("site/") for name in names)
    expected_result_placeholders = {
        "examples/results/.gitkeep",
        "examples/results/pls_path_comparison/.gitkeep",
        "examples/results/synthetic_tutorial/.gitkeep",
        "examples/results/pulp_post_analysis/.gitkeep",
        "examples/results/sugarcane_post_analysis/.gitkeep",
        "examples/results/tobacco_post_analysis/.gitkeep",
    }
    archived_results = {
        member.name.removeprefix("./")
        for member in members
        if member.isfile() and member.name.removeprefix("./").startswith("examples/results/")
    }
    assert archived_results == expected_result_placeholders


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
    decision_paths = sorted((root / "docs" / "decisions").glob("[0-9][0-9][0-9][0-9]-*.md"))
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


def test_repository_has_no_paper_reproduction_scaffolding() -> None:
    root = _repository_root()
    retired_paths = [
        root / "paper",
        root / "scripts" / "reproduce_paper",
    ]
    assert not any(path.exists() for path in retired_paths)


def test_retired_linnerud_integration_is_absent() -> None:
    root = _repository_root()
    retired_paths = [
        root / "datasets" / "linnerud",
        root / "examples" / "09_linnerud_real_data.py",
        root / "tests" / "data" / "test_linnerud_dataset.py",
    ]
    assert not any(path.exists() for path in retired_paths)
