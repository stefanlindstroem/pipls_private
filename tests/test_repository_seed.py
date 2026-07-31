from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

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
    ".llm/analysis.md",
    ".llm/testing.md",
    ".llm/snapshot.sh",
    ".llm/create_patch.sh",
    ".llm/strategy.md",
}
_DECISION_ROW = re.compile(r"^\| `(?P<filename>\d{4}-[a-z0-9-]+\.md)` \|", re.MULTILINE)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_metadata() -> dict[str, Any]:
    with (_repository_root() / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


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


def test_named_authors_and_complete_bsd_license() -> None:
    root = _repository_root()
    project = _project_metadata()
    authors = [author["name"] for author in project["authors"]]
    license_text = (root / "LICENSE").read_text(encoding="utf-8")

    assert authors == [
        "Vishal Agrawal",
        "Fritjof Nilsson",
        "Stefan B. Lindström",
    ]
    assert (
        "Copyright (c) 2026 Vishal Agrawal, Fritjof Nilsson, and Stefan B. Lindström"
        in license_text
    )
    assert "Pi-PLS authors" not in license_text
    assert "Redistribution and use in source and binary forms, with or without modification" in (
        license_text
    )
    assert "IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE" in license_text
    assert "LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION" in license_text


def test_citation_metadata_names_software_and_companion_manuscript() -> None:
    root = _repository_root()
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    project = _project_metadata()

    expected_authors = [
        {"family-names": "Agrawal", "given-names": "Vishal"},
        {"family-names": "Nilsson", "given-names": "Fritjof"},
        {"family-names": "Lindström", "given-names": "Stefan B."},
    ]
    assert citation["cff-version"] == "1.2.0"
    assert citation["type"] == "software"
    assert citation["license"] == "BSD-3-Clause"
    assert citation["version"] == project["version"]
    assert citation["authors"] == expected_authors

    paper = citation["preferred-citation"]
    assert paper["type"] == "article"
    assert paper["title"] == (
        "Panoramic Partial Least Squares (Pi-PLS): Transparent, parsimonious, and more "
        "interpretable multivariate regression model"
    )
    assert paper["authors"] == expected_authors
    assert paper["journal"] == "Computers & Chemical Engineering"
    assert paper["status"] == "submitted"
    assert paper["year"] == 2026
    assert paper["notes"] == "Manuscript under revision, CACE-D-26-00847."


def test_source_distribution_manifest_includes_documentation_build_inputs() -> None:
    manifest_lines = {
        line.strip()
        for line in (_repository_root() / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    assert {
        "include CITATION.cff",
        "include Makefile",
        "include mkdocs.yml",
        "include tools/check_sdist_docs.py",
        "include tools/check_distributions.py",
        "include tools/render_synthetic_tutorial.py",
        "include tools/render_pulp_tutorial.py",
        "recursive-include docs *.md *.js",
        "recursive-include examples/results .gitkeep",
    } <= manifest_lines


def test_optional_dependency_groups_match_maintained_workflows() -> None:
    project = _project_metadata()
    runtime = set(project["dependencies"])
    extras = project["optional-dependencies"]

    assert set(extras) == {"dev", "examples", "docs"}
    assert set(extras["examples"]) <= set(extras["dev"])
    assert not any(requirement.startswith("pytest-cov") for requirement in extras["dev"])
    assert not any(requirement.startswith("ruff") for requirement in extras["docs"])
    assert not any(
        requirement.startswith(("pandas", "matplotlib", "adjustText")) for requirement in runtime
    )

    docs = set(extras["docs"])
    assert {
        "build>=1.2,<2",
        "pandas>=2.0",
        "matplotlib>=3.8",
        "adjustText>=1.4,<2",
        "mkdocs>=1.6,<2",
        "mkdocs-material>=9.5,<9.7",
        "mkdocstrings-python>=2,<3",
    } <= docs


def test_public_installation_is_noneditable_and_contributor_setup_is_editable() -> None:
    root = _repository_root()
    readme = (root / "README.md").read_text(encoding="utf-8")
    contributing = (root / "CONTRIBUTING.md").read_text(encoding="utf-8")
    served_docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / "docs").rglob("*.md"))
        if "decisions" not in path.parts
    )

    assert "python -m pip install ." in readme
    assert 'python -m pip install ".[examples]"' in readme
    assert "pip install -e" not in readme
    assert "pip install -e" not in served_docs
    assert 'python -m pip install -e ".[dev,docs]"' in contributing
    assert "make install" in contributing


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

    target_groups = {
        "Start here": ("help", "install", "check"),
        "Development": ("test", "lint", "format", "typecheck", "clean"),
        "Documentation and examples": (
            "examples",
            "docs",
            "docs-serve",
            "docs-figures",
            "docs-dist",
        ),
        "Distribution and maintenance": ("build", "dist-check", "snapshot"),
    }
    public_targets = {target for targets in target_groups.values() for target in targets}

    assert ".DEFAULT_GOAL := help" in makefile
    assert "Usage: make <target>" in help_output
    assert "Usage: make <target>" in default_output
    assert re.search(r"^First setup:\s+make install$", help_output, re.MULTILINE)
    assert re.search(r"^Routine validation:\s+make check$", help_output, re.MULTILINE)
    assert public_targets == set(re.findall(r"^([A-Za-z0-9_.-]+):.*## .+$", makefile, re.MULTILINE))

    section_positions = [help_output.index(f"{section}:") for section in target_groups]
    assert section_positions == sorted(section_positions)
    for index, targets in enumerate(target_groups.values()):
        start = section_positions[index]
        end = section_positions[index + 1] if index + 1 < len(section_positions) else None
        block = help_output[start:end]
        for target in targets:
            assert re.search(rf"^  {re.escape(target)}\s+\S", block, re.MULTILINE)

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
    root = _repository_root()
    workflow_path = root / ".github" / "workflows" / "documentation.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    docs_steps = workflow["jobs"]["build"]["steps"]
    commands = [step["run"] for step in docs_steps if "run" in step]

    assert 'python -m pip install -e ".[docs]"' in commands
    assert "make docs" in commands
    assert "make docs-dist" in commands
    assert (
        "docs"
        not in yaml.safe_load(
            (root / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
        )["jobs"]
    )


def test_documentation_ci_deploys_only_the_master_pages_site(tmp_path: Path) -> None:
    root = _repository_root()
    workflow_path = root / ".github" / "workflows" / "documentation.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)
    build_steps = workflow["jobs"]["build"]["steps"]
    deploy = workflow["jobs"]["deploy"]

    assert "actions/configure-pages@v6" in workflow_text
    assert "actions/upload-pages-artifact@v5" in workflow_text
    assert "actions/deploy-pages@v5" in workflow_text
    assert deploy["if"] == "github.event_name == 'push' && github.ref == 'refs/heads/master'"
    assert deploy["environment"] == {
        "name": "github-pages",
        "url": "${{ steps.deployment.outputs.page_url }}",
    }
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}
    assert deploy["concurrency"] == {
        "group": "github-pages",
        "cancel-in-progress": False,
    }
    conditional_steps = [step for step in build_steps if "if" in step]
    assert conditional_steps
    assert {step["if"] for step in conditional_steps} == {
        "github.event_name == 'push' && github.ref == 'refs/heads/master'"
    }

    output = tmp_path / "mkdocs-pages.yml"
    subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "configure_pages_docs.py"),
            "--repository",
            "example/pipls",
            "--server-url",
            "https://github.com",
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
    )
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["site_url"] == "https://example.github.io/pipls/"
    assert config["repo_url"] == "https://github.com/example/pipls"
    assert config["edit_uri"] == "edit/master/docs/"


def test_examples_extra_declares_data_and_rendering_dependencies() -> None:
    project = _project_metadata()
    runtime = project["dependencies"]
    extras = project["optional-dependencies"]

    assert extras["examples"] == [
        "pandas>=2.0",
        "matplotlib>=3.8",
        "adjustText>=1.4,<2",
    ]
    assert set(extras["examples"]) <= set(extras["dev"])
    assert not any(requirement.startswith(("matplotlib", "adjustText")) for requirement in runtime)
    assert "plot" not in extras


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
        f"examples/{number:02d}_{name}.py"
        for number, name in enumerate(
            (
                "minimal_fit_and_plot",
                "synthetic_path_selection",
                "leave_one_out_validation",
                "pls_path_comparison",
                "pulp_real_data",
                "sugarcane_real_data",
                "tobacco_real_data",
            ),
            start=1,
        )
    ]
    actual = [
        path.relative_to(root).as_posix()
        for path in sorted((root / "examples").glob("[0-9][0-9]_*.py"))
    ]
    positions = [completed.stdout.index(filename) for filename in expected]

    assert actual == expected
    assert positions == sorted(positions)
    assert completed.stdout.count("MPLBACKEND=Agg") == 1


def test_llm_project_maps_every_runtime_module() -> None:
    root = _repository_root()
    project = (root / ".llm" / "project.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"`(src/pipls/[A-Za-z0-9_]+\.py)`", project))
    runtime_modules = {
        path.relative_to(root).as_posix() for path in (root / "src" / "pipls").glob("*.py")
    }

    assert documented == runtime_modules


def test_patch_request_template_has_downloadable_checksum_handoff() -> None:
    template = (_repository_root() / ".llm" / "templates" / "PATCH_REQUEST.md").read_text(
        encoding="utf-8"
    )

    assert "downloadable unified Git patch" in template
    assert "SHA-256" in template
    for command in (
        "git apply",
        "make check",
        "git add -A",
        "git commit",
        "make snapshot",
    ):
        assert command in template


def test_required_llm_contracts_exist_and_are_formatted() -> None:
    root = _repository_root()
    missing = sorted(path for path in _REQUIRED_LLM_CONTRACTS if not (root / path).is_file())
    assert not missing, f"Missing repository contracts: {missing}"

    for relative_path in sorted(_REQUIRED_LLM_CONTRACTS):
        path = root / relative_path
        if path.suffix == ".md":
            _assert_markdown_format(path)


def test_readme_and_contributing_have_distinct_audiences() -> None:
    root = _repository_root()
    readme = (root / "README.md").read_text(encoding="utf-8")
    contributing = (root / "CONTRIBUTING.md").read_text(encoding="utf-8")

    for public_workflow in (
        "PiPLSRegression",
        "PiPLSSearchCV",
        "for_n_components",
        "../../deployments/github-pages",
        "docs/tutorials/synthetic.md",
        "docs/tutorials/pulp.md",
    ):
        assert public_workflow in readme

    for maintainer_command in ("make docs-dist", "make dist-check", "make snapshot"):
        assert maintainer_command not in readme
        assert maintainer_command in contributing

    repository_paths = ("src/pipls/", "constraints/", ".llm/")
    assert not any(repository_path in readme for repository_path in repository_paths)
    assert all(repository_path in contributing for repository_path in repository_paths)


def test_public_markdown_has_no_ascii_control_characters() -> None:
    root = _repository_root()
    markdown_paths = sorted(
        path
        for directory in (root / "docs", root / "examples", root / "datasets")
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
    linked_entries = re.findall(r"\((\d{4}-[a-z0-9-]+\.md)\)", index)
    linked_files = set(linked_entries)
    decision_files = {
        path.name for path in (root / "docs" / "decisions").glob("[0-9][0-9][0-9][0-9]-*.md")
    }

    assert len(linked_entries) == len(linked_files), "decision records must be indexed exactly once"
    assert linked_files == decision_files


def test_llm_layer_is_outside_installable_package() -> None:
    package_root = Path(pipls.__file__).resolve().parent
    assert ".llm" not in {part.name for part in package_root.parents}


def _create_snapshot_test_repository(path: Path) -> Path:
    root = path / "repository"
    (root / ".llm").mkdir(parents=True)
    shutil.copy2(_repository_root() / ".llm" / "snapshot.sh", root / ".llm" / "snapshot.sh")
    (root / "README.md").write_text("# Snapshot fixture\n", encoding="utf-8")
    (root / ".gitignore").write_text(
        "docs/assets/generated/\n.pytest_cache/\n*-snapshot.tar.gz\n",
        encoding="utf-8",
    )

    result_directories = (
        "examples/results",
        "examples/results/pls_path_comparison",
        "examples/results/synthetic_tutorial",
        "examples/results/pulp_post_analysis",
        "examples/results/sugarcane_post_analysis",
        "examples/results/tobacco_post_analysis",
    )
    for relative in result_directories:
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch()

    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "snapshot@example.invalid"], cwd=root, check=True
    )
    subprocess.run(["git", "config", "user.name", "Snapshot Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "Create snapshot fixture"],
        cwd=root,
        check=True,
    )
    return root


def test_snapshot_has_committed_repository_contents_at_archive_root(tmp_path: Path) -> None:
    root = _create_snapshot_test_repository(tmp_path)
    archive = root / "fixture-snapshot.tar.gz"

    subprocess.run(
        [str(root / ".llm" / "snapshot.sh"), archive.name],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        names = {member.name.removeprefix("./") for member in members}
        metadata_member = handle.extractfile(".llm/SNAPSHOT_INFO")
        assert metadata_member is not None
        metadata = metadata_member.read().decode("utf-8")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert "README.md" in names
    assert ".llm/SNAPSHOT_INFO" in names
    assert not any(name.startswith(f"{root.name}/") for name in names)
    assert f"commit: {commit}" in metadata
    assert "dirty: false" in metadata
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


def test_example_result_tree_contains_placeholders_and_ignored_outputs() -> None:
    repository_root = _repository_root()
    result_root = repository_root / "examples" / "results"
    expected_placeholders = {
        ".gitkeep",
        "pls_path_comparison/.gitkeep",
        "pulp_post_analysis/.gitkeep",
        "sugarcane_post_analysis/.gitkeep",
        "synthetic_tutorial/.gitkeep",
        "tobacco_post_analysis/.gitkeep",
    }
    result_files = {
        path.relative_to(result_root).as_posix()
        for path in result_root.rglob("*")
        if path.is_file()
    }

    assert expected_placeholders <= result_files

    generated_paths = sorted(
        f"examples/results/{path}" for path in result_files - expected_placeholders
    )
    if generated_paths:
        ignored = subprocess.run(
            ["git", "check-ignore", "--no-index", "--stdin"],
            cwd=repository_root,
            input="\n".join(generated_paths),
            check=False,
            capture_output=True,
            text=True,
        )
        assert set(ignored.stdout.splitlines()) == set(generated_paths)


def test_snapshot_refuses_modified_staged_and_untracked_files(tmp_path: Path) -> None:
    for state in ("modified", "staged", "untracked"):
        root = _create_snapshot_test_repository(tmp_path / state)
        if state == "untracked":
            (root / "notes.txt").write_text("untracked\n", encoding="utf-8")
        else:
            (root / "README.md").write_text(f"# {state}\n", encoding="utf-8")
            if state == "staged":
                subprocess.run(["git", "add", "README.md"], cwd=root, check=True)

        archive = tmp_path / f"{state}.tar.gz"
        completed = subprocess.run(
            [str(root / ".llm" / "snapshot.sh"), str(archive)],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode != 0
        assert "Refusing to create a snapshot from a dirty worktree" in completed.stderr
        assert not archive.exists()


def test_snapshot_refuses_committed_generated_example_outputs(tmp_path: Path) -> None:
    root = _create_snapshot_test_repository(tmp_path)
    generated = root / "examples" / "results" / "generated.pdf"
    generated.write_bytes(b"%PDF-generated fixture\n")
    subprocess.run(["git", "add", "examples/results/generated.pdf"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "Commit generated output"],
        cwd=root,
        check=True,
    )
    archive = tmp_path / "snapshot.tar.gz"

    completed = subprocess.run(
        [str(root / ".llm" / "snapshot.sh"), str(archive)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "committed generated example outputs" in completed.stderr
    assert "examples/results/generated.pdf" in completed.stderr
    assert not archive.exists()


def test_snapshot_ignores_ignored_generated_files(tmp_path: Path) -> None:
    root = _create_snapshot_test_repository(tmp_path)
    generated = root / "docs" / "assets" / "generated" / "tutorial.svg"
    generated.parent.mkdir(parents=True)
    generated.write_text("<svg/>\n", encoding="utf-8")
    cache = root / ".pytest_cache" / "state"
    cache.parent.mkdir()
    cache.write_text("cache\n", encoding="utf-8")
    archive = tmp_path / "snapshot.tar.gz"

    subprocess.run(
        [str(root / ".llm" / "snapshot.sh"), str(archive)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    with tarfile.open(archive, "r:gz") as handle:
        names = {member.name.removeprefix("./") for member in handle.getmembers()}
    assert not any(name.startswith("docs/assets/generated/") for name in names)
    assert not any(name.startswith(".pytest_cache/") for name in names)


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
