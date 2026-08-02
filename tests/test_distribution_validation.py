from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_distribution_check_is_a_public_make_target_and_sdist_input() -> None:
    root = _repository_root()
    dry_run = subprocess.run(
        ["make", "-n", "dist-check"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    help_output = subprocess.run(
        ["make", "help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert "tools/check_distributions.py" in dry_run.stdout
    assert "dist-check" in help_output
    assert "prune .llm" in manifest
    assert "include CITATION.cff" in manifest
    assert "include tools/check_distributions.py" in manifest
    assert "recursive-include src/pipls/_data/pulp *.csv *.json *.md *.txt" in manifest
    assert "recursive-include examples/results .gitkeep" in manifest
    assert "[tool.setuptools.package-data]" in pyproject
    for pattern in (
        '"_data/pulp/*.csv"',
        '"_data/pulp/*.json"',
        '"_data/pulp/*.md"',
        '"_data/pulp/*.txt"',
    ):
        assert pattern in pyproject


def test_distribution_helper_builds_once_and_checks_both_artifacts() -> None:
    helper = (_repository_root() / "tools" / "check_distributions.py").read_text(encoding="utf-8")

    assert helper.count('"-m", "build"') == 1
    assert '_single_artifact(artifacts, "*.whl", "wheel")' in helper
    assert 'artifacts, "*.tar.gz", "source distribution"' in helper
    assert '("wheel", wheel, False)' in helper
    assert '("sdist", source_distribution, True)' in helper
    assert "cwd=run_directory" in helper
    assert 'environment.pop("PYTHONPATH", None)' in helper
    assert "_assert_development_archive_excluded(wheel)" in helper
    assert "_assert_development_archive_excluded(source_distribution)" in helper
    assert "_check_source_distribution_example" in helper
    assert 'f"pipls[examples] @ {artifact.resolve().as_uri()}"' in helper
    assert 'source / "examples" / "01_pulp_quick_start.py"' in helper
    assert 'source / "examples" / "results" / "pulp_quick_start.pdf"' in helper


def test_distribution_smoke_test_covers_public_installed_behavior() -> None:
    helper = (_repository_root() / "tools" / "check_distributions.py").read_text(encoding="utf-8")

    for public_import in (
        "import pipls.datasets",
        "import pipls.inspection",
        "import pipls.metrics",
        "import pipls.search",
        "from pipls import PiPLSRegression, PiPLSSearchCV",
        "from pipls.datasets import load_pulp",
    ):
        assert public_import in helper

    assert 'version("pipls") == pipls.__version__' in helper
    assert "package_file.relative_to(source_root)" in helper
    assert "package_file.relative_to(environment_root)" in helper
    assert "PiPLSRegression(n_components=1, predictor_rank=2).fit(X, Y)" in helper
    assert "prediction.shape == (2, 2)" in helper
    assert "pulp = load_pulp()" in helper
    assert "pulp_X, pulp_Y = load_pulp(return_X_y=True)" in helper
    assert "pulp.data.shape == (46, 14)" in helper
    assert "pulp.target.shape == (46, 8)" in helper


def test_build_workflow_validates_installed_distributions() -> None:
    workflow = yaml.safe_load(
        (_repository_root() / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
    )
    distribution_steps = workflow["jobs"]["build"]["steps"]
    commands = [step["run"] for step in distribution_steps if "run" in step]

    assert "python -m pip install --upgrade pip build" in commands
    assert "make dist-check" in commands
    assert "make build" not in commands
