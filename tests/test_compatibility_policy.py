from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib

import yaml

_SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
_RUNTIME_DEPENDENCIES = {
    "numpy": "numpy>=1.26,<3",
    "scikit-learn": "scikit-learn>=1.4,<2",
    "joblib": "joblib>=1.2,<2",
}
_MINIMUM_CONSTRAINTS = {
    "numpy": "numpy==1.26.*",
    "scikit-learn": "scikit-learn==1.4.*",
    "joblib": "joblib==1.2.*",
}
_DOCUMENTED_RANGES = {
    "numpy": ">=1.26,<3",
    "scikit-learn": ">=1.4,<2",
    "joblib": ">=1.2,<2",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_metadata() -> dict[str, object]:
    with (_repository_root() / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def test_project_metadata_declares_the_supported_compatibility_range() -> None:
    project = _project_metadata()

    assert project["requires-python"] == ">=3.10"
    assert set(project["dependencies"]) == set(_RUNTIME_DEPENDENCIES.values())

    classifiers = set(project["classifiers"])
    assert "Programming Language :: Python :: 3" in classifiers
    assert "Programming Language :: Python :: 3 :: Only" in classifiers
    assert {
        f"Programming Language :: Python :: {version}" for version in _SUPPORTED_PYTHONS
    } <= classifiers


def test_minimum_constraints_match_the_declared_lower_dependency_lines() -> None:
    constraints = {
        line.split("==", 1)[0]: line
        for line in (_repository_root() / "constraints" / "minimum.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    }

    assert constraints == _MINIMUM_CONSTRAINTS
    assert set(constraints) == set(_RUNTIME_DEPENDENCIES)
    for name, constraint in constraints.items():
        minimum_line = constraint.removesuffix(".*").replace("==", ">=", 1)
        assert _RUNTIME_DEPENDENCIES[name].startswith(minimum_line)


def test_compatibility_documentation_matches_metadata_and_constraints() -> None:
    text = (_repository_root() / "docs" / "compatibility.md").read_text(encoding="utf-8")

    assert "Python 3.10 through 3.14" in text
    assert '`requires-python = ">=3.10"`' in text
    for supported_range in _DOCUMENTED_RANGES.values():
        assert f"`{supported_range}`" in text
    for constraint in _MINIMUM_CONSTRAINTS.values():
        assert constraint in text


def _test_workflow_jobs() -> dict[str, object]:
    workflow = yaml.safe_load(
        (_repository_root() / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    )
    return workflow["jobs"]


def _setup_python_version(steps: list[dict[str, object]]) -> str:
    setup = next(step for step in steps if step.get("uses") == "actions/setup-python@v5")
    return setup["with"]["python-version"]


def _run_commands(steps: list[dict[str, object]]) -> list[str]:
    return [step["run"] for step in steps if "run" in step]


def test_test_workflow_separates_the_three_compatibility_responsibilities() -> None:
    jobs = _test_workflow_jobs()

    assert {
        "minimum-dependencies",
        "supported-python",
        "latest-dependencies",
    } <= set(jobs)

    minimum = jobs["minimum-dependencies"]
    minimum_steps = minimum["steps"]
    assert "strategy" not in minimum
    assert _setup_python_version(minimum_steps) == "3.10"
    assert 'python -m pip install -c constraints/minimum.txt -e ".[dev]"' in _run_commands(
        minimum_steps
    )

    supported = jobs["supported-python"]
    assert supported["strategy"]["fail-fast"] is False
    assert tuple(supported["strategy"]["matrix"]["python-version"]) == _SUPPORTED_PYTHONS
    assert 'python -m pip install -e ".[dev]"' in _run_commands(supported["steps"])

    latest = jobs["latest-dependencies"]
    latest_steps = latest["steps"]
    assert "strategy" not in latest
    assert _setup_python_version(latest_steps) == "3.14"
    assert (
        'python -m pip install --upgrade -e ".[dev]" "numpy<3" "scikit-learn<2" "joblib<2"'
        in _run_commands(latest_steps)
    )


def test_every_compatibility_job_prints_resolved_versions() -> None:
    jobs = _test_workflow_jobs()

    for job_name in ("minimum-dependencies", "supported-python", "latest-dependencies"):
        version_step = next(
            step for step in jobs[job_name]["steps"] if step.get("name") == "Show resolved versions"
        )
        command = version_step["run"]
        assert "sys.version.split()[0]" in command
        assert "numpy.__version__" in command
        assert "sklearn.__version__" in command
        assert "joblib.__version__" in command


def test_minimum_constraints_are_included_in_source_distributions() -> None:
    manifest_lines = (_repository_root() / "MANIFEST.in").read_text(encoding="utf-8").splitlines()

    assert "include constraints/minimum.txt" in manifest_lines
