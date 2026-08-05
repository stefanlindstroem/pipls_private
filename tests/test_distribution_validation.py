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
    assert not any(line.startswith("recursive-include datasets ") for line in manifest)
    assert "include CITATION.cff" in manifest
    assert "include tools/check_distributions.py" in manifest
    assert "recursive-include src/pipls/_data/pulp *.csv *.json *.md *.txt" in manifest
    assert (
        "recursive-include src/pipls/_data/sugarcane *.csv *.json *.md *.txt"
        in manifest
    )
    assert (
        "recursive-include src/pipls/_data/tobacco *.csv *.json *.md *.txt"
        in manifest
    )
    assert "recursive-include examples/results .gitkeep" in manifest
    assert "[tool.setuptools.package-data]" in pyproject
    for pattern in (
        '"_data/pulp/*.csv"',
        '"_data/pulp/*.json"',
        '"_data/pulp/*.md"',
        '"_data/pulp/*.txt"',
        '"_data/sugarcane/*.csv"',
        '"_data/sugarcane/*.json"',
        '"_data/sugarcane/*.md"',
        '"_data/sugarcane/*.txt"',
        '"_data/tobacco/*.csv"',
        '"_data/tobacco/*.json"',
        '"_data/tobacco/*.md"',
        '"_data/tobacco/*.txt"',
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
    assert "_assert_reference_resources_included(wheel)" in helper
    assert "_assert_reference_resources_included(source_distribution)" in helper
    assert "_assert_documentation_sources_included(source_distribution)" in helper
    assert '_COMPUTATIONAL_PERFORMANCE_GUIDE = "docs/computational_performance.md"' in helper
    assert "_REFERENCE_DATASETS = (\"pulp\", \"sugarcane\", \"tobacco\")" in helper
    assert "_REFERENCE_RESOURCE_FILES" in helper
    assert "_check_source_distribution_example" in helper
    assert 'f"pipls[examples] @ {artifact.resolve().as_uri()}"' in helper
    assert 'source / "examples" / "01_pulp_quick_start.py"' in helper
    assert 'source / "examples" / "results" / "pulp_quick_start.pdf"' in helper


def test_source_distribution_docs_include_and_render_performance_guide() -> None:
    helper = (_repository_root() / "tools" / "check_sdist_docs.py").read_text(
        encoding="utf-8"
    )

    assert 'source / "docs" / "computational_performance.md"' in helper
    assert (
        'source / "site" / "computational_performance" / "index.html"'
        in helper
    )


def test_distribution_smoke_test_covers_public_installed_behavior() -> None:
    helper = (_repository_root() / "tools" / "check_distributions.py").read_text(encoding="utf-8")

    for public_import in (
        "import pipls.component_path",
        "import pipls.datasets",
        "import pipls.inspection",
        "import pipls.metrics",
        "import pipls.search",
        "from pipls import PiPLSRegression, PiPLSSearchCV",
        "from pipls.component_path import PiPLSPredictorRankEvidence",
        "PiPLSDataset,",
        "PiPLSLatentGeometryTruth,",
        "PiPLSRegressionTruth,",
        "make_pipls_latent_geometry,",
        "make_pipls_regression,",
        "make_pipls_train_test,",
    ):
        assert public_import in helper

    assert 'version("pipls") == pipls.__version__' in helper
    assert 'item.__module__ == "pipls.datasets"' in helper
    assert "isinstance(synthetic.truth, PiPLSRegressionTruth)" in helper
    assert "package_file.relative_to(source_root)" in helper
    assert "package_file.relative_to(environment_root)" in helper
    assert "PiPLSRegression(n_components=1, predictor_rank=2).fit(X, Y)" in helper
    assert "prediction.shape == (2, 2)" in helper
    assert "predictor_rank_relative_tolerance=1e6" in helper
    assert 'search_method="exhaustive"' in helper
    assert 'search.search_method == "exhaustive"' in helper
    assert "selection = search.select(n_components=1)" in helper
    assert "isinstance(evidence, PiPLSPredictorRankEvidence)" in helper
    assert "profile.selection == selection" in helper
    assert "pulp = load_pulp()" in helper
    assert "pulp_X, pulp_Y = load_pulp(return_X_y=True)" in helper
    assert "pulp.X.shape == (46, 14)" in helper
    assert "pulp.Y.shape == (46, 8)" in helper
    assert "sugarcane = load_sugarcane()" in helper
    assert "sugarcane_X, sugarcane_Y = load_sugarcane(return_X_y=True)" in helper
    assert "sugarcane.X.shape == (57, 1721)" in helper
    assert "sugarcane.Y.shape == (57, 4)" in helper
    assert "tobacco = load_tobacco()" in helper
    assert "tobacco_X, tobacco_Y = load_tobacco(return_X_y=True)" in helper
    assert "tobacco.X.shape == (347, 1557)" in helper
    assert "tobacco.Y.shape == (347, 13)" in helper


def test_build_workflow_validates_installed_distributions() -> None:
    workflow = yaml.safe_load(
        (_repository_root() / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
    )
    distribution_steps = workflow["jobs"]["build"]["steps"]
    commands = [step["run"] for step in distribution_steps if "run" in step]

    assert "python -m pip install --upgrade pip build" in commands
    assert "make dist-check" in commands
    assert "make build" not in commands
