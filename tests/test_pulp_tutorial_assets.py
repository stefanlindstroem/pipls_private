from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

FIGURE_FILENAMES = (
    "component_path.svg",
    "scores.svg",
    "biplot.svg",
    "x_loadings.svg",
    "y_loadings.svg",
    "predictor_directions.svg",
    "dilation.svg",
    "response_directions.svg",
    "weighted_response_directions.svg",
    "coefficients.svg",
    "observed_vs_predicted.svg",
    "residuals_vs_predicted.svg",
    "standardized_rmse.svg",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def generated_pulp_assets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repository = _repository_root()
    output_dir = tmp_path_factory.mktemp("pulp-tutorial-assets") / "pulp"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(repository / "src"),
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    subprocess.run(
        [
            sys.executable,
            str(repository / "tools" / "render_pulp_tutorial.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repository,
        env=environment,
        check=True,
    )
    return output_dir


def test_pulp_tutorial_renderer_writes_declared_parseable_svgs(
    generated_pulp_assets: Path,
) -> None:
    manifest_path = generated_pulp_assets / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["dataset"]["name"] == "Pulp"
    assert manifest["analysis"]["chosen_n_components"] >= 1
    assert (
        manifest["analysis"]["chosen_predictor_rank"] >= manifest["analysis"]["chosen_n_components"]
    )
    assert manifest["analysis"]["displayed_components"] == [1, 2, 3]
    assert manifest["analysis"]["detailed_responses"] == ["CSF", "Density", "TI"]

    figures = manifest["figures"]
    assert tuple(item["filename"] for item in figures) == FIGURE_FILENAMES
    assert set(path.name for path in generated_pulp_assets.iterdir()) == {
        *FIGURE_FILENAMES,
        "manifest.json",
    }
    for item in figures:
        figure_path = generated_pulp_assets / item["filename"]
        ET.parse(figure_path)
        assert item["sha256"] == _sha256(figure_path)


def test_pulp_tutorial_manifest_records_dataset_hashes(
    generated_pulp_assets: Path,
) -> None:
    repository = _repository_root()
    manifest = json.loads((generated_pulp_assets / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["dataset"]["files"] == {
        "X.csv": _sha256(repository / "datasets" / "pulp" / "X.csv"),
        "Y.csv": _sha256(repository / "datasets" / "pulp" / "Y.csv"),
    }


def test_documentation_targets_own_generated_pulp_assets() -> None:
    repository = _repository_root()
    makefile = (repository / "Makefile").read_text(encoding="utf-8")
    gitignore = (repository / ".gitignore").read_text(encoding="utf-8")
    manifest = (repository / "MANIFEST.in").read_text(encoding="utf-8")
    with (repository / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)
    sdist_checker = (repository / "tools" / "check_sdist_docs.py").read_text(encoding="utf-8")

    assert "docs-figures:" in makefile
    assert "docs: docs-figures" in makefile
    assert "docs-serve: docs-figures" in makefile
    assert "docs/assets/generated" in makefile
    assert "docs/assets/generated/" in gitignore
    assert "include tools/render_pulp_tutorial.py" in manifest
    docs_dependencies = pyproject["project"]["optional-dependencies"]["docs"]
    assert "pandas>=2.0" in docs_dependencies
    assert "matplotlib>=3.8" in docs_dependencies
    assert "render_pulp_tutorial.py" in sdist_checker
    assert 'source / "docs" / "tutorials" / "pulp.md"' in sdist_checker
    assert 'source / "examples" / "10_pulp_real_data.py"' in sdist_checker
    assert 'source / "site" / "tutorials" / "pulp" / "index.html"' in sdist_checker
    assert "PULP_TUTORIAL_FIGURES" in sdist_checker


def test_pulp_tutorial_is_the_primary_generated_workflow() -> None:
    repository = _repository_root()
    tutorial_path = repository / "docs" / "tutorials" / "pulp.md"
    tutorial = tutorial_path.read_text(encoding="utf-8")
    workflow = (repository / "examples" / "_support" / "pulp_workflow.py").read_text(
        encoding="utf-8"
    )
    with (repository / "mkdocs.yml").open(encoding="utf-8") as stream:
        mkdocs = yaml.safe_load(stream)

    tutorial_nav = next(item["Tutorial"] for item in mkdocs["nav"] if "Tutorial" in item)
    assert tutorial_nav == [{"Pulp workflow": "tutorials/pulp.md"}]
    snippets = next(
        extension["pymdownx.snippets"]
        for extension in mkdocs["markdown_extensions"]
        if isinstance(extension, dict) and "pymdownx.snippets" in extension
    )
    assert snippets == {
        "base_path": ["."],
        "check_paths": True,
        "dedent_subsections": True,
    }

    linked_targets = re.findall(r"\]\(([^)#]+)(?:#[^)]+)?\)", tutorial)
    assert "tutorials/pulp.md" in (repository / "docs" / "index.md").read_text(
        encoding="utf-8"
    )
    assert "tutorials/pulp.md" in (repository / "docs" / "quickstart.md").read_text(
        encoding="utf-8"
    )
    assert "tutorials/pulp.md" in (repository / "docs" / "examples.md").read_text(
        encoding="utf-8"
    )
    assert "../cross_validation.md" in linked_targets
    assert "../model_inspection.md" in linked_targets
    assert "../theory.md" in linked_targets

    snippet_sections = {
        "load-pulp-data",
        "build-pulp-pipeline",
        "evaluate-pulp-component-path",
        "select-pulp-predictor-rank",
        "fit-pulp-pipeline",
        "pulp-oof-predictions",
        "pulp-inspection-results",
    }
    for section in snippet_sections:
        assert f'examples/_support/pulp_workflow.py:{section}' in tutorial
        assert f"# --8<-- [start:{section}]" in workflow
        assert f"# --8<-- [end:{section}]" in workflow

    for filename in FIGURE_FILENAMES:
        assert f"../assets/generated/pulp/{filename}" in tutorial

    plotting_functions = {
        "plot_scores",
        "plot_biplot",
        "plot_x_loadings",
        "plot_y_loadings",
        "plot_pipls_predictor_directions",
        "plot_pipls_dilation",
        "plot_pipls_response_directions",
        "plot_pipls_weighted_response_directions",
        "plot_coefficients",
        "plot_observed_vs_predicted",
        "plot_residuals_vs_predicted",
        "plot_standardized_rmse",
    }
    for function_name in plotting_functions:
        assert f"../api/plotting.md#pipls.plotting.{function_name}" in tutorial

    prediction_kind = (
        'PULP_PREDICTION_KIND: PredictionKind = "selection-conditioned OOF predictions"'
    )
    assert prediction_kind in workflow
    assert 'examples/10_pulp_real_data.py' in tutorial
