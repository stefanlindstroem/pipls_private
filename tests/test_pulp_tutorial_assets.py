from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

import pytest

from tests._mkdocs import load_mkdocs_config

FIGURE_FILENAMES = (
    "component_path.svg",
    "selected_component_path.svg",
    "predictor_rank_profile.svg",
    "biplot.svg",
    "predictor_directions.svg",
    "weighted_response_directions.svg",
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
    python_path = str(repository / "src")
    if importlib.util.find_spec("adjustText") is None:
        stub_dir = tmp_path_factory.mktemp("pulp-adjusttext-stub")
        (stub_dir / "adjustText.py").write_text(
            "def adjust_text(texts, *args, **kwargs):\n    return texts\n",
            encoding="utf-8",
        )
        python_path = f"{stub_dir}{os.pathsep}{python_path}"
    environment.update(
        {
            "PYTHONPATH": python_path,
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


def test_pulp_tutorial_renderer_records_repeated_cv_and_valid_figures(
    generated_pulp_assets: Path,
) -> None:
    manifest = json.loads(
        (generated_pulp_assets / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == 1
    assert manifest["dataset"]["id"] == "pulp"
    assert manifest["dataset"]["version"] == "1"
    analysis = manifest["analysis"]
    assert analysis["chosen_n_components"] == 3
    assert analysis["chosen_predictor_rank"] == 9
    assert analysis["evaluated_predictor_ranks"] == list(range(3, 11))
    assert analysis["cross_validation"] == {
        "splitter": "RepeatedKFold",
        "n_splits": 5,
        "n_repeats": 10,
        "random_state": 0,
        "materialized_splits": 50,
    }
    assert analysis["oof_predictions_per_observation"] == 10
    assert analysis["prediction_kind"] == (
        "selection-conditioned OOF predictions"
    )

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

    initial_path = (generated_pulp_assets / "component_path.svg").read_text(
        encoding="utf-8"
    )
    selected_path = (
        generated_pulp_assets / "selected_component_path.svg"
    ).read_text(encoding="utf-8")
    assert "Chosen:" not in initial_path
    assert "Chosen:" in selected_path


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
    assert "pandas>=2.0" not in docs_dependencies
    assert "matplotlib>=3.8" in docs_dependencies
    assert "adjustText>=1.4,<2" in docs_dependencies
    assert "render_pulp_tutorial.py" in sdist_checker
    assert 'source / "docs" / "tutorials" / "pulp.md"' in sdist_checker
    assert 'source / "examples" / "05_pulp_real_data.py"' in sdist_checker
    assert 'source / "site" / "tutorials" / "pulp" / "index.html"' in sdist_checker
    assert "PULP_TUTORIAL_FIGURES" in sdist_checker
    assert '"splitter": "RepeatedKFold"' in sdist_checker
    assert '"materialized_splits": 50' in sdist_checker
    assert (
        'pulp_analysis.get("oof_predictions_per_observation") != 10'
        in sdist_checker
    )


def test_pulp_tutorial_uses_checked_snippets_assets_and_public_links() -> None:
    repository = _repository_root()
    tutorial = (repository / "docs" / "tutorials" / "pulp.md").read_text(
        encoding="utf-8"
    )
    example = (repository / "examples" / "05_pulp_real_data.py").read_text(
        encoding="utf-8"
    )
    renderer = (repository / "tools" / "render_pulp_tutorial.py").read_text(
        encoding="utf-8"
    )
    mkdocs = load_mkdocs_config(repository / "mkdocs.yml")

    snippets = next(
        extension["pymdownx.snippets"]
        for extension in mkdocs["markdown_extensions"]
        if isinstance(extension, dict) and "pymdownx.snippets" in extension
    )
    assert snippets["check_paths"] is True
    assert "from sklearn.model_selection import RepeatedKFold" in example
    assert (
        "RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)" in example
    )
    assert "from sklearn.model_selection import RepeatedKFold" in renderer

    linked_targets = set(re.findall(r"\]\(([^)#]+)(?:#[^)]+)?\)", tutorial))
    assert {
        "../api/path.md",
        "../api/regression.md",
        "../model_inspection.md",
    } <= linked_targets

    example_sections = {
        "pulp-tutorial-setup",
        "load-pulp-data",
        "inspect-pulp-component-path",
        "choose-pulp-selection",
        "inspect-pulp-selected-evidence",
        "plot-pulp-component-path",
        "plot-pulp-selected-component-path",
        "plot-pulp-rank-profile",
        "pulp-oof-predictions",
        "pulp-oof-inspection-results",
        "fit-pulp-model",
        "pulp-fitted-model-inspection-results",
    }
    for section in example_sections:
        assert f"examples/05_pulp_real_data.py:{section}" in tutorial
        assert f"# --8<-- [start:{section}]" in example
        assert f"# --8<-- [end:{section}]" in example

    assert tutorial.index("inspect-pulp-selected-evidence") < tutorial.index(
        "pulp-oof-predictions"
    )
    assert tutorial.index("pulp-oof-predictions") < tutorial.index(
        "pulp-oof-inspection-results"
    )
    assert tutorial.index("pulp-oof-inspection-results") < tutorial.index(
        "fit-pulp-model"
    )
    assert tutorial.index("fit-pulp-model") < tutorial.index(
        "pulp-fitted-model-inspection-results"
    )
    assert "selection = model.selection_" not in example
    assert "selection = model.selection_" not in renderer
    assert "selection = search.select(n_components=CHOSEN_N_COMPONENTS)" in example
    assert "selection = search.select(n_components=chosen_n_components)" in renderer
    assert "model = search.refit(" in example
    assert "selection=selection" in example
    assert "model = search.refit(X, Y, selection=selection)" in renderer

    renderer_sections = {
        "render-pulp-biplot",
        "render-pulp-predictor-directions",
        "render-pulp-weighted-response-directions",
        "render-pulp-observed-vs-predicted",
        "render-pulp-residuals-vs-predicted",
        "render-pulp-standardized-rmse",
    }
    for section in renderer_sections:
        assert f"tools/render_pulp_tutorial.py:{section}" in tutorial
        assert f"# --8<-- [start:{section}]" in renderer
        assert f"# --8<-- [end:{section}]" in renderer

    for filename in FIGURE_FILENAMES:
        assert f"../assets/generated/pulp/{filename}" in tutorial


def test_pulp_assets_and_model_inspection_links_have_distinct_owners() -> None:
    repository = _repository_root()
    docs = repository / "docs"
    tutorial = (docs / "tutorials" / "pulp.md").read_text(encoding="utf-8")
    inspection = (docs / "model_inspection.md").read_text(encoding="utf-8")

    image_owners = {
        path.relative_to(docs).as_posix()
        for path in docs.rglob("*.md")
        if "decisions" not in path.relative_to(docs).parts
        and "assets/generated/pulp/" in path.read_text(encoding="utf-8")
    }
    assert image_owners == {"tutorials/pulp.md"}

    tutorial_anchors = {
        "score-loading-biplot",
        "predictor-directions",
        "observed-versus-predicted",
        "residuals-versus-predicted",
        "standardized-rmse",
    }
    for anchor in tutorial_anchors:
        assert f"{{ #{anchor} }}" in inspection
        assert f"../model_inspection.md#{anchor}" in tutorial
