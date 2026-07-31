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
    assert manifest["analysis"]["chosen_n_components"] == 3
    assert manifest["analysis"]["chosen_predictor_rank"] == 10
    assert manifest["analysis"]["evaluated_predictor_ranks"] == list(range(3, 11))
    assert manifest["analysis"]["predictor_rank_at_upper_boundary"] is True
    assert manifest["analysis"]["displayed_components"] == [1, 2, 3]
    assert manifest["analysis"]["factor_sign_anchor"] == {
        "response": "TI",
        "sign": "positive",
    }
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

    expected_labels = {
        "predictor_directions.svg": r"Predictor direction $P_{:k}$",
        "weighted_response_directions.svg": (
            r"Weighted response direction $d_kQ_{:k}$"
        ),
    }
    for filename, label in expected_labels.items():
        svg = (generated_pulp_assets / filename).read_text(encoding="utf-8")
        assert label in svg, (filename, label)
        assert "Weighted direction $" not in svg, filename
        assert r"\mathbf{P}_{:" not in svg, filename
        assert r"\mathbf{Q}_{:" not in svg, filename
        assert r"\mathbf{D}_{" not in svg, filename
        assert "$q_{:" not in svg, filename
        assert "$d_kq_{:" not in svg, filename


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
    assert "adjustText>=1.4,<2" in docs_dependencies
    assert "render_pulp_tutorial.py" in sdist_checker
    assert 'source / "docs" / "tutorials" / "pulp.md"' in sdist_checker
    assert 'source / "examples" / "05_pulp_real_data.py"' in sdist_checker
    assert 'source / "site" / "tutorials" / "pulp" / "index.html"' in sdist_checker
    assert "PULP_TUTORIAL_FIGURES" in sdist_checker


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
    with (repository / "mkdocs.yml").open(encoding="utf-8") as stream:
        mkdocs = yaml.safe_load(stream)

    snippets = next(
        extension["pymdownx.snippets"]
        for extension in mkdocs["markdown_extensions"]
        if isinstance(extension, dict) and "pymdownx.snippets" in extension
    )
    assert snippets["check_paths"] is True

    linked_targets = set(re.findall(r"\]\(([^)#]+)(?:#[^)]+)?\)", tutorial))
    assert {
        "../api/path.md",
        "../api/regression.md",
        "../model_inspection.md",
    } <= linked_targets

    example_sections = {
        "pulp-tutorial-setup",
        "load-pulp-data",
        "evaluate-pulp-component-path",
        "select-pulp-parameters",
        "plot-pulp-component-path",
        "extract-pulp-rank-profile",
        "plot-pulp-rank-profile",
        "fit-pulp-model",
        "pulp-oof-predictions",
        "pulp-inspection-results",
    }
    for section in example_sections:
        assert f"examples/05_pulp_real_data.py:{section}" in tutorial
        assert f"# --8<-- [start:{section}]" in example
        assert f"# --8<-- [end:{section}]" in example

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
