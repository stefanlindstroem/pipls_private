from __future__ import annotations

import re
from pathlib import Path

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
        "fit-pulp-model",
        "inspect-pulp-selection",
        "plot-pulp-component-path",
        "plot-pulp-rank-profile",
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
