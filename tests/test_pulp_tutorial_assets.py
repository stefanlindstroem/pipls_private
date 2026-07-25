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
    assert "adjustText>=1.4,<2" in docs_dependencies
    assert "render_pulp_tutorial.py" in sdist_checker
    assert 'source / "docs" / "tutorials" / "pulp.md"' in sdist_checker
    assert 'source / "examples" / "05_pulp_real_data.py"' in sdist_checker
    assert 'source / "site" / "tutorials" / "pulp" / "index.html"' in sdist_checker
    assert "PULP_TUTORIAL_FIGURES" in sdist_checker


def test_pulp_tutorial_is_the_complete_generated_workflow() -> None:
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

    tutorial_nav = next(item["Tutorials"] for item in mkdocs["nav"] if "Tutorials" in item)
    assert [next(iter(item.values())) for item in tutorial_nav] == [
        "tutorials/synthetic.md",
        "tutorials/pulp.md",
    ]
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
    assert "tutorials/synthetic.md" in (
        repository / "docs" / "api" / "regression.md"
    ).read_text(encoding="utf-8")
    assert "tutorials/pulp.md" in (repository / "docs" / "examples.md").read_text(
        encoding="utf-8"
    )
    assert "../api/regression.md" in linked_targets
    assert "../path_analysis.md" in linked_targets
    assert "../model_inspection.md" in linked_targets
    assert "../theory.md" in linked_targets

    example_sections = {
        "pulp-tutorial-setup",
        "load-pulp-data",
        "evaluate-pulp-component-path",
        "select-pulp-parameters",
        "fit-pulp-model",
        "pulp-oof-predictions",
        "pulp-inspection-results",
        "plot-pulp-component-path",
        "extract-pulp-rank-profile",
        "plot-pulp-rank-profile",
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

    coverage_start = tutorial.index("## What this tutorial covers")
    setup_start = tutorial.index("## Setup")
    coverage = tutorial[coverage_start:setup_start]
    assert coverage_start < setup_start
    assert len(re.findall(r"^\d+\. ", coverage, flags=re.MULTILINE)) >= 6

    assert tutorial.index("pulp-tutorial-setup") < tutorial.index("load-pulp-data")
    assert tutorial.index("select-pulp-parameters") < tutorial.index(
        "plot-pulp-component-path"
    ) < tutorial.index("fit-pulp-model")
    assert '--8<-- "examples/05_pulp_real_data.py"' not in tutorial
    assert "six SVG figures" not in tutorial

    selected_line = "selected = path.for_n_components(CHOSEN_N_COMPONENTS)"
    component_plot = '# --8<-- [start:plot-pulp-component-path]'
    profile_line = "rank_profile = path_search.predictor_rank_profile("
    model_fit = "model = PiPLSRegression("
    assert (
        example.index(selected_line)
        < example.index(component_plot)
        < example.index(profile_line)
        < example.index(model_fit)
    )
    assert "DETAILED_RESPONSE_COUNT = 3" in example
    assert "tuple(range(DETAILED_RESPONSE_COUNT))" in example
    for source in (example, renderer):
        assert "display_components = tuple(range(CHOSEN_N_COMPONENTS))" in source
        assert "DISPLAY_COMPONENTS" not in source
        assert "len(display_components)" not in source
    assert "DETAILED_RESPONSES" not in example
    assert "DETAILED_RESPONSES" not in renderer

    renderer_biplot = renderer.index("# --8<-- [start:render-pulp-biplot]")
    renderer_adjust = renderer.index("    adjust_text(", renderer_biplot)
    assert renderer.index("    axis.set_aspect", renderer_biplot) < renderer_adjust
    assert renderer.index("    axis.legend()", renderer_biplot) < renderer_adjust
    assert "from adjustText import adjust_text" in example
    assert "from matplotlib.patches import FancyArrowPatch" in example
    assert "prevent_crossings=False" in renderer
    assert "iter_lim=200" in renderer

    for section in renderer_sections:
        section_start = renderer.index(f"# --8<-- [start:{section}]")
        section_end = renderer.index(f"# --8<-- [end:{section}]", section_start)
        snippet = renderer[section_start:section_end]
        assert "plt.subplots(" in snippet
        assert "_figure(" not in snippet

    residual_start = renderer.index(
        "# --8<-- [start:render-pulp-residuals-vs-predicted]"
    )
    residual_end = renderer.index(
        "# --8<-- [end:render-pulp-residuals-vs-predicted]", residual_start
    )
    assert "detailed_array = np.array(" in renderer[residual_start:residual_end]

    renderer_selected = renderer.index("selected = component_path.for_n_components(")
    renderer_plot = renderer.index("    _render_component_path(", renderer_selected)
    renderer_profile = renderer.index(
        "rank_profile = path_search.predictor_rank_profile(", renderer_plot
    )
    renderer_rank_plot = renderer.index(
        "    _render_predictor_rank_profile(", renderer_profile
    )
    assert (
        renderer_selected
        < renderer_plot
        < renderer_profile
        < renderer_rank_plot
        < renderer.index("model = PiPLSRegression(")
    )

    for source in (tutorial, example, renderer):
        assert "Number of response components" not in source
    assert 'axis.set_xlabel("Number of components")' in example
    assert 'axis.set_xlabel("Number of components")' in renderer
    assert 'legend(title="Component")' not in example
    assert 'legend(title="Component")' not in renderer
    assert 'cv_results["predictor_rank"]' not in example
    assert 'cv_results["predictor_rank"]' not in renderer

    assert (
        tutorial.index("biplot.svg")
        < tutorial.index("predictor_directions.svg")
        < tutorial.index("weighted_response_directions.svg")
        < tutorial.index("observed_vs_predicted.svg")
        < tutorial.index("residuals_vs_predicted.svg")
        < tutorial.index("standardized_rmse.svg")
    )
    omitted_figures = {
        "scores.svg",
        "x_loadings.svg",
        "y_loadings.svg",
        "dilation.svg",
        "response_directions.svg",
    }
    for filename in omitted_figures:
        assert f"../assets/generated/pulp/{filename}" not in tutorial
        assert f'"{filename}"' not in renderer
    assert "### Regression coefficients" not in tutorial
    assert "coefficients.svg" not in tutorial
    assert "plot_coefficients" not in renderer

    assert "pipls.plotting" not in example
    assert "pipls.plotting" not in renderer
    assert "factors.predictor_directions" in example
    assert "factors.predictor_directions" in renderer
    assert "factors.weighted_response_directions" in renderer
    assert "factors.dilation" in example
    assert "factors.response_directions" in example
    assert "factors.weighted_response_directions" in example

    for field in {
        "diagnostics.observed_standardized",
        "diagnostics.predicted_standardized",
        "diagnostics.residual_standardized",
        "diagnostics.standardized_rmse",
    }:
        assert field in example
        assert field in renderer
    assert "axes[0].scatter(" in example
    assert "axes[1].scatter(" in example
    assert "axes[2].bar(" in example
    assert 'set_ylabel("Standardized residual")' in example
    assert 'set_ylabel("Standardized residual")' in renderer
    assert r"Residual $y-\hat y$ (standardized)" not in example
    assert r"Residual $y-\hat y$ (standardized)" not in renderer
    assert 'prediction_kind="selection-conditioned OOF predictions"' in example
    assert "run_pulp_workflow" not in example
    assert "run_pulp_workflow" not in renderer
    assert "PiPLSPathCV().fit(X, Y)" in renderer
    assert "cross_val_predict(" in renderer

def test_documentation_layers_have_distinct_ownership() -> None:
    repository = _repository_root()
    docs = repository / "docs"
    tutorial_path = docs / "tutorials" / "pulp.md"
    tutorial = tutorial_path.read_text(encoding="utf-8")
    inspection = (docs / "model_inspection.md").read_text(encoding="utf-8")
    examples = (docs / "examples.md").read_text(encoding="utf-8")
    regression_reference = (docs / "api" / "regression.md").read_text(encoding="utf-8")
    path_reference = (docs / "path_analysis.md").read_text(encoding="utf-8")
    with (repository / "mkdocs.yml").open(encoding="utf-8") as stream:
        mkdocs = yaml.safe_load(stream)

    image_owners = {
        path.relative_to(docs).as_posix()
        for path in docs.rglob("*.md")
        if "decisions" not in path.relative_to(docs).parts
        and "assets/generated/pulp/" in path.read_text(encoding="utf-8")
    }
    assert image_owners == {"tutorials/pulp.md"}

    inspection_anchors = {
        "scores",
        "score-loading-biplot",
        "x-loadings",
        "y-loadings",
        "predictor-directions",
        "dilation",
        "response-directions",
        "weighted-response-directions",
        "regression-coefficients",
        "observed-versus-predicted",
        "residuals-versus-predicted",
        "standardized-rmse",
        "observation-diagnostics",
    }
    for anchor in inspection_anchors:
        assert f"{{ #{anchor} }}" in inspection

    tutorial_anchors = {
        "score-loading-biplot",
        "predictor-directions",
        "observed-versus-predicted",
        "residuals-versus-predicted",
        "standardized-rmse",
    }
    for anchor in tutorial_anchors:
        assert f"../model_inspection.md#{anchor}" in tutorial

    assert "../model_inspection.md#regression-coefficients" not in tutorial
    assert not (docs / "api" / "plotting.md").exists()

    assert "examples/results/" not in inspection
    assert "post_analysis.pdf" not in inspection
    assert "Sugarcane workflow" not in inspection
    assert "Tobacco workflow" not in inspection
    assert "assets/generated/pulp/" not in examples

    removed_guides = {
        "quickstart.md",
        "estimator_api.md",
        "parameter_selection.md",
        "preprocessing.md",
    }
    assert not any((docs / filename).exists() for filename in removed_guides)
    navigation_text = (repository / "mkdocs.yml").read_text(encoding="utf-8")
    assert not any(filename in navigation_text for filename in removed_guides)
    assert [next(iter(item)) for item in mkdocs["nav"]] == [
        "Home",
        "Tutorials",
        "Examples",
        "Reference",
        "Project validation",
        "Scientific background",
    ]

    project_validation = next(
        item["Project validation"] for item in mkdocs["nav"] if "Project validation" in item
    )
    assert project_validation == [
        {"Reference datasets": "datasets.md"},
        {"Benchmarks": "benchmarks.md"},
        {"Reproducibility": "reproducibility.md"},
        {"Compatibility": "compatibility.md"},
    ]

    assert "tutorials/synthetic.md" in regression_reference
    assert "tutorials/synthetic.md" in path_reference
    assert "Example 04" not in path_reference
    assert "Examples 05" not in path_reference
