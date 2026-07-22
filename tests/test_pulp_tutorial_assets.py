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
    assert "render_pulp_tutorial.py" in sdist_checker
    assert 'source / "docs" / "tutorials" / "pulp.md"' in sdist_checker
    assert 'source / "examples" / "10_pulp_real_data.py"' in sdist_checker
    assert 'source / "site" / "tutorials" / "pulp" / "index.html"' in sdist_checker
    assert "PULP_TUTORIAL_FIGURES" in sdist_checker


def test_pulp_tutorial_is_the_primary_generated_workflow() -> None:
    repository = _repository_root()
    tutorial_path = repository / "docs" / "tutorials" / "pulp.md"
    tutorial = tutorial_path.read_text(encoding="utf-8")
    example = (repository / "examples" / "10_pulp_real_data.py").read_text(
        encoding="utf-8"
    )
    renderer = (repository / "tools" / "render_pulp_tutorial.py").read_text(
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
    assert "tutorials/pulp.md" in (repository / "docs" / "index.md").read_text(encoding="utf-8")
    assert "tutorials/pulp.md" in (repository / "docs" / "quickstart.md").read_text(
        encoding="utf-8"
    )
    assert "tutorials/pulp.md" in (repository / "docs" / "examples.md").read_text(encoding="utf-8")
    assert "../cross_validation.md" in linked_targets
    assert "../model_inspection.md" in linked_targets
    assert "../theory.md" in linked_targets

    snippet_sections = {
        "load-pulp-data",
        "evaluate-pulp-component-path",
        "select-pulp-parameters",
        "fit-pulp-model",
        "pulp-oof-predictions",
        "pulp-inspection-results",
        "plot-pulp-component-path",
    }
    for section in snippet_sections:
        assert f"examples/10_pulp_real_data.py:{section}" in tutorial
        assert f"# --8<-- [start:{section}]" in example
        assert f"# --8<-- [end:{section}]" in example

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

    select_snippet = 'examples/10_pulp_real_data.py:select-pulp-parameters'
    plot_snippet = 'examples/10_pulp_real_data.py:plot-pulp-component-path'
    fit_snippet = 'examples/10_pulp_real_data.py:fit-pulp-model'
    assert (
        tutorial.index(select_snippet)
        < tutorial.index(plot_snippet)
        < tutorial.index(fit_snippet)
    )

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
    renderer_selected = renderer.index("selected = component_path.for_n_components(")
    renderer_plot = renderer.index("    _render_component_path(", renderer_selected)
    renderer_profile = renderer.index(
        "rank_profile = path_search.predictor_rank_profile(",
        renderer_plot,
    )
    renderer_rank_plot = renderer.index(
        "    _render_predictor_rank_profile(",
        renderer_profile,
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
    assert "for n_components, cv_mse, predictor_rank in zip(" not in example
    assert "for n_components, mean_mse, predictor_rank in zip(" not in renderer

    assert "selects the rank that minimizes" in tutorial
    assert "rank that minimized mean CV-MSE conditional on" in tutorial
    assert "rank 10 gives the lowest evaluated mean CV-MSE" in tutorial
    assert "CV-MSE minimum: rank" in example
    assert "CV-MSE minimum: rank" in renderer
    assert "path_search.predictor_rank_profile(selected.n_components)" in example
    assert "path_search.predictor_rank_profile(selected.n_components)" in renderer
    assert 'cv_results["predictor_rank"]' not in example
    assert 'cv_results["predictor_rank"]' not in renderer
    assert 'legend(title="Component")' not in example
    assert 'legend(title="Component")' not in renderer

    assert 'prediction_kind="selection-conditioned OOF predictions"' in example
    assert "run_pulp_workflow" not in example
    assert "run_pulp_workflow" not in renderer
    assert "pulp_workflow" not in renderer
    assert "PiPLSPathCV(refit=False).fit(X, Y)" in renderer
    assert "cross_val_predict(" in renderer
    assert "examples/10_pulp_real_data.py" in tutorial


def test_documentation_layers_have_distinct_ownership() -> None:
    repository = _repository_root()
    docs = repository / "docs"
    tutorial_path = docs / "tutorials" / "pulp.md"
    tutorial = tutorial_path.read_text(encoding="utf-8")
    inspection = (docs / "model_inspection.md").read_text(encoding="utf-8")
    plotting_reference = (docs / "api" / "plotting.md").read_text(encoding="utf-8")
    examples = (docs / "examples.md").read_text(encoding="utf-8")
    parameter_selection = (docs / "parameter_selection.md").read_text(encoding="utf-8")
    path_reference = (docs / "path_analysis.md").read_text(encoding="utf-8")

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

    tutorial_anchors = inspection_anchors - {"observation-diagnostics"}
    for anchor in tutorial_anchors:
        assert f"../model_inspection.md#{anchor}" in tutorial

    assert "examples/results/" not in inspection
    assert "post_analysis.pdf" not in inspection
    assert "Sugarcane workflow" not in inspection
    assert "Tobacco workflow" not in inspection
    assert "```python" not in plotting_reference
    assert "assets/generated/pulp/" not in examples

    assert "path_analysis.md" in parameter_selection
    assert "tutorials/pulp.md" in parameter_selection
    assert "Example 09" not in path_reference
    assert "Examples 10" not in path_reference
