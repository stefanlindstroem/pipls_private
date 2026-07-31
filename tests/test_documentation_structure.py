from __future__ import annotations

import inspect
import re
import unicodedata
from pathlib import Path

import yaml

from pipls import (
    PiPLSComponentPath,
    PiPLSDecomposition,
    PiPLSRegression,
    PiPLSSearchCV,
)
from pipls.datasets import PiPLSLatentGeometryTruth, make_pipls_latent_geometry
from pipls.inspection import PiPLSDisplayFactors

_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target>[^)]+)\)")
_HEADING = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$", re.MULTILINE)
_EXPLICIT_ANCHOR = re.compile(r"\{\s*#(?P<anchor>[A-Za-z0-9_.:-]+)\s*\}")
_MKDOCSTRINGS_DIRECTIVE = re.compile(
    r"^::: (?P<object>[A-Za-z_][A-Za-z0-9_.]*)\s*$",
    re.MULTILINE,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _heading_slug(title: str) -> str:
    title = re.sub(r"\s+\{[^}]*\}\s*$", "", title)
    title = re.sub(r"`([^`]*)`", r"\1", title)
    title = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", title)
    title = re.sub(r"<[^>]+>", "", title)
    normalized = unicodedata.normalize("NFKD", title)
    ascii_title = normalized.encode("ascii", "ignore").decode("ascii").lower()
    ascii_title = re.sub(r"[^\w\s-]", "", ascii_title)
    return re.sub(r"[-\s]+", "-", ascii_title).strip("-")


def _document_anchors(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    anchors = set(_EXPLICIT_ANCHOR.findall(text))
    anchors.update(_MKDOCSTRINGS_DIRECTIVE.findall(text))
    anchors.update(_heading_slug(match.group("title")) for match in _HEADING.finditer(text))
    return anchors


def _local_target(raw_target: str) -> tuple[str, str] | None:
    target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
    if not target or target.startswith(("http://", "https://", "mailto:")):
        return None
    path, _, anchor = target.partition("#")
    return path, anchor


def test_served_markdown_local_links_and_anchors_resolve() -> None:
    docs_root = (_repository_root() / "docs").resolve()
    served_pages = sorted(
        path
        for path in docs_root.rglob("*.md")
        if "decisions" not in path.relative_to(docs_root).parts
    )

    for source in served_pages:
        for match in _MARKDOWN_LINK.finditer(source.read_text(encoding="utf-8")):
            parsed = _local_target(match.group("target"))
            if parsed is None:
                continue
            relative_path, anchor = parsed
            destination = (source.parent / relative_path).resolve() if relative_path else source
            assert destination.is_relative_to(docs_root), (
                f"{source.relative_to(docs_root)} links outside docs/: {match.group('target')}"
            )
            generated_assets = docs_root / "assets" / "generated"
            if destination.is_relative_to(generated_assets):
                continue
            assert destination.is_file(), (
                f"{source.relative_to(docs_root)} links to a missing file: {match.group('target')}"
            )
            if anchor and destination.suffix == ".md":
                assert anchor in _document_anchors(destination), (
                    f"{source.relative_to(docs_root)} links to a missing anchor: "
                    f"{match.group('target')}"
                )


def test_api_overview_maps_the_public_result_objects() -> None:
    api_overview = (_repository_root() / "docs" / "api" / "index.md").read_text(
        encoding="utf-8"
    )
    result_objects = {
        "PiPLSComponentPath",
        "PiPLSComponentResult",
        "PiPLSPredictorRankProfile",
        "PiPLSValidationReport",
        "PiPLSDecomposition",
        "LatentStructure",
        "PiPLSDisplayFactors",
        "BiplotCoordinates",
        "ObservationDiagnostics",
        "PredictionDiagnostics",
        "PiPLSDataset",
        "PiPLSLatentGeometryTruth",
        "PiPLSSyntheticTruth",
    }

    for object_name in result_objects:
        assert f"`{object_name}`" in api_overview


def test_dataset_api_routes_to_repository_reference_datasets() -> None:
    root = _repository_root()
    page = (root / "docs" / "api" / "datasets.md").read_text(encoding="utf-8")
    overview = (root / "docs" / "api" / "index.md").read_text(encoding="utf-8")

    for dataset_name in ("Pulp", "Sugarcane", "Tobacco"):
        assert dataset_name in page
        assert dataset_name in overview
    assert "../datasets.md" in page
    assert "../examples.md#complete-real-data-analyses" in page
    assert "../datasets.md" in overview


def test_troubleshooting_is_a_programming_reference_page() -> None:
    root = _repository_root()
    page = (root / "docs" / "troubleshooting.md").read_text(encoding="utf-8")
    with (root / "mkdocs.yml").open(encoding="utf-8") as stream:
        navigation = yaml.safe_load(stream)["nav"]

    reference = next(item["Reference"] for item in navigation if "Reference" in item)
    assert {"Troubleshooting": "troubleshooting.md"} in reference
    assert "docs/troubleshooting.md" in (root / "README.md").read_text(encoding="utf-8")
    assert "troubleshooting.md" in (root / "docs" / "index.md").read_text(encoding="utf-8")

    required_targets = {
        "api/regression.md",
        "api/path.md",
        "path_analysis.md",
    }
    linked_targets = {
        parsed[0]
        for match in _MARKDOWN_LINK.finditer(page)
        if (parsed := _local_target(match.group("target"))) is not None
    }
    assert required_targets <= linked_targets
    assert "assets/generated/" not in page
    assert "--8<--" not in page


def test_served_guides_do_not_expose_internal_phase_or_patch_labels() -> None:
    docs_root = _repository_root() / "docs"
    internal_label = re.compile(r"\b(?:Phase [A-Z][A-Za-z0-9.-]*|Patch D\d+)\b")

    for path in sorted(docs_root.rglob("*.md")):
        if "decisions" in path.relative_to(docs_root).parts:
            continue
        assert internal_label.search(path.read_text(encoding="utf-8")) is None, path


def test_reference_navigation_is_consolidated_around_owning_pages() -> None:
    root = _repository_root()
    with (root / "mkdocs.yml").open(encoding="utf-8") as stream:
        navigation = yaml.safe_load(stream)["nav"]

    reference = next(item["Reference"] for item in navigation if "Reference" in item)
    assert reference == [
        {"API overview": "api/index.md"},
        {"Fixed regression": "api/regression.md"},
        {"Path selection": "api/path.md"},
        {"Path-selection details": "path_analysis.md"},
        {"Troubleshooting": "troubleshooting.md"},
        {
            "Model inspection": [
                {"Concepts": "model_inspection.md"},
                {"API": "api/inspection.md"},
            ]
        },
        {"Dataset API and generators": "api/datasets.md"},
    ]

    retired_wrappers = {
        "cross_validation.md",
        "api/decomposition.md",
        "api/validation.md",
        "api/exceptions.md",
        "api/metrics.md",
    }
    assert not any((root / "docs" / relative).exists() for relative in retired_wrappers)

    regression = (root / "docs" / "api" / "regression.md").read_text(encoding="utf-8")
    path = (root / "docs" / "api" / "path.md").read_text(encoding="utf-8")
    assert "::: pipls.PiPLSDecomposition" in regression
    assert "::: pipls.StatisticalSupportWarning" in regression
    assert "::: pipls.PiPLSValidationReport" in path
    assert "::: pipls.metrics.response_standardized_mean_squared_error" in path
    assert "::: pipls.metrics.neg_response_standardized_mean_squared_error" in path

    for heading, directive in (
        ("## Concise component path", "::: pipls.PiPLSComponentPath"),
        ("## One component result", "::: pipls.PiPLSComponentResult"),
        ("## Predictor-rank profile", "::: pipls.PiPLSPredictorRankProfile"),
    ):
        section = path.split(heading, maxsplit=1)[1].split(directive, maxsplit=1)[0]
        assert section.strip(), f"{heading} requires explanatory text before its API directive"

    inspection_concepts = (root / "docs" / "model_inspection.md").read_text(
        encoding="utf-8"
    )
    inspection_api = (root / "docs" / "api" / "inspection.md").read_text(
        encoding="utf-8"
    )
    assert "api/inspection.md" in inspection_concepts
    assert "../model_inspection.md" in inspection_api


def test_path_api_explains_workflows_and_candidate_configuration() -> None:
    root = _repository_root()
    page = (root / "docs" / "api" / "path.md").read_text(encoding="utf-8")
    details = (root / "docs" / "path_analysis.md").read_text(encoding="utf-8")
    troubleshooting = (root / "docs" / "troubleshooting.md").read_text(
        encoding="utf-8"
    )
    search_doc = inspect.getdoc(PiPLSSearchCV) or ""

    workflow = page.split("## Choose the workflow", maxsplit=1)[1].split(
        "## Inspect the path and fit one fixed model", maxsplit=1
    )[0]
    assert "PiPLSRegression" in workflow
    assert "PiPLSSearchCV()" in workflow
    assert "selection_rule" in workflow
    assert "refit=True" in workflow

    fixed_fit = page.split(
        "## Inspect the path and fit one fixed model", maxsplit=1
    )[1].split("## Configure the candidate estimator", maxsplit=1)[0]
    for required in (
        "search = PiPLSSearchCV().fit(X, Y)",
        "path = search.component_path_",
        "path.for_n_components(CHOSEN_N_COMPONENTS)",
        "n_components=selected.n_components",
        "predictor_rank=selected.predictor_rank",
    ):
        assert required in fixed_fit

    configuration = page.split(
        "## Configure the candidate estimator", maxsplit=1
    )[1].split("::: pipls.PiPLSSearchCV", maxsplit=1)[0]
    for required in (
        "template = PiPLSRegression(",
        'svd_solver="full"',
        "random_state=0",
        "clone(template).set_params(",
        "estimator=None",
        "search.selected_pipls_.decomposition_.predictor_svd_solver",
        "refit=True",
    ):
        assert required in configuration

    canonical_link = "api/path.md#configure-the-candidate-estimator"
    assert canonical_link in details
    assert "## I need to change scaling or the SVD solver during search" in troubleshooting
    assert canonical_link in troubleshooting
    assert "PiPLSSearchCV` has no separate `scale` or `svd_solver` parameter" in troubleshooting

    for setting in ("scale", "copy", "svd_solver", "random_state"):
        assert f"``{setting}``" in search_doc
    assert "replace only ``n_components`` and ``predictor_rank``" in search_doc


def test_path_reference_defines_resolved_ceilings_before_rank_policies() -> None:
    page = (_repository_root() / "docs" / "path_analysis.md").read_text(
        encoding="utf-8"
    )

    bounds = page.index("## Search bounds")
    ceilings = page.index("### Resolved ceilings")
    component_requests = page.index("### Component-count requests")
    rank_policies = page.index("## Predictor-rank policies")
    assert bounds < ceilings < component_requests < rank_policies
    assert r"r_{\pi,\mathrm{max}}" in page[ceilings:component_requests]
    assert r"h_{\mathrm{max}}" in page[ceilings:component_requests]


def test_new_user_entry_explains_scope_before_routing_to_workflows() -> None:
    root = _repository_root()
    readme = (root / "README.md").read_text(encoding="utf-8")
    home = (root / "docs" / "index.md").read_text(encoding="utf-8")

    for page in (readme, home):
        assert "multivariate" in page
        assert "predictor" in page
        assert "ordinary PLS" in page
        assert "every regression problem" in page or "general performance claim" in page

    assert home.index("## When Pi-PLS may be useful") < home.index("## Choose a tutorial")
    assert readme.index("ordinary PLS") < readme.index("Start with:")


def test_tutorials_defer_maintenance_details_until_after_the_workflow() -> None:
    tutorial_root = _repository_root() / "docs" / "tutorials"
    synthetic = (tutorial_root / "synthetic.md").read_text(encoding="utf-8")
    pulp = (tutorial_root / "pulp.md").read_text(encoding="utf-8")

    synthetic_coverage = synthetic.index("## What this tutorial covers")
    synthetic_workflow = synthetic.index("## Generate training and test data")
    synthetic_reproduce = synthetic.index("## Reproduce this tutorial")
    assert synthetic_coverage < synthetic_workflow < synthetic_reproduce
    assert "The executable calculation is maintained" not in synthetic[:synthetic_reproduce]
    assert synthetic.index("The executable calculation is maintained") > synthetic_reproduce
    assert synthetic.index("make docs-figures") > synthetic_reproduce

    pulp_coverage = pulp.index("## What this tutorial covers")
    pulp_setup = pulp.index("## Setup")
    pulp_workflow = pulp.index("## The data and modeling question")
    pulp_reproduce = pulp.index("## Reproduce this tutorial")
    assert pulp_coverage < pulp_setup < pulp_workflow < pulp_reproduce
    assert "Standalone interpretation-figure recipes are maintained" not in pulp[:pulp_reproduce]
    assert pulp.index("Standalone interpretation-figure recipes are maintained") > pulp_reproduce
    assert pulp.index("make docs-figures") > pulp_reproduce
    assert pulp.index('python -m pip install ".[examples]"') < pulp_workflow


def test_current_user_workflows_rely_on_the_selection_only_path_default() -> None:
    root = _repository_root()
    current_sources = [
        root / "README.md",
        root / "docs" / "troubleshooting.md",
        root / "docs" / "path_analysis.md",
        root / "docs" / "tutorials" / "synthetic.md",
        root / "docs" / "datasets.md",
        root / "examples" / "README.md",
        *sorted((root / "examples").glob("[0-9][0-9]_*.py")),
        root / "tools" / "render_synthetic_tutorial.py",
        root / "tools" / "render_pulp_tutorial.py",
    ]

    combined = "\n".join(path.read_text(encoding="utf-8") for path in current_sources)
    assert "PiPLSSearchCV(refit=False)" not in combined
    assert "refit=False," not in combined
    assert "PiPLSSearchCV()" in combined

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert 'selection_rule="one_standard_error"' in readme
    assert "model = search.selected_pipls_" in readme
    assert "Y_pred = model.predict(X_test)" in readme
    assert "Y_pred = search.predict(X_test)" not in readme

    path_analysis = (root / "docs" / "path_analysis.md").read_text(encoding="utf-8")
    troubleshooting = (root / "docs" / "troubleshooting.md").read_text(encoding="utf-8")
    for page in (path_analysis, troubleshooting):
        assert "model = search.selected_pipls_" in page
        assert "Y_pred = model.predict(X_new)" in page


def test_path_details_expose_adaptive_search_completion_status() -> None:
    page = (_repository_root() / "docs" / "path_analysis.md").read_text(encoding="utf-8")

    assert "path_search_exhaustive_" in page
    assert 'search_method="auto"' in page
    assert 'search_method="optimal"' in page


def test_cv_mse_error_bar_documentation_defines_explicit_one_se_heuristic() -> None:
    root = _repository_root()
    path_analysis = (root / "docs" / "path_analysis.md").read_text(encoding="utf-8")

    assert "One-standard-error component heuristic" in path_analysis
    assert "1-SE rule" in path_analysis
    assert "cv_mse_standard_error" in path_analysis
    assert "smallest evaluated component count" in path_analysis
    assert "PiPLSSearchCV" in path_analysis
    assert "Predeclared final-model selection" in path_analysis
    assert 'selection_rule="one_standard_error"' in path_analysis
    assert "selected_result_" in path_analysis
    assert "best_score_" in path_analysis
    assert "Tobacco one-standard-error workflow" in path_analysis
    assert "confidence intervals" in path_analysis

    for relative_path in (
        "docs/tutorials/synthetic.md",
        "docs/tutorials/pulp.md",
    ):
        tutorial = (root / relative_path).read_text(encoding="utf-8")
        assert "one fold-based standard error" in tutorial
        assert "one-standard-error rule" in tutorial
        assert "explicit" in tutorial


def test_example_catalogue_is_scannable_and_runnable() -> None:
    root = _repository_root()
    page_path = root / "docs" / "examples.md"
    page = page_path.read_text(encoding="utf-8")

    table_start = page.index("## Choose an example")
    run_one_start = page.index("## Run one example")
    table = page[table_start:run_one_start]
    rows = [
        line
        for line in table.splitlines()
        if line.startswith("| `") and line.endswith("|")
    ]

    expected_scripts = [
        path.name
        for path in sorted((root / "examples").glob("[0-9][0-9]_*.py"))
    ]
    documented_scripts: list[str] = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert len(cells) == 3
        assert all(cells)
        assert cells[0].startswith("`") and cells[0].endswith("`")
        documented_scripts.append(cells[0].strip("`"))

    assert documented_scripts == expected_scripts
    assert 'python -m pip install ".[examples]"' in page
    assert "python examples/02_synthetic_path_selection.py" in page
    assert "`examples/results/`" in page
    assert "make examples" in page

    anchors = _document_anchors(page_path)
    for anchor in (
        "leave-one-out-validation",
        "complete-real-data-analyses",
        "tobacco-one-standard-error-selection",
    ):
        assert anchor in anchors


def test_component_path_recommendations_have_one_maintained_application() -> None:
    root = _repository_root()
    method_names = (
        "minimum_cv_mse_result",
        "one_standard_error_result",
    )
    reference_pages = (
        root / "docs" / "path_analysis.md",
        root / "docs" / "api" / "index.md",
        root / "docs" / "api" / "path.md",
    )

    for page in reference_pages:
        page_text = page.read_text(encoding="utf-8")
        for method_name in method_names:
            assert method_name in page_text

    tobacco = root / "examples" / "07_tobacco_real_data.py"
    tobacco_text = tobacco.read_text(encoding="utf-8")
    assert "minimum_cv_mse_result" in tobacco_text
    assert "one_standard_error_result" in tobacco_text
    assert "one_se_threshold" in tobacco_text
    assert 'label="1-SE threshold"' in tobacco_text

    other_numbered_examples = sorted((root / "examples").glob("[0-9][0-9]_*.py"))
    promoted_paths = [
        root / "README.md",
        root / "docs" / "index.md",
        *sorted((root / "docs" / "tutorials").glob("*.md")),
        *(path for path in other_numbered_examples if path != tobacco),
        *sorted((root / "tools").glob("render_*_tutorial.py")),
    ]
    for promoted_path in promoted_paths:
        promoted_text = promoted_path.read_text(encoding="utf-8")
        for method_name in method_names:
            assert method_name not in promoted_text, promoted_path

    path_analysis = (root / "docs" / "path_analysis.md").read_text(encoding="utf-8")
    examples = (root / "docs" / "examples.md").read_text(encoding="utf-8")
    examples_readme = (root / "examples" / "README.md").read_text(encoding="utf-8")
    assert "examples.md#tobacco-one-standard-error-selection" in path_analysis
    assert "path_analysis.md#one-standard-error-component-heuristic" in examples
    assert "path_analysis.md#result-object-recommendations" in examples
    assert "api/path.md" in examples
    assert "no clear elbow" in examples
    assert "horizontal 1-SE threshold" in examples
    assert "one_standard_error_result()" in examples_readme
    assert "../docs/path_analysis.md#one-standard-error-component-heuristic" in examples_readme
    assert "../docs/examples.md#tobacco-one-standard-error-selection" in examples_readme
    assert "horizontal 1-SE threshold" in examples_readme


def test_authors_license_and_citation_page_is_public_and_consistent() -> None:
    root = _repository_root()
    page = (root / "docs" / "citation.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    with (root / "mkdocs.yml").open(encoding="utf-8") as stream:
        navigation = yaml.safe_load(stream)["nav"]

    scientific_background = next(
        item["Scientific background"]
        for item in navigation
        if "Scientific background" in item
    )
    assert {"Authors, license, and citation": "citation.md"} in scientific_background

    for name in ("Vishal Agrawal", "Fritjof Nilsson", "Stefan B. Lindström"):
        assert name in page
        assert name in readme
    assert "BSD 3-Clause License" in page
    assert "commercial use" in page
    assert "CACE-D-26-00847" in page
    assert "Manuscript under revision" in page
    assert "docs/citation.md" in readme
    assert "CITATION.cff" in readme


def test_theory_distinguishes_complete_matrices_from_indexed_columns() -> None:
    root = _repository_root()
    mathjax = (root / "docs" / "javascripts" / "mathjax.js").read_text(
        encoding="utf-8"
    )
    theory = (root / "docs" / "theory.md").read_text(encoding="utf-8")

    assert r"\bm" not in mathjax
    assert r"\bm" not in theory
    assert "$d_k=D_{kk}$" in theory
    assert "$(P_{:k},d_k,Q_{:k})$" in theory
    assert r"$\mathbf{X}P_{:k}$" in theory
    assert r"$\mathbf{Y}Q_{:k}$" in theory
    assert r"\mathbf{D}_{kk}" not in theory
    assert r"\mathbf{P}_{:k}" not in theory
    assert r"\mathbf{Q}_{:k}" not in theory


def test_theory_page_exposes_the_canonical_fixed_construction() -> None:
    root = _repository_root()
    theory = (root / "docs" / "theory.md").read_text(encoding="utf-8")
    mathematics = (root / ".llm" / "mathematics.md").read_text(encoding="utf-8")

    sections = (
        "## Scientific source and package scope",
        "## Canonical terminology",
        "## 1. Rank-controlled predictor projection",
        "## 2. Covariance-driven response projection",
        "## 3. Least squares in the reduced coordinates",
        "## 4. Diagonal latent coupling",
        "## Why the method is panoramic",
        "## Nominal fitted dimension",
        "## Relationships to established methods",
        "## Package realization",
        "## Selection, validation, and synthetic-data boundaries",
    )
    positions = [theory.index(section) for section in sections]
    assert positions == sorted(positions)

    for required in (
        (
            r"\mathbf{X}=\mathbf{X}\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}"
            r"+\mathbf{X}(\mathbf{I}_p-\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T})"
        ),
        r"\max_{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h}",
        r"\mathbf{Y}\mathbf{Q}=\mathbf{X}\mathbf{P}\mathbf{D}+\mathbf{E}_\pi",
        r"=\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}",
        r"(r_\pi+q-h)h",
    ):
        assert required in theory

    for required in (
        r"\mathbf{X}_{\mathrm{cs}}\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}",
        r"\max_{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h}",
        r"\mathbf{Y}_{\mathrm{cs}}\mathbf{Q}",
        r"(r_\pi+q-h)h",
    ):
        assert required in mathematics


def test_theory_page_defines_canonical_pipls_terminology() -> None:
    root = _repository_root()
    theory = (root / "docs" / "theory.md").read_text(encoding="utf-8")
    mathematics = (root / ".llm" / "mathematics.md").read_text(encoding="utf-8")
    public_api = (root / ".llm" / "public_api.md").read_text(encoding="utf-8")

    for required in (
        "retained predictor basis",
        "retained-subspace projector",
        "orthonormal predictor directions",
        "orthonormal response directions",
        "$d_k=D_{kk}$",
        "paired latent mode",
        "`predictor_rank`",
        "`n_components`",
        (
            r"\mathbf{D}\mathbf{Q}^{\mathsf T}"
            r"=(\mathbf{Q}\mathbf{D})^{\mathsf T}"
        ),
        "`x_loadings_` and `y_loadings_`",
    ):
        assert required in theory

    assert "number of paired latent modes $h$" in public_api
    assert r"retained predictor-subspace dimension $r_\pi$" in public_api
    assert "orthonormal predictor directions" in mathematics
    assert "orthonormal response directions" in mathematics
    assert r"(\mathbf{Q}\mathbf{D})^{\mathsf T}" in mathematics



def test_canonical_terminology_is_propagated_to_public_entry_points() -> None:
    root = _repository_root()
    readme = (root / "README.md").read_text(encoding="utf-8")
    home = (root / "docs" / "index.md").read_text(encoding="utf-8")
    api_overview = (root / "docs" / "api" / "index.md").read_text(
        encoding="utf-8"
    )
    regression_page = (root / "docs" / "api" / "regression.md").read_text(
        encoding="utf-8"
    )
    inspection_page = (root / "docs" / "model_inspection.md").read_text(
        encoding="utf-8"
    )
    synthetic_tutorial = (root / "docs" / "tutorials" / "synthetic.md").read_text(
        encoding="utf-8"
    )
    pulp_tutorial = (root / "docs" / "tutorials" / "pulp.md").read_text(
        encoding="utf-8"
    )
    examples = (root / "examples" / "README.md").read_text(encoding="utf-8")

    for page in (readme, home, api_overview, synthetic_tutorial, examples):
        assert "paired latent modes" in page

    for page in (readme, home, api_overview, regression_page, pulp_tutorial):
        assert "predictor direction" in page
        assert "response direction" in page

    assert "retained predictor-subspace dimension" in api_overview
    assert r"$\mathbf{X}$" in api_overview
    assert r"$\mathbf{Y}$" in api_overview
    assert r"$\mathbf{P}$" in api_overview
    assert r"$\mathbf{Q}$" in api_overview
    assert "$d_k=D_{kk}$" in regression_page
    assert r"$\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$" in regression_page
    assert "$d_kQ_{:k}$" in inspection_page
    assert r"$\mathbf{Q}\mathbf{D}$" in inspection_page
    assert "predictor rotations" not in inspection_page
    assert "response rotations" not in inspection_page

    # Public identifiers remain stable even though prose names their values as directions.
    assert "`predictor_rotations`" in regression_page
    assert "`response_rotations`" in regression_page

    regression_doc = inspect.getdoc(PiPLSRegression) or ""
    decomposition_doc = inspect.getdoc(PiPLSDecomposition) or ""
    regression_map_doc = (
        inspect.getdoc(PiPLSDecomposition.standardized_regression_map) or ""
    )
    search_doc = inspect.getdoc(PiPLSSearchCV) or ""
    path_doc = inspect.getdoc(PiPLSComponentPath) or ""
    factors_doc = inspect.getdoc(PiPLSDisplayFactors) or ""

    assert "number of paired latent modes" in regression_doc
    assert "retained predictor-subspace dimension" in regression_doc
    assert r"Orthonormal predictor directions $\mathbf{P}$" in decomposition_doc
    assert r"Orthonormal response directions $\mathbf{Q}$" in decomposition_doc
    assert r"$\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$" in regression_map_doc
    assert "paired-mode count" in search_doc
    assert "paired-mode count" in path_doc
    assert "$d_k Q_{:k}$" in factors_doc


def test_reference_pages_use_bold_complete_matrices_and_plain_indexed_columns(
) -> None:
    root = _repository_root()
    reference_pages = (
        root / "docs" / "api" / "index.md",
        root / "docs" / "api" / "regression.md",
        root / "docs" / "api" / "path.md",
        root / "docs" / "api" / "inspection.md",
        root / "docs" / "model_inspection.md",
    )
    reference = "\n".join(
        path.read_text(encoding="utf-8") for path in reference_pages
    )

    for old in ("$P$", "$Q$", "$D$", "$QD$", r"$PDQ^{\mathsf T}$", "$Y$"):
        assert old not in reference
    for required in (
        r"$\mathbf{P}$",
        r"$\mathbf{Q}$",
        r"$\mathbf{D}$",
        r"$\mathbf{Q}\mathbf{D}$",
        r"$\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$",
        "$d_k=D_{kk}$",
        "$d_kQ_{:k}$",
        "$(P_{:k},d_k,Q_{:k})$",
    ):
        assert required in reference

    for invalid in (r"\mathbf{D}_{kk}", r"\mathbf{P}_{:k}", r"\mathbf{Q}_{:k}"):
        assert invalid not in reference


def test_companion_manuscript_synthetic_data_guide_is_public_and_scoped() -> None:
    root = _repository_root()
    guide = (root / "docs" / "manuscript_reproduction.md").read_text(
        encoding="utf-8"
    )
    with (root / "mkdocs.yml").open(encoding="utf-8") as stream:
        navigation = yaml.safe_load(stream)["nav"]

    scientific_background = next(
        item["Scientific background"]
        for item in navigation
        if "Scientific background" in item
    )
    assert {
        "Companion manuscript synthetic data": "manuscript_reproduction.md"
    } in scientific_background

    sections = (
        "## Three reproducibility levels",
        "## Generate the manuscript distribution",
        "## Verify the stored latent geometry",
        "## Record one deterministic realization",
        "## Known dimensions in the synthetic experiments",
        "## What this guide does not reproduce",
        "## Relationship to ordinary package workflows",
    )
    positions = [guide.index(section) for section in sections]
    assert positions == sorted(positions)

    for required in (
        "make_pipls_latent_geometry",
        r"\boldsymbol{\Lambda}_{\mathrm{p}}\mathbf{L}_{\mathrm{p}}",
        r"\boldsymbol{\Lambda}_{\mathrm{s}}\mathbf{L}_{\mathrm{sr}}",
        r"\boldsymbol{\varepsilon}_{\mathrm{X}}",
        r"\boldsymbol{\varepsilon}_{\mathrm{Y}}",
        r"r_\pi=d_{\mathrm{p}}+d_{\mathrm{s}}",
        r"h=d_{\mathrm{s}}",
        "random_state=0",
        "PiPLSSearchCV",
        "downstream reproduction repository",
    ):
        assert required in guide

    assert "real-data rank-selection rule" in guide
    assert "**Complete manuscript results.**" in guide
    assert "complete simulation grid" in guide

    cross_link_sources = (
        root / "README.md",
        root / "docs" / "index.md",
        root / "docs" / "theory.md",
        root / "docs" / "reproducibility.md",
        root / "docs" / "citation.md",
        root / "docs" / "api" / "datasets.md",
    )
    for source in cross_link_sources:
        assert "manuscript_reproduction.md" in source.read_text(encoding="utf-8")


def test_mathematical_typography_and_descriptive_subscripts_are_consistent() -> None:
    root = _repository_root()
    dataset_page = (root / "docs" / "datasets.md").read_text(encoding="utf-8")
    manuscript_page = (root / "docs" / "manuscript_reproduction.md").read_text(
        encoding="utf-8"
    )
    theory_page = (root / "docs" / "theory.md").read_text(encoding="utf-8")
    path_page = (root / "docs" / "path_analysis.md").read_text(encoding="utf-8")
    mathematics = (root / ".llm" / "mathematics.md").read_text(encoding="utf-8")

    for page in (dataset_page, manuscript_page):
        for required in (
            r"\mathbf{X}",
            r"\mathbf{Y}",
            r"\boldsymbol{\Lambda}_{\mathrm{p}}",
            r"\boldsymbol{\Lambda}_{\mathrm{s}}",
            r"\boldsymbol{\Lambda}_{\mathrm{r}}",
            r"\mathbf{L}_{\mathrm{sp}}",
            r"\mathbf{L}_{\mathrm{sr}}",
            r"\boldsymbol{\varepsilon}_{\mathrm{X}}",
            r"\boldsymbol{\varepsilon}_{\mathrm{Y}}",
        ):
            assert required in page
        for obsolete in (
            r"\Lambda_p",
            r"\Lambda_s",
            r"\Lambda_r",
            r"L_{sp}",
            r"L_{sr}",
            r"\varepsilon_X",
            r"\varepsilon_Y",
        ):
            assert obsolete not in page

    for required in (
        r"\mathbf{U}_{\mathrm{X}}",
        r"\mathbf{S}_{\mathrm{X}}",
        r"\mathbf{V}_{\mathrm{X}}",
        r"\boldsymbol{\Sigma}_{\mathrm{ZY}}",
        r"\|_{\mathrm{F}}",
    ):
        assert required in theory_page

    for required in (
        r"h_{\mathrm{max}}",
        r"r_{\pi,\mathrm{max}}",
        r"p_{\mathrm{min}}",
        r"h_{\mathrm{min}}",
    ):
        assert required in path_page


    for variable_index in ("d_k=D_{kk}", "P_{:k}", "Q_{:k}", "s_i"):
        assert variable_index in mathematics
    assert "Descriptive, role, block, method, and extremum subscripts" in mathematics


def test_generated_dataset_equations_use_markdown_math_and_canonical_notation() -> None:
    root = _repository_root()
    source = (root / "src" / "pipls" / "datasets.py").read_text(encoding="utf-8")
    truth_doc = inspect.getdoc(PiPLSLatentGeometryTruth) or ""
    generator_doc = inspect.getdoc(make_pipls_latent_geometry) or ""

    for path in (root / "src" / "pipls").glob("*.py"):
        assert ".. math::" not in path.read_text(encoding="utf-8"), path

    for path in (root / "docs").rglob("*.md"):
        if "decisions" not in path.parts:
            text = path.read_text(encoding="utf-8")
            assert "\n\\[\n" not in text, path
            assert "\n\\]\n" not in text, path

    assert ".. math::" not in source
    for docstring in (truth_doc, generator_doc):
        assert r"\begin{equation}" in docstring
        assert r"\end{equation}" in docstring
        assert r"\boldsymbol{\Lambda}_{\mathrm{p}}" in docstring
        assert r"\mathbf{L}_{\mathrm{sp}}" in docstring
        assert r"\boldsymbol{\varepsilon}_{\mathrm{X}}" in docstring
        assert ":math:" not in docstring
