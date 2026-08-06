from __future__ import annotations

import ast
import inspect
import re
import unicodedata
from pathlib import Path

from pipls import PiPLSRegression, PiPLSSearchCV
from tests._mkdocs import load_mkdocs_config

_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target>[^)]+)\)")
_HEADING = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$", re.MULTILINE)
_EXPLICIT_ANCHOR = re.compile(r"\{\s*#(?P<anchor>[A-Za-z0-9_.:-]+)\s*\}")
_PYTHON_BLOCK = re.compile(r"```python\n(.*?)\n```", flags=re.DOTALL)
_MERMAID_BLOCK = re.compile(r"```mermaid\n(?P<body>.*?)\n```", flags=re.DOTALL)
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


def _navigation_paths(items: list[object]) -> set[str]:
    paths: set[str] = set()
    for item in items:
        if isinstance(item, str):
            paths.add(item)
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str):
                    paths.add(value)
                elif isinstance(value, list):
                    paths.update(_navigation_paths(value))
    return paths


def _linked_paths(path: Path) -> set[str]:
    return {
        parsed[0]
        for match in _MARKDOWN_LINK.finditer(path.read_text(encoding="utf-8"))
        if (parsed := _local_target(match.group("target"))) is not None
    }


def _python_calls(path: Path) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for block in _PYTHON_BLOCK.findall(path.read_text(encoding="utf-8")):
        tree = ast.parse(block)
        calls.extend(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    return calls


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _keyword_map(call: ast.Call) -> dict[str, ast.expr]:
    assert all(keyword.arg is not None for keyword in call.keywords)
    return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}


def _integer_literal(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    )


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
        "PiPLSPredictorRankEvidence",
        "PiPLSSelection",
        "PiPLSPredictorRankProfile",
        "PiPLSOOFReport",
        "PiPLSDecomposition",
        "LatentStructure",
        "PiPLSDisplayFactors",
        "BiplotCoordinates",
        "ObservationDiagnostics",
        "PredictionDiagnostics",
        "PiPLSDataset",
        "PiPLSLatentGeometryTruth",
        "PiPLSRegressionTruth",
    }

    for object_name in result_objects:
        assert f"`{object_name}`" in api_overview


def test_served_tutorials_have_one_vertical_workflow_flowchart() -> None:
    tutorial_root = _repository_root() / "docs" / "tutorials"
    expected = {
        "quick_start.md": {
            "labels": (
                "Load Pulp data",
                "Search candidate models",
                "Select by rule and refit",
                "Inspect fitted values",
            ),
            "prose": (
                "load the Pulp data",
                "search the candidate models",
                "select by rule and refit",
                "inspect fitted values",
            ),
            "feedback": False,
        },
        "synthetic.md": {
            "labels": (
                "Generate training and test data",
                "Fit search",
                "Inspect component path",
                "Choose component count and create selection",
                "Inspect selected path and conditional rank profile",
                "Refit the same selection",
                "Predict external test data",
                "revise if dissatisfied",
            ),
            "prose": (
                "generate independent training and test data",
                "fit the search",
                "inspect the component path",
                "choose a component count and create one selection",
                "inspect the selected path and conditional predictor-rank profile",
                "refit the same selection",
                "predict the external test data",
                "return to the selection step",
            ),
            "feedback": True,
        },
        "pulp.md": {
            "labels": (
                "Load Pulp data",
                "Fit search",
                "Inspect component path",
                "Choose component count and create selection",
                "Inspect selected path, conditional rank profile, and OOF predictions",
                "Refit the same selection",
                "Inspect the fitted model",
                "Render reports",
                "revise if dissatisfied",
            ),
            "prose": (
                "load the Pulp data",
                "fit the search",
                "inspect the component path",
                "choose a component count and create one selection",
                "inspect the selected path, conditional predictor-rank profile, and OOF predictions",
                "refit the same selection",
                "inspect the fitted model",
                "render the reports",
                "return to the selection step",
            ),
            "feedback": True,
        },
    }

    for filename, contract in expected.items():
        text = (tutorial_root / filename).read_text(encoding="utf-8")
        blocks = list(_MERMAID_BLOCK.finditer(text))
        assert len(blocks) == 1
        body = blocks[0].group("body")
        assert body.splitlines()[0] in {"flowchart TD", "flowchart TB"}
        assert all(label in body for label in contract["labels"])
        if contract["feedback"]:
            assert "-. revise if dissatisfied .->" in body

        prose = text[: blocks[0].start()] + text[blocks[0].end() :]
        normalized_prose = " ".join(prose.split())
        assert all(fragment in normalized_prose for fragment in contract["prose"])
        first_section = text.index("\n## ")
        assert blocks[0].start() < first_section


def test_model_producing_tutorials_state_selection_ownership_positively() -> None:
    tutorial_root = _repository_root() / "docs" / "tutorials"

    for filename in ("synthetic.md", "pulp.md"):
        text = (tutorial_root / filename).read_text(encoding="utf-8")

        normalized = " ".join(text.split())
        assert "model.selection_" in text
        assert "without fitting a final model" in normalized
        assert re.search(
            r"`search\.select\(\)`[^.\n]*\bnot required\b",
            text,
        ) is None


def test_selection_conditioned_oof_is_described_as_inspection_not_qualification() -> None:
    root = _repository_root()
    active_guides = (
        root / "README.md",
        root / "docs" / "index.md",
        root / "docs" / "examples.md",
        root / "docs" / "tutorials" / "pulp.md",
        root / "examples" / "README.md",
    )
    stale = (
        "qualify it with OOF predictions",
        "qualify one manual Pulp selection",
        "optionally qualify",
        "OOF qualification",
        "qualified selection",
    )

    for path in active_guides:
        text = path.read_text(encoding="utf-8")
        assert all(phrase not in text for phrase in stale), path

    pulp = (root / "docs" / "tutorials" / "pulp.md").read_text(encoding="utf-8")
    assert "not independent qualification" in pulp
    assert "return to the selection step" in pulp


def test_active_workflow_guides_use_pre_refit_selection_handoff() -> None:
    root = _repository_root()
    analytical_guides = (
        root / "README.md",
        root / "docs" / "examples.md",
        root / "docs" / "path_analysis.md",
        root / "docs" / "tutorials" / "quick_start.md",
        root / "examples" / "README.md",
    )
    stale_phrases = (
        "selection = model.selection_",
        "selection=model.selection_",
        "Modeling completes before",
        "complete modeling before",
        "completes modeling before",
        "Search and refitting complete modeling",
        "complete search and full-data refitting before",
        "workflows obtain the fitted row from `model.selection_`",
    )

    for path in analytical_guides:
        text = path.read_text(encoding="utf-8")
        assert "selection = search.select" in text, path
        assert "selection=selection" in text, path
        assert all(phrase not in text for phrase in stale_phrases), path

    dataset_guide = (root / "docs" / "datasets.md").read_text(encoding="utf-8")
    assert all(phrase not in dataset_guide for phrase in stale_phrases)


def test_required_public_guides_are_reachable_through_navigation() -> None:
    root = _repository_root()
    navigation = load_mkdocs_config(root / "mkdocs.yml")["nav"]

    navigation_paths = _navigation_paths(navigation)
    required_paths = {
        "tutorials/quick_start.md",
        "tutorials/synthetic.md",
        "tutorials/pulp.md",
        "examples.md",
        "api/index.md",
        "api/regression.md",
        "api/path.md",
        "path_analysis.md",
        "computational_performance.md",
        "troubleshooting.md",
        "model_inspection.md",
        "api/inspection.md",
        "api/datasets.md",
        "datasets.md",
        "reproducibility.md",
        "compatibility.md",
        "theory.md",
        "manuscript_reproduction.md",
        "citation.md",
    }

    assert required_paths <= navigation_paths
    assert all((root / "docs" / relative).is_file() for relative in navigation_paths)


def test_computational_performance_guide_has_reference_position_and_structure() -> None:
    root = _repository_root()
    navigation = load_mkdocs_config(root / "mkdocs.yml")["nav"]

    reference = next(item["Reference"] for item in navigation if "Reference" in item)
    labels = [next(iter(item)) for item in reference]
    assert labels.index("Computational performance") == labels.index("Path-selection details") + 1
    assert labels.index("Troubleshooting") == labels.index("Computational performance") + 1

    page = (root / "docs" / "computational_performance.md").read_text(encoding="utf-8")
    required_headings = {
        "A cost model for path training",
        "Develop with a smaller validation protocol",
        "Choose predictor-rank coverage deliberately",
        "Use fixed or restricted predictor-rank policies when justified",
        "Restrict the component path when justified",
        "Use randomized predictor SVD for large problems",
        "Use parallelism deliberately",
        "Avoid repeated OOF computation",
        "Inspect the work performed",
        "Separate development and final-analysis workflows",
        "Summary of trade-offs",
    }
    headings = {match.group("title") for match in _HEADING.finditer(page)}
    assert required_headings <= headings


def test_computational_performance_guide_is_linked_from_user_routes() -> None:
    root = _repository_root()
    expected = {
        root / "README.md": "docs/computational_performance.md",
        root / "docs" / "api" / "path.md": "../computational_performance.md",
        root / "docs" / "api" / "regression.md": "../computational_performance.md",
        root / "docs" / "examples.md": "computational_performance.md",
        root / "docs" / "tutorials" / "pulp.md": "../computational_performance.md",
        root / "docs" / "troubleshooting.md": "computational_performance.md",
    }

    for source, target in expected.items():
        assert target in _linked_paths(source), source


def test_computational_performance_examples_match_public_constructor_contracts() -> None:
    page = _repository_root() / "docs" / "computational_performance.md"
    signatures = {
        "PiPLSRegression": set(inspect.signature(PiPLSRegression).parameters),
        "PiPLSSearchCV": set(inspect.signature(PiPLSSearchCV).parameters),
    }

    for call in _python_calls(page):
        name = _call_name(call)
        keywords = _keyword_map(call)
        if name in signatures:
            assert set(keywords) <= signatures[name]

        if name == "PiPLSSearchCV":
            method = keywords.get("search_method")
            if method is not None:
                assert isinstance(method, ast.Constant)
                assert method.value in {"adaptive", "exhaustive"}

            ranks = keywords.get("predictor_rank_values")
            one_candidate = (isinstance(ranks, ast.Constant) and ranks.value == "max") or (
                isinstance(ranks, (ast.List, ast.Tuple)) and len(ranks.elts) == 1
            )
            if one_candidate:
                assert "search_method" not in keywords

        if name == "PiPLSRegression":
            solver = keywords.get("svd_solver")
            if isinstance(solver, ast.Constant) and solver.value == "randomized":
                assert _integer_literal(keywords.get("random_state"))

        if name == "KFold":
            shuffle = keywords.get("shuffle")
            if isinstance(shuffle, ast.Constant) and shuffle.value is True:
                assert _integer_literal(keywords.get("random_state"))

        if name == "RepeatedKFold":
            assert _integer_literal(keywords.get("random_state"))


def test_computational_performance_guide_separates_work_counts_from_wall_time() -> None:
    page = (_repository_root() / "docs" / "computational_performance.md").read_text(
        encoding="utf-8"
    )

    for result_key in (
        'cv_results_["n_components"]',
        'cv_results_["mean_fit_time"]',
        'cv_results_["std_fit_time"]',
        'cv_results_["mean_score_time"]',
        'cv_results_["std_score_time"]',
    ):
        assert result_key in page
    assert "wall time" in page.lower()
    assert ".fit_transform(" not in page


def test_dataset_api_routes_to_repository_reference_material() -> None:
    page = _repository_root() / "docs" / "api" / "datasets.md"

    assert {"../datasets.md", "../examples.md"} <= _linked_paths(page)


def test_troubleshooting_routes_to_stable_programming_references() -> None:
    root = _repository_root()
    page = root / "docs" / "troubleshooting.md"

    assert {
        "api/regression.md",
        "api/path.md",
        "computational_performance.md",
        "path_analysis.md",
    } <= _linked_paths(page)
    text = page.read_text(encoding="utf-8")
    assert "assets/generated/" not in text
    assert "--8<--" not in text


def test_served_guides_do_not_expose_internal_phase_or_patch_labels() -> None:
    docs_root = _repository_root() / "docs"
    internal_label = re.compile(r"\b(?:Phase [A-Z][A-Za-z0-9.-]*|Patch D\d+)\b")

    for path in sorted(docs_root.rglob("*.md")):
        if "decisions" in path.relative_to(docs_root).parts:
            continue
        assert internal_label.search(path.read_text(encoding="utf-8")) is None, path


def test_example_catalogue_matches_numbered_scripts() -> None:
    root = _repository_root()
    page = (root / "docs" / "examples.md").read_text(encoding="utf-8")
    rows = [
        line
        for line in page.splitlines()
        if line.startswith("| `") and line.endswith("|")
    ]

    documented_scripts: list[str] = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert len(cells) == 3
        assert all(cells)
        documented_scripts.append(cells[0].strip("`"))

    expected_scripts = [
        path.name for path in sorted((root / "examples").glob("[0-9][0-9]_*.py"))
    ]
    assert documented_scripts == expected_scripts


def test_companion_manuscript_guide_routes_to_public_theory_and_generator_api() -> None:
    guide = _repository_root() / "docs" / "manuscript_reproduction.md"

    assert {"theory.md", "api/datasets.md", "citation.md"} <= _linked_paths(guide)


def _snippet_sections(source: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    active: str | None = None
    for line in source.splitlines():
        start = re.fullmatch(r"\s*# --8<-- \[start:([^]]+)]", line)
        end = re.fullmatch(r"\s*# --8<-- \[end:([^]]+)]", line)
        if start is not None:
            active = start.group(1)
            sections[active] = []
        elif end is not None:
            assert active == end.group(1)
            active = None
        elif active is not None:
            sections[active].append(line)
    return {name: "\n".join(lines) for name, lines in sections.items()}


def test_rendered_tutorials_show_explicit_kfold_definitions() -> None:
    repository = _repository_root()
    snippet_pattern = re.compile(r'--8<-- "(examples/[^:"]+):([^"]+)"')

    for tutorial_path in sorted((repository / "docs" / "tutorials").glob("*.md")):
        tutorial = tutorial_path.read_text(encoding="utf-8")
        references = snippet_pattern.findall(tutorial)
        by_source: dict[str, set[str]] = {}
        for source_name, section in references:
            by_source.setdefault(source_name, set()).add(section)

        for source_name, referenced_sections in by_source.items():
            source = (repository / source_name).read_text(encoding="utf-8")
            if re.search(r"^CV\s*=\s*KFold\(", source, flags=re.MULTILINE) is None:
                continue

            sections = _snippet_sections(source)
            visible_source = "\n".join(
                sections[section]
                for section in referenced_sections
                if section in sections
            )
            assert "from sklearn.model_selection import KFold" in visible_source
            assert "CV = KFold(" in visible_source


def test_public_python_blocks_do_not_use_an_undefined_cv_variable() -> None:
    repository = _repository_root()
    code_fence = re.compile(r"```python\n(.*?)\n```", flags=re.DOTALL)

    for document in sorted((repository / "docs").rglob("*.md")):
        if "decisions" in document.parts:
            continue
        for block in code_fence.findall(document.read_text(encoding="utf-8")):
            if "cv=cv" in block:
                assert re.search(r"^cv\s*=", block, flags=re.MULTILINE) is not None


def test_current_predictor_rank_search_terminology_is_consistent() -> None:
    root = _repository_root()
    current_documents = [
        root / "README.md",
        *sorted(
            path
            for path in (root / "docs").rglob("*.md")
            if "decisions" not in path.relative_to(root / "docs").parts
        ),
        *sorted((root / "examples").glob("[0-9][0-9]_*.py")),
        *sorted((root / "tools").glob("*.py")),
        root / "src" / "pipls" / "search.py",
        root / "src" / "pipls" / "_model_selection.py",
    ]
    retired_assignment = re.compile(
        r"search_method\s*=\s*[\"'](?:auto|optimal)[\"']"
    )

    for path in current_documents:
        assert retired_assignment.search(path.read_text(encoding="utf-8")) is None, path

    retained_decisions = sorted(
        path
        for path in (root / "docs" / "decisions").glob("[0-9][0-9][0-9][0-9]-*.md")
        if not path.name.startswith("0149-")
    )
    for path in retained_decisions:
        assert retired_assignment.search(path.read_text(encoding="utf-8")) is None, path

    regression_text = (root / "src" / "pipls" / "regression.py").read_text(
        encoding="utf-8"
    )
    assert 'svd_solver : {"auto", "full", "randomized"}' in regression_text
