from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import yaml

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


def test_model_producing_tutorials_state_selection_ownership_positively() -> None:
    tutorial_root = _repository_root() / "docs" / "tutorials"

    for filename in ("synthetic.md", "pulp.md"):
        text = (tutorial_root / filename).read_text(encoding="utf-8")

        assert "model.selection_" in text
        assert "without fitting a final model" in text
        assert re.search(
            r"`search\.select\(\)`[^.\n]*\bnot required\b",
            text,
        ) is None


def test_required_public_guides_are_reachable_through_navigation() -> None:
    root = _repository_root()
    with (root / "mkdocs.yml").open(encoding="utf-8") as stream:
        navigation = yaml.safe_load(stream)["nav"]

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
    with (root / "mkdocs.yml").open(encoding="utf-8") as stream:
        navigation = yaml.safe_load(stream)["nav"]

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


def test_dataset_api_routes_to_repository_reference_material() -> None:
    page = _repository_root() / "docs" / "api" / "datasets.md"

    assert {"../datasets.md", "../examples.md"} <= _linked_paths(page)


def test_troubleshooting_routes_to_stable_programming_references() -> None:
    root = _repository_root()
    page = root / "docs" / "troubleshooting.md"

    assert {"api/regression.md", "api/path.md", "path_analysis.md"} <= _linked_paths(page)
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
