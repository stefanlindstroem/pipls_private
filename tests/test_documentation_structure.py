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
        "PiPLSRegressionTruth",
    }

    for object_name in result_objects:
        assert f"`{object_name}`" in api_overview


def test_required_public_guides_are_reachable_through_navigation() -> None:
    root = _repository_root()
    with (root / "mkdocs.yml").open(encoding="utf-8") as stream:
        navigation = yaml.safe_load(stream)["nav"]

    navigation_paths = _navigation_paths(navigation)
    required_paths = {
        "tutorials/synthetic.md",
        "tutorials/pulp.md",
        "examples.md",
        "api/index.md",
        "api/regression.md",
        "api/path.md",
        "path_analysis.md",
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
