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
        "PiPLSSyntheticTruth",
    }

    for object_name in result_objects:
        assert f"`{object_name}`" in api_overview


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
        "api/exceptions.md",
        "path_analysis.md",
        "cross_validation.md",
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
