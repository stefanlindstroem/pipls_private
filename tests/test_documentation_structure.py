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


def test_path_reference_defines_resolved_ceilings_before_rank_policies() -> None:
    page = (_repository_root() / "docs" / "path_analysis.md").read_text(
        encoding="utf-8"
    )

    bounds = page.index("## Search bounds")
    ceilings = page.index("### Resolved ceilings")
    component_requests = page.index("### Component-count requests")
    rank_policies = page.index("## Predictor-rank policies")
    assert bounds < ceilings < component_requests < rank_policies
    assert r"r_{\pi,\max}" in page[ceilings:component_requests]
    assert r"h_{\max}" in page[ceilings:component_requests]
