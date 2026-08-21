from __future__ import annotations

import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")

CANONICAL_ANCHORS = {
    "datasets.md": (
        "pulp-real-data-integration",
        "sugarcane-spectral-integration",
        "tobacco-spectral-integration",
    ),
    "citation.md": ("companion-paper",),
    "theory.md": (
        "problem-setting-and-two-rank-controls",
        "canonical-terminology",
        "rank-controlled-predictor-projection",
        "response-subspace-selection",
        "diagonal-latent-coupling",
        "why-the-method-is-panoramic",
        "interpretation-of-the-ranks",
        "relationships-to-established-methods",
        "ordinary-least-squares",
        "reduced-rank-regression",
        "canonical-correlation-analysis",
        "pls-and-pls-svd",
        "selection-validation-and-synthetic-data-boundaries",
    ),
}

DATASET_TARGETS = {
    "pulp": ("datasets.md", "pulp-real-data-integration"),
    "sugarcane": ("datasets.md", "sugarcane-spectral-integration"),
    "tobacco": ("datasets.md", "tobacco-spectral-integration"),
}

DATASET_PAGE_REQUIREMENTS = {
    "index.md": ("pulp", "tobacco"),
    "examples.md": ("pulp", "sugarcane", "tobacco"),
    "api/datasets.md": ("pulp", "sugarcane", "tobacco"),
    "api/index.md": ("pulp", "sugarcane", "tobacco"),
    "api/path.md": ("tobacco",),
    "path_analysis.md": ("tobacco",),
    "reproducibility.md": ("pulp", "sugarcane", "tobacco"),
    "theory.md": ("pulp",),
    "tutorials/pulp.md": ("pulp",),
    "tutorials/quick_start.md": ("pulp",),
}

PUBLICATION_PAGE_REQUIREMENTS = (
    "index.md",
    "api/path.md",
    "api/regression.md",
    "computational_performance.md",
    "datasets.md",
    "examples.md",
    "manuscript_reproduction.md",
    "reproducibility.md",
    "theory.md",
    "troubleshooting.md",
)

THEORY_PAGE_REQUIREMENTS = {
    "index.md": ("diagonal-latent-coupling",),
    "api/index.md": ("interpretation-of-the-ranks", "canonical-terminology"),
    "api/inspection.md": ("diagonal-latent-coupling",),
    "api/path.md": ("interpretation-of-the-ranks", "response-subspace-selection"),
    "api/regression.md": (
        "interpretation-of-the-ranks",
        "canonical-terminology",
        "response-subspace-selection",
        "diagonal-latent-coupling",
    ),
    "computational_performance.md": (
        "rank-controlled-predictor-projection",
        "response-subspace-selection",
    ),
    "examples.md": ("response-subspace-selection",),
    "manuscript_reproduction.md": ("response-subspace-selection", "interpretation-of-the-ranks"),
    "model_inspection.md": ("diagonal-latent-coupling",),
    "path_analysis.md": ("interpretation-of-the-ranks",),
    "reproducibility.md": ("response-subspace-selection",),
    "troubleshooting.md": ("interpretation-of-the-ranks", "response-subspace-selection"),
    "tutorials/pulp.md": ("interpretation-of-the-ranks", "diagonal-latent-coupling"),
    "tutorials/synthetic.md": ("interpretation-of-the-ranks",),
}


def _page(relative_path: str) -> Path:
    return DOCS_ROOT / relative_path


def _resolved_doc_targets(relative_path: str) -> set[tuple[str, str]]:
    page = _page(relative_path)
    targets: set[tuple[str, str]] = set()

    for raw_target in MARKDOWN_LINK.findall(page.read_text(encoding="utf-8")):
        target = raw_target.strip()
        if target.startswith(("http://", "https://", "mailto:")):
            continue

        path_text, separator, anchor = target.partition("#")
        if path_text:
            destination = (page.parent / path_text).resolve()
        else:
            destination = page.resolve()

        try:
            relative_destination = destination.relative_to(DOCS_ROOT.resolve())
        except ValueError:
            continue
        if relative_destination.suffix != ".md" or not destination.is_file():
            continue

        targets.add((relative_destination.as_posix(), anchor if separator else ""))

    return targets


def test_canonical_cross_reference_anchors_are_explicit_and_unique() -> None:
    for relative_path, anchors in CANONICAL_ANCHORS.items():
        text = _page(relative_path).read_text(encoding="utf-8")
        for anchor in anchors:
            assert text.count(f"{{#{anchor}}}") == 1, (relative_path, anchor)


def test_reference_dataset_pages_link_to_canonical_dataset_sections() -> None:
    for relative_path, datasets in DATASET_PAGE_REQUIREMENTS.items():
        targets = _resolved_doc_targets(relative_path)
        for dataset in datasets:
            assert DATASET_TARGETS[dataset] in targets, (relative_path, dataset)


def test_companion_publication_pages_link_to_canonical_citation_section() -> None:
    expected = ("citation.md", "companion-paper")
    for relative_path in PUBLICATION_PAGE_REQUIREMENTS:
        assert expected in _resolved_doc_targets(relative_path), relative_path


def test_technical_pages_link_to_specific_theory_sections() -> None:
    for relative_path, anchors in THEORY_PAGE_REQUIREMENTS.items():
        targets = _resolved_doc_targets(relative_path)
        for anchor in anchors:
            assert ("theory.md", anchor) in targets, (relative_path, anchor)


def test_contextual_navigation_routes_remain_connected() -> None:
    requirements = {
        "compatibility.md": ("reproducibility.md", "installed-distribution-reproducibility"),
        "computational_performance.md": ("path_analysis.md", ""),
        "datasets.md": ("api/datasets.md", ""),
    }
    for relative_path, expected in requirements.items():
        assert expected in _resolved_doc_targets(relative_path), relative_path


def test_every_served_markdown_page_links_to_another_documentation_page() -> None:
    served_pages = sorted(
        path
        for path in DOCS_ROOT.rglob("*.md")
        if "decisions" not in path.relative_to(DOCS_ROOT).parts
    )

    for page in served_pages:
        relative_path = page.relative_to(DOCS_ROOT).as_posix()
        assert any(
            destination != relative_path
            for destination, _anchor in _resolved_doc_targets(relative_path)
        ), relative_path
