from __future__ import annotations

import importlib
import re
from pathlib import Path

import pipls

_TOP_LEVEL_DIRECTIVE = re.compile(
    r"^::: pipls\.([A-Za-z_][A-Za-z0-9_]*)$",
    re.MULTILINE,
)
_SUBMODULE_DIRECTIVE = re.compile(
    r"^::: (pipls\.[A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$",
    re.MULTILINE,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _api_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((_repository_root() / "docs" / "api").glob("*.md"))
    )


def _directive_options(text: str, object_path: str) -> str:
    marker = f"::: {object_path}\n"
    _, remainder = text.split(marker, maxsplit=1)
    option_lines: list[str] = []
    for line in remainder.splitlines():
        if line and not line.startswith(" "):
            break
        option_lines.append(line)
    return "\n".join(option_lines)


def test_core_api_pages_cover_the_top_level_public_objects() -> None:
    documented = set(_TOP_LEVEL_DIRECTIVE.findall(_api_text()))
    expected = set(pipls.__all__) - {"__version__"}

    assert documented == expected


def test_submodule_api_pages_cover_declared_public_objects() -> None:
    documented: dict[str, set[str]] = {}
    for module_name, object_name in _SUBMODULE_DIRECTIVE.findall(_api_text()):
        documented.setdefault(module_name, set()).add(object_name)

    expected = {
        module_name: set(importlib.import_module(module_name).__all__)
        for module_name in (
            "pipls.component_path",
            "pipls.datasets",
            "pipls.decomposition",
            "pipls.inspection",
            "pipls.metrics",
            "pipls.validation",
        )
    }

    assert documented == expected


def test_api_pages_reference_only_public_import_paths() -> None:
    api_text = _api_text()

    assert "::: pipls._" not in api_text
    assert "show_source: true" not in api_text
    assert "pipls.model_selection" not in api_text


def test_returned_result_records_hide_constructor_signatures() -> None:
    api_text = _api_text()
    returned_records = {
        "pipls.component_path.PiPLSComponentPath",
        "pipls.component_path.PiPLSPredictorRankEvidence",
        "pipls.component_path.PiPLSSelection",
        "pipls.component_path.PiPLSPredictorRankProfile",
        "pipls.validation.PiPLSOOFReport",
        "pipls.decomposition.PiPLSDecomposition",
        "pipls.inspection.BiplotCoordinates",
        "pipls.inspection.LatentStructure",
        "pipls.inspection.ObservationDiagnostics",
        "pipls.inspection.PiPLSDisplayFactors",
        "pipls.inspection.PredictionDiagnostics",
        "pipls.datasets.PiPLSLatentGeometryTruth",
        "pipls.datasets.PiPLSRegressionTruth",
    }

    for object_path in returned_records:
        assert "show_signature: false" in _directive_options(api_text, object_path)

    dataset_options = _directive_options(api_text, "pipls.datasets.PiPLSDataset")
    assert "show_signature: false" not in dataset_options
