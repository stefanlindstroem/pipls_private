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
            "pipls.datasets",
            "pipls.inspection",
            "pipls.metrics",
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
        "pipls.PiPLSComponentPath",
        "pipls.PiPLSComponentResult",
        "pipls.PiPLSPredictorRankProfile",
        "pipls.PiPLSValidationReport",
        "pipls.PiPLSDecomposition",
        "pipls.inspection.BiplotCoordinates",
        "pipls.inspection.LatentStructure",
        "pipls.inspection.ObservationDiagnostics",
        "pipls.inspection.PiPLSDisplayFactors",
        "pipls.inspection.PredictionDiagnostics",
        "pipls.datasets.PiPLSLatentGeometryTruth",
        "pipls.datasets.PiPLSSyntheticTruth",
    }

    for object_path in returned_records:
        assert "show_signature: false" in _directive_options(api_text, object_path)

    dataset_options = _directive_options(api_text, "pipls.datasets.PiPLSDataset")
    assert "show_signature: false" not in dataset_options


def test_fixed_regression_reference_covers_output_configuration_and_rank_diagnostics() -> None:
    regression = (
        _repository_root() / "docs" / "api" / "regression.md"
    ).read_text(encoding="utf-8")
    options = _directive_options(regression, "pipls.PiPLSRegression")

    assert "- set_output" in options
    for field in (
        "predictor_numerical_rank",
        "predictor_numerical_rank_is_exact",
        "rank_tolerance",
        "predictor_svd_solver",
    ):
        assert f"`{field}`" in regression


def test_component_path_reference_lists_recommendation_methods() -> None:
    path_page = (
        _repository_root() / "docs" / "api" / "path.md"
    ).read_text(encoding="utf-8")
    options = _directive_options(path_page, "pipls.PiPLSComponentPath")

    assert "- minimum_cv_mse_result" in options
    assert "- one_standard_error_result" in options
