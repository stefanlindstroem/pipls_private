from __future__ import annotations

import ast
from pathlib import Path

import pipls.inspection as inspection
import pipls.plotting as plotting


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _imports_name(path: Path, name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == name for alias in node.names):
                return True
        elif isinstance(node, ast.Import):
            if any(alias.name == name for alias in node.names):
                return True
    return False


def _calls_name(path: Path, name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        for node in ast.walk(tree)
    )


def test_numbered_examples_do_not_import_or_fit_ordinary_pls() -> None:
    examples_dir = _repository_root() / "examples"
    numbered = sorted(examples_dir.glob("[0-9][0-9]_*.py"))
    assert numbered

    for path in numbered:
        assert not _imports_name(path, "PLSRegression"), path
        assert not _calls_name(path, "PLSRegression"), path


def test_real_data_examples_keep_the_pls_comparison_path_only() -> None:
    examples_dir = _repository_root() / "examples"
    for filename in (
        "10_pulp_real_data.py",
        "11_sugarcane_real_data.py",
        "12_tobacco_real_data.py",
    ):
        text = (examples_dir / filename).read_text(encoding="utf-8")
        assert "evaluate_pls_component_path(" in text
        assert "latent_structure(pipls_model)" in text
        assert "PiPLSRegression(" in text


def test_ordinary_pls_is_confined_to_the_example_comparison_helper() -> None:
    examples_dir = _repository_root() / "examples"
    users = [
        path.relative_to(examples_dir).as_posix()
        for path in examples_dir.rglob("*.py")
        if _imports_name(path, "PLSRegression") or _calls_name(path, "PLSRegression")
    ]
    assert users == ["_support/pls_component_path.py"]


def test_shared_public_analysis_names_are_estimator_neutral() -> None:
    shared_inspection = {
        "BiplotCoordinates",
        "LatentStructure",
        "ObservationDiagnostics",
        "PredictionDiagnostics",
        "PredictionKind",
        "biplot_coordinates",
        "latent_structure",
        "observation_diagnostics",
        "prediction_diagnostics",
    }
    pipls_specific_inspection = {"PiPLSDisplayFactors", "pipls_display_factors"}
    assert set(inspection.__all__) == shared_inspection | pipls_specific_inspection

    shared_plotting = {
        "PredictorStyle",
        "plot_biplot",
        "plot_coefficients",
        "plot_observation_diagnostics",
        "plot_prediction_diagnostics",
        "plot_scores",
        "plot_x_loadings",
        "plot_y_loadings",
    }
    pipls_specific_plotting = {"plot_pipls_decomposition"}
    assert set(plotting.__all__) == shared_plotting | pipls_specific_plotting

    for name in shared_inspection | shared_plotting:
        assert "pipls" not in name.lower()
        assert not name.lower().startswith("pls")
        assert not name.lower().startswith("plot_pls")


def test_shared_inspection_uses_no_concrete_plsregression_restriction() -> None:
    source = (_repository_root() / "src" / "pipls" / "inspection.py").read_text(
        encoding="utf-8"
    )
    assert "from sklearn.cross_decomposition import PLSRegression" not in source
    assert "isinstance(model, PLSRegression)" not in source
