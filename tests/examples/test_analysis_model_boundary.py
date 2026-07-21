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
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
        for node in ast.walk(tree)
    )


def test_numbered_examples_do_not_import_or_fit_ordinary_pls() -> None:
    examples_dir = _repository_root() / "examples"
    numbered = sorted(examples_dir.glob("[0-9][0-9]_*.py"))
    assert numbered

    for path in numbered:
        assert not _imports_name(path, "PLSRegression"), path
        assert not _calls_name(path, "PLSRegression"), path


def test_dedicated_example_owns_the_pls_comparison_path() -> None:
    examples_dir = _repository_root() / "examples"
    comparison = (examples_dir / "09_pls_path_comparison.py").read_text(encoding="utf-8")
    assert "evaluate_pls_component_path(" in comparison
    assert "plot_component_path_comparison(" in comparison

    for filename in (
        "10_pulp_real_data.py",
        "11_sugarcane_real_data.py",
        "12_tobacco_real_data.py",
    ):
        text = (examples_dir / filename).read_text(encoding="utf-8")
        assert "evaluate_pls_component_path(" not in text
        assert "plot_pipls_component_path(" in text
        assert "latent_structure(model)" in text
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
        "plot_observed_vs_predicted",
        "plot_residuals_vs_predicted",
        "plot_standardized_rmse",
        "plot_scores",
        "plot_x_loadings",
        "plot_y_loadings",
    }
    pipls_specific_plotting = {
        "plot_pipls_dilation",
        "plot_pipls_predictor_directions",
        "plot_pipls_response_directions",
        "plot_pipls_weighted_response_directions",
    }
    assert set(plotting.__all__) == shared_plotting | pipls_specific_plotting

    for name in shared_inspection | shared_plotting:
        assert "pipls" not in name.lower()
        assert not name.lower().startswith("pls")
        assert not name.lower().startswith("plot_pls")


def test_shared_inspection_uses_no_concrete_plsregression_restriction() -> None:
    source = (_repository_root() / "src" / "pipls" / "inspection.py").read_text(encoding="utf-8")
    assert "from sklearn.cross_decomposition import PLSRegression" not in source
    assert "isinstance(model, PLSRegression)" not in source


def test_post_analysis_helper_owns_all_report_composition() -> None:
    path = _repository_root() / "examples" / "_support" / "post_analysis_artifacts.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    render = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "render_post_analysis_report"
    )
    report_plotters = {
        "plot_biplot",
        "plot_coefficients",
        "plot_observation_diagnostics",
        "plot_observed_vs_predicted",
        "plot_pipls_dilation",
        "plot_pipls_predictor_directions",
        "plot_pipls_response_directions",
        "plot_pipls_weighted_response_directions",
        "plot_residuals_vs_predicted",
        "plot_scores",
        "plot_standardized_rmse",
        "plot_x_loadings",
        "plot_y_loadings",
    }
    calls = [
        node
        for node in ast.walk(render)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in report_plotters
    ]

    assert {call.func.id for call in calls if isinstance(call.func, ast.Name)} == report_plotters
    assert all(any(keyword.arg == "ax" for keyword in call.keywords) for call in calls)
    assert "plot_prediction_diagnostics" not in source
    assert "plot_pipls_decomposition" not in source
    assert "plt.subplots(\n            2,\n            2," in source
    assert "plt.subplots(\n                1,\n                3," in source
    assert source.count("plt.subplot_mosaic(") == 4
    assert '["scores", "biplot", "x_loadings"]' in source
    assert '["y_loadings", "observations", "observations"]' in source
    assert '[["scores", "biplot"], ["x_loadings", "y_loadings"]]' in source
    assert '[["scores", "x_loadings"], ["y_loadings", "observations"]]' in source
    assert '[["scores", "x_loadings", "y_loadings"]]' in source
    assert source.count("include_prediction_kind=False") == 3
    assert 'figure.suptitle(f"{dataset_name} Pi-PLS latent-model views")' in source
