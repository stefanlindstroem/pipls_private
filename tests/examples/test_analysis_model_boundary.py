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
    assert "axis.errorbar(" in comparison
    assert "plot_component_path" not in comparison
    assert ".to_csv(" not in comparison

    pulp_text = (examples_dir / "10_pulp_real_data.py").read_text(encoding="utf-8")
    assert "evaluate_pls_component_path(" not in pulp_text
    assert "plot_pipls_component_path(" not in pulp_text
    assert "axis.errorbar(" in pulp_text
    assert "latent_structure(model)" in pulp_text
    assert "PiPLSRegression(" in pulp_text
    assert "run_pulp_workflow(" not in pulp_text

    sugarcane = (examples_dir / "11_sugarcane_real_data.py").read_text(encoding="utf-8")
    assert "evaluate_pls_component_path(" not in sugarcane
    assert "plot_pipls_component_path(" not in sugarcane
    assert "axis.errorbar(" in sugarcane
    assert "latent_structure(model)" in sugarcane
    assert "PiPLSRegression(" in sugarcane

    tobacco = (examples_dir / "12_tobacco_real_data.py").read_text(encoding="utf-8")
    assert "evaluate_pls_component_path(" not in tobacco
    assert "plot_pipls_component_path(" not in tobacco
    assert "axis.errorbar(" in tobacco
    assert "latent_structure(model)" in tobacco
    assert "PiPLSRegression(" in tobacco


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


def test_sugarcane_example_owns_direct_figure_composition() -> None:
    path = _repository_root() / "examples" / "11_sugarcane_real_data.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    expected_plotters = {
        "plot_coefficients",
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
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_plotters
    ]

    assert {call.func.id for call in calls if isinstance(call.func, ast.Name)} == expected_plotters
    assert all(any(keyword.arg == "ax" for keyword in call.keywords) for call in calls)
    assert source.count("plt.subplots(") == 5
    assert source.count("figure.savefig(") == 5
    assert source.count("plt.close(figure)") == 5
    assert "post_analysis_artifacts" not in source
    assert "fixed_model_oof" not in source
    assert "plot_component_path" not in source


def test_tobacco_example_owns_direct_figure_composition() -> None:
    path = _repository_root() / "examples" / "12_tobacco_real_data.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    expected_plotters = {
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
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected_plotters
    ]

    assert {call.func.id for call in calls if isinstance(call.func, ast.Name)} == expected_plotters
    assert all(any(keyword.arg == "ax" for keyword in call.keywords) for call in calls)
    assert source.count("plt.subplots(") == 5
    assert source.count("PdfPages(") == 2
    assert source.count("figure.savefig(") == 3
    assert source.count("report.savefig(figure)") == 2
    assert source.count("plt.close(figure)") == 5
    assert "post_analysis_artifacts" not in source
    assert "fixed_model_oof" not in source
    assert "plot_component_path" not in source
