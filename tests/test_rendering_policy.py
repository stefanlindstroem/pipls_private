from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

import pipls
import pipls.inspection as inspection

_RENDERING_PACKAGES = {"matplotlib", "adjustText"}
_RENDERING_METHODS = {
    "add_patch",
    "bar",
    "errorbar",
    "figure",
    "plot",
    "quiver",
    "savefig",
    "scatter",
    "subplots",
    "text",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _numbered_examples() -> list[Path]:
    return sorted((_repository_root() / "examples").glob("[0-9][0-9]_*.py"))


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _attribute_path(node: ast.expr) -> str | None:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _pyproject() -> dict[str, Any]:
    with (_repository_root() / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_runtime_package_exposes_no_plotting_api() -> None:
    root = _repository_root()
    assert not (root / "src" / "pipls" / "plotting.py").exists()
    assert importlib.util.find_spec("pipls.plotting") is None

    for module in (pipls, inspection):
        assert all(not name.startswith("plot_") for name in module.__all__)

    for path in (root / "src" / "pipls").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert _import_roots(tree).isdisjoint(_RENDERING_PACKAGES), path


def test_rendering_dependencies_are_optional_and_example_owned() -> None:
    project = _pyproject()["project"]
    runtime = project["dependencies"]
    extras = project["optional-dependencies"]

    assert not any(
        requirement.startswith(("matplotlib", "adjustText"))
        for requirement in runtime
    )
    assert "plot" not in extras
    for extra in ("dev", "docs", "examples"):
        requirements = extras[extra]
        assert any(value.startswith("matplotlib") for value in requirements)
        assert any(value.startswith("adjustText") for value in requirements)


def test_runtime_imports_without_rendering_dependencies() -> None:
    root = _repository_root()
    script = r'''
import builtins

blocked = {"matplotlib", "adjustText"}
original_import = builtins.__import__


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in blocked:
        raise ModuleNotFoundError(f"blocked optional dependency: {name}")
    return original_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded_import
import pipls
import pipls.inspection
assert pipls.PiPLSRegression is not None
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_numbered_examples_keep_rendering_caller_owned() -> None:
    examples = _numbered_examples()
    assert examples

    for path in examples:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = _import_roots(tree)
        calls = {
            name
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and (name := _call_name(node)) is not None
        }

        if path.name == "03_leave_one_out_validation.py":
            assert imports.isdisjoint(_RENDERING_PACKAGES)
            assert calls.isdisjoint(_RENDERING_METHODS)
            continue

        assert "matplotlib" in imports, path
        assert "subplots" in calls, path
        assert "savefig" in calls, path
        assert all(not name.startswith("plot_") for name in calls), path


def test_example_support_contains_no_rendering_layer() -> None:
    support = _repository_root() / "examples" / "_support"
    for path in support.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls = {
            name
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and (name := _call_name(node)) is not None
        }
        assert _import_roots(tree).isdisjoint(_RENDERING_PACKAGES), path
        assert calls.isdisjoint(_RENDERING_METHODS), path


def test_tutorial_examples_render_named_result_arrays_directly() -> None:
    root = _repository_root()
    synthetic = (root / "examples" / "02_synthetic_path_selection.py").read_text(
        encoding="utf-8"
    )
    pulp = (root / "examples" / "05_pulp_real_data.py").read_text(encoding="utf-8")

    for field in (
        "path.n_components",
        "path.cv_mse_mean",
        "rank_profile.predictor_rank",
        "rank_profile.cv_mse_mean",
        "diagnostics.observed_standardized",
        "diagnostics.predicted_standardized",
    ):
        assert field in synthetic

    for field in (
        "biplot.sample_coordinates",
        "biplot.predictor_coordinates",
        "factors.predictor_directions",
        "structure.x_scores",
        "diagnostics.standardized_rmse",
    ):
        assert field in pulp

    assert "biplot_coordinates(" in pulp
    assert "adjust_text(" in pulp
    assert pulp.index("biplot_axis.legend()") < pulp.index("adjust_text(")


def test_real_data_prediction_plots_use_response_neutral_residual_labels() -> None:
    root = _repository_root()
    sources = [
        root / "examples" / "05_pulp_real_data.py",
        root / "examples" / "06_sugarcane_real_data.py",
        root / "examples" / "07_tobacco_real_data.py",
        root / "tools" / "render_pulp_tutorial.py",
    ]

    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert 'set_ylabel("Standardized residual")' in text
        assert r"Residual $y-\hat y$ (standardized)" not in text


def test_maintained_cv_mse_error_bars_use_fold_based_standard_error() -> None:
    root = _repository_root()
    relative_paths = (
        "examples/02_synthetic_path_selection.py",
        "examples/04_pls_path_comparison.py",
        "examples/05_pulp_real_data.py",
        "examples/06_sugarcane_real_data.py",
        "examples/07_tobacco_real_data.py",
        "tools/render_synthetic_tutorial.py",
        "tools/render_pulp_tutorial.py",
    )

    for relative_path in relative_paths:
        path = root / relative_path
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        errorbar_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _call_name(node) == "errorbar"
        ]

        assert errorbar_calls, path
        assert "cv_mse_fold_sd" not in text, path
        assert 'set_ylabel("Mean response-standardized CV-MSE (±1 SE)")' in text
        for call in errorbar_calls:
            yerr = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "yerr"),
                None,
            )
            assert yerr is not None, path
            attribute = _attribute_path(yerr)
            assert attribute is not None, path
            assert attribute.endswith(".cv_mse_standard_error"), (path, attribute)

    comparison = (root / "examples" / "04_pls_path_comparison.py").read_text(
        encoding="utf-8"
    )
    assert "pipls_path.cv_mse_mean + pipls_path.cv_mse_standard_error" in comparison
    assert "pls_path.cv_mse_mean + pls_path.cv_mse_standard_error" in comparison

    pulp_renderer = (root / "tools" / "render_pulp_tutorial.py").read_text(
        encoding="utf-8"
    )
    assert (
        "component_path.cv_mse_mean + component_path.cv_mse_standard_error"
        in pulp_renderer
    )


def _call_source_segments(path: Path, names: set[str]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    return [
        ast.get_source_segment(text, node) or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) in names
    ]


def _source_between(path: Path, start: str, end: str) -> str:
    text = path.read_text(encoding="utf-8")
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index]


def test_maintained_rendered_method_names_use_pi_symbol() -> None:
    root = _repository_root()
    relative_paths = (
        "examples/01_minimal_fit_and_plot.py",
        "examples/02_synthetic_path_selection.py",
        "examples/04_pls_path_comparison.py",
        "examples/05_pulp_real_data.py",
        "examples/06_sugarcane_real_data.py",
        "examples/07_tobacco_real_data.py",
        "tools/render_synthetic_tutorial.py",
        "tools/render_pulp_tutorial.py",
    )

    for relative_path in relative_paths:
        path = root / relative_path
        text = path.read_text(encoding="utf-8")
        title_calls = _call_source_segments(path, {"set_title", "suptitle"})

        assert title_calls, path
        assert r"$\Pi$-PLS" in text, path
        assert all("Pi-PLS" not in call for call in title_calls), path


def test_maintained_paths_and_rank_profiles_share_zero_based_y_limits() -> None:
    root = _repository_root()
    expected_counts = {
        "examples/02_synthetic_path_selection.py": 2,
        "examples/04_pls_path_comparison.py": 1,
        "examples/05_pulp_real_data.py": 2,
        "examples/06_sugarcane_real_data.py": 1,
        "examples/07_tobacco_real_data.py": 1,
        "tools/render_synthetic_tutorial.py": 2,
        "tools/render_pulp_tutorial.py": 2,
    }
    limit_call = "set_ylim(0.0, max(1.0, 1.05 * upper))"

    for relative_path, expected_count in expected_counts.items():
        text = (root / relative_path).read_text(encoding="utf-8")
        assert text.count(limit_call) == expected_count, relative_path


def test_pipls_factor_figures_use_matrix_element_notation_without_tile_titles() -> None:
    root = _repository_root()
    factor_sources = (
        root / "examples" / "01_minimal_fit_and_plot.py",
        root / "examples" / "05_pulp_real_data.py",
        root / "examples" / "06_sugarcane_real_data.py",
        root / "examples" / "07_tobacco_real_data.py",
        root / "tools" / "render_pulp_tutorial.py",
    )

    for path in factor_sources:
        text = path.read_text(encoding="utf-8")
        assert "$q_{:" not in text, path
        assert "$d_kq_{:" not in text, path
        assert "$d_1q_{:" not in text, path

    tile_ranges = {
        "examples/01_minimal_fit_and_plot.py": (
            "figure, axes = plt.subplots(2, 2",
            "figure.suptitle(",
        ),
        "examples/05_pulp_real_data.py": (
            "# Plot the Pi-PLS factors directly",
            "figure.suptitle(",
        ),
        "examples/06_sugarcane_real_data.py": (
            "# Plot the Pi-PLS factors directly",
            "figure.suptitle(",
        ),
        "examples/07_tobacco_real_data.py": (
            "# Plot the Pi-PLS factors directly",
            "figure.suptitle(",
        ),
    }
    for relative_path, (start, end) in tile_ranges.items():
        block = _source_between(root / relative_path, start, end)
        assert ".set_title(" not in block, relative_path


def test_latent_and_prediction_tiles_have_no_subplot_titles() -> None:
    root = _repository_root()
    ranges = {
        "examples/05_pulp_real_data.py": (
            ("# Plot scores, a score-loading biplot, and loadings.", "figure.suptitle("),
            ("# Plot selection-conditioned prediction diagnostics.", "figure.suptitle("),
        ),
        "examples/06_sugarcane_real_data.py": (
            ("# Plot selection-conditioned prediction diagnostics.", "figure.suptitle("),
            ("# Plot scores and loadings from the selected full-data model.", "figure.suptitle("),
        ),
        "examples/07_tobacco_real_data.py": (
            (
                "# Plot selection-conditioned diagnostics in deterministic source-order response pages.",
                "figure.suptitle(",
            ),
            (
                "# Plot scores, loadings, and raw observation diagnostics.",
                "figure.suptitle(",
            ),
        ),
    }

    for relative_path, blocks in ranges.items():
        path = root / relative_path
        for start, end in blocks:
            assert ".set_title(" not in _source_between(path, start, end), (
                relative_path,
                start,
            )

    for relative_path in ranges:
        path = root / relative_path
        for call in _call_source_segments(path, {"suptitle"}):
            if "prediction diagnostics" in call:
                assert r"\n" not in call, relative_path


def test_dilation_plots_use_numeric_component_tick_labels() -> None:
    root = _repository_root()
    expected = {
        "examples/01_minimal_fit_and_plot.py": 'set_xticklabels(["1"])',
        "examples/05_pulp_real_data.py": (
            "set_xticklabels(np.arange(1, CHOSEN_N_COMPONENTS + 1))"
        ),
        "examples/06_sugarcane_real_data.py": (
            "set_xticklabels(np.arange(1, factors.n_components + 1))"
        ),
        "examples/07_tobacco_real_data.py": (
            "set_xticklabels(np.arange(1, factors.n_components + 1))"
        ),
    }

    for relative_path, numeric_ticks in expected.items():
        text = (root / relative_path).read_text(encoding="utf-8")
        assert numeric_ticks in text, relative_path
        assert 'set_xlabel("Component")' in text, relative_path

    combined = "\n".join(
        (root / relative_path).read_text(encoding="utf-8")
        for relative_path in expected
    )
    assert 'set_xticklabels(["Component 1"])' not in combined
    assert "set_xticklabels(component_labels" not in combined


def test_tobacco_dense_response_axes_use_angled_labels() -> None:
    tobacco = (
        _repository_root() / "examples" / "07_tobacco_real_data.py"
    ).read_text(encoding="utf-8")

    assert 'set_xticklabels(response_names, rotation=45, ha="right")' in tobacco
    assert 'rotation=45,\n            ha="right",\n        )' in tobacco


def test_standalone_pulp_tutorial_titles_do_not_repeat_axis_quantities() -> None:
    renderer = (
        _repository_root() / "tools" / "render_pulp_tutorial.py"
    ).read_text(encoding="utf-8")

    for removed_title in (
        "Pulp predictor directions",
        "Pulp weighted response directions",
        "Pulp observed versus predicted",
        "Pulp residual versus predicted",
        "Pulp standardized RMSE",
    ):
        assert removed_title not in renderer

    assert renderer.count('axis.set_title(\n        rf"Pulp $\\Pi$-PLS —') == 3
