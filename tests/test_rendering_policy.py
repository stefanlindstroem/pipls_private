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
