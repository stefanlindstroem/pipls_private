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
from tests._source_contracts import call_name, dotted_name, import_roots, parse_module

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
        tree = parse_module(path)
        assert import_roots(tree).isdisjoint(_RENDERING_PACKAGES), path


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


def test_maintained_cv_mse_error_bars_use_split_standard_deviation() -> None:
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
        tree = parse_module(path)
        errorbar_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and call_name(node) == "errorbar"
        ]

        assert errorbar_calls, path
        assert "cv_mse_standard_error" not in text, path
        assert "(±1 SE)" not in text, path
        assert "(±1 SD)" in text, path
        for call in errorbar_calls:
            yerr = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "yerr"),
                None,
            )
            assert yerr is not None, path
            attribute = dotted_name(yerr)
            assert attribute is not None, path
            assert attribute.endswith(".cv_mse_std"), (path, attribute)
