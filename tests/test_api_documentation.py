from __future__ import annotations

import importlib
import re
import subprocess
import sys
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
            "pipls.plotting",
        )
    }

    assert documented == expected


def test_api_pages_reference_only_public_import_paths() -> None:
    api_text = _api_text()

    assert "::: pipls._" not in api_text
    assert "show_source: true" not in api_text
    assert "pipls.model_selection" not in api_text


def test_plotting_module_import_does_not_require_matplotlib() -> None:
    script = r"""
import builtins

original_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "matplotlib" or name.startswith("matplotlib."):
        raise ModuleNotFoundError("Matplotlib intentionally blocked")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
import pipls.plotting
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=_repository_root(),
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(_repository_root() / "src")},
    )
