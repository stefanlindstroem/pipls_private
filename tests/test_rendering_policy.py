from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pipls
import pipls.inspection as inspection


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_runtime_package_exposes_no_plotting_api() -> None:
    assert importlib.util.find_spec("pipls.plotting") is None

    for module in (pipls, inspection):
        assert all(not name.startswith("plot_") for name in module.__all__)


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



def test_pulp_example_runs_without_adjusttext(tmp_path: Path) -> None:
    root = _repository_root()
    example_path = tmp_path / "04_pulp_real_data.py"
    example_path.write_text(
        (root / "examples" / "04_pulp_real_data.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "results" / "pulp_post_analysis").mkdir(parents=True)
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        """import builtins

_original_import = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(\".\", 1)[0] == \"adjustText\":
        raise ModuleNotFoundError(
            \"blocked optional dependency: adjustText\",
            name=\"adjustText\",
        )
    return _original_import(name, globals, locals, fromlist, level)


builtins.__import__ = _guarded_import
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": f"{tmp_path}{os.pathsep}{root / 'src'}",
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, str(example_path)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Selected Pi-PLS: n_components=3, predictor_rank=9" in completed.stdout
    assert (tmp_path / "results" / "pulp_post_analysis" / "latent_structure.pdf").is_file()
