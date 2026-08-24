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

blocked = {"matplotlib", "textalloc"}
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



def test_pulp_example_runs_without_textalloc(tmp_path: Path) -> None:
    root = _repository_root()
    example_path = tmp_path / "04_pulp_real_data.py"
    example_path.write_text(
        (root / "examples" / "04_pulp_real_data.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "results" / "pulp_post_analysis").mkdir(parents=True)
    support_dir = tmp_path / "_support"
    support_dir.mkdir()
    (support_dir / "pulp_biplot.py").write_text(
        (root / "examples" / "_support" / "pulp_biplot.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        """import builtins

_original_import = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(\".\", 1)[0] == \"textalloc\":
        raise ModuleNotFoundError(
            \"blocked optional dependency: textalloc\",
            name=\"textalloc\",
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


def test_pulp_renderers_use_shared_biplot_helper() -> None:
    root = _repository_root()
    for relative_path in (
        "examples/04_pulp_real_data.py",
        "tools/render_pulp_tutorial.py",
    ):
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "plot_pulp_biplot" in source
        assert "adjust_text" not in source
        assert "x_scatter" not in source
        assert "y_scatter" not in source
