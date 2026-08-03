from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

FIGURE_FILENAMES = ("observed_vs_fitted.svg",)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def generated_quick_start_assets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repository = _repository_root()
    output_dir = tmp_path_factory.mktemp("quick-start-tutorial-assets") / "quick_start"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(repository / "src"),
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    subprocess.run(
        [
            sys.executable,
            str(repository / "tools" / "render_quick_start_tutorial.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repository,
        env=environment,
        check=True,
    )
    return output_dir


def test_quick_start_renderer_writes_one_parseable_fitted_value_svg(
    generated_quick_start_assets: Path,
) -> None:
    manifest = json.loads(
        (generated_quick_start_assets / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == 1
    assert manifest["dataset"] == {"id": "pulp", "version": "1"}
    assert manifest["analysis"]["selected_n_components"] == 3
    assert manifest["analysis"]["selected_predictor_rank"] == 10
    assert manifest["analysis"]["prediction_kind"] == "fitted values"
    mean_standardized_rmse = manifest["analysis"]["mean_standardized_rmse"]
    assert math.isfinite(mean_standardized_rmse)
    assert 0.0 < mean_standardized_rmse < 1.0

    figures = manifest["figures"]
    assert tuple(item["filename"] for item in figures) == FIGURE_FILENAMES
    assert set(path.name for path in generated_quick_start_assets.iterdir()) == {
        *FIGURE_FILENAMES,
        "manifest.json",
    }
    figure_path = generated_quick_start_assets / FIGURE_FILENAMES[0]
    ET.parse(figure_path)
    assert figures[0]["sha256"] == _sha256(figure_path)


def test_quick_start_tutorial_owns_snippets_asset_and_navigation() -> None:
    repository = _repository_root()
    tutorial = (repository / "docs" / "tutorials" / "quick_start.md").read_text(
        encoding="utf-8"
    )
    example = (repository / "examples" / "01_pulp_quick_start.py").read_text(
        encoding="utf-8"
    )
    with (repository / "mkdocs.yml").open(encoding="utf-8") as stream:
        mkdocs = yaml.safe_load(stream)

    tutorials = next(item["Tutorials"] for item in mkdocs["nav"] if "Tutorials" in item)
    assert [next(iter(item.values())) for item in tutorials] == [
        "tutorials/quick_start.md",
        "tutorials/synthetic.md",
        "tutorials/pulp.md",
    ]

    for section in (
        "load-pulp-data",
        "fit-selected-pulp-model",
        "plot-standardized-fitted-values",
    ):
        assert f"examples/01_pulp_quick_start.py:{section}" in tutorial
        assert f"# --8<-- [start:{section}]" in example
        assert f"# --8<-- [end:{section}]" in example

    assert "../assets/generated/quick_start/observed_vs_fitted.svg" in tutorial
    assert "fitted values" in tutorial
    assert "calibration fit" in tutorial
    assert "not an out-of-fold" in tutorial
    assert "selection-conditioned OOF diagnostics" in tutorial
    assert "model.selection_" in tutorial
    assert "search.oof_report" in tutorial
    assert "validation_report" not in tutorial


def test_documentation_targets_own_generated_quick_start_assets() -> None:
    repository = _repository_root()
    makefile = (repository / "Makefile").read_text(encoding="utf-8")
    manifest = (repository / "MANIFEST.in").read_text(encoding="utf-8")
    sdist_checker = (repository / "tools" / "check_sdist_docs.py").read_text(
        encoding="utf-8"
    )

    assert "tools/render_quick_start_tutorial.py" in makefile
    assert makefile.index("tools/render_quick_start_tutorial.py") < makefile.index(
        "tools/render_synthetic_tutorial.py"
    )
    assert "include tools/render_quick_start_tutorial.py" in manifest
    assert "render_quick_start_tutorial.py" in sdist_checker
    assert 'source / "docs" / "tutorials" / "quick_start.md"' in sdist_checker
    assert 'source / "examples" / "01_pulp_quick_start.py"' in sdist_checker
    assert 'source / "site" / "tutorials" / "quick_start" / "index.html"' in sdist_checker
    assert "QUICK_START_FIGURES" in sdist_checker
