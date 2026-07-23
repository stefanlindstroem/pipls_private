from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

FIGURE_FILENAMES = (
    "component_path.svg",
    "predictor_rank_profile.svg",
    "observed_vs_predicted.svg",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def generated_synthetic_assets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repository = _repository_root()
    output_dir = tmp_path_factory.mktemp("synthetic-tutorial-assets") / "synthetic"
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
            str(repository / "tools" / "render_synthetic_tutorial.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repository,
        env=environment,
        check=True,
    )
    return output_dir


def test_synthetic_tutorial_renderer_writes_declared_parseable_svgs(
    generated_synthetic_assets: Path,
) -> None:
    manifest = json.loads(
        (generated_synthetic_assets / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == 1
    assert manifest["generator"] == {
        "name": "make_pipls_train_test",
        "random_state": 0,
        "n_train": 120,
        "n_test": 60,
        "n_features": 8,
        "n_targets": 3,
        "n_shared": 2,
        "n_predictor_specific": 2,
        "n_response_specific": 1,
    }
    assert manifest["analysis"]["chosen_n_components"] == 2
    assert manifest["analysis"]["chosen_predictor_rank"] == 4
    assert manifest["analysis"]["evaluated_component_counts"] == [1, 2, 3]
    assert manifest["analysis"]["evaluated_predictor_ranks"] == list(range(2, 9))
    assert manifest["analysis"]["prediction_kind"] == "external test predictions"
    assert math.isfinite(manifest["analysis"]["external_test_r2"])

    figures = manifest["figures"]
    assert tuple(item["filename"] for item in figures) == FIGURE_FILENAMES
    assert set(path.name for path in generated_synthetic_assets.iterdir()) == {
        *FIGURE_FILENAMES,
        "manifest.json",
    }
    for item in figures:
        figure_path = generated_synthetic_assets / item["filename"]
        ET.parse(figure_path)
        assert item["sha256"] == _sha256(figure_path)


def test_documentation_targets_own_generated_synthetic_assets() -> None:
    repository = _repository_root()
    makefile = (repository / "Makefile").read_text(encoding="utf-8")
    manifest = (repository / "MANIFEST.in").read_text(encoding="utf-8")
    sdist_checker = (repository / "tools" / "check_sdist_docs.py").read_text(
        encoding="utf-8"
    )
    with (repository / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)

    assert "tools/render_synthetic_tutorial.py" in makefile
    assert makefile.index("tools/render_synthetic_tutorial.py") < makefile.index(
        "tools/render_pulp_tutorial.py"
    )
    assert "include tools/render_synthetic_tutorial.py" in manifest
    assert "render_synthetic_tutorial.py" in sdist_checker
    assert 'source / "docs" / "tutorials" / "synthetic.md"' in sdist_checker
    assert 'source / "examples" / "02_synthetic_path_selection.py"' in sdist_checker
    assert 'source / "site" / "tutorials" / "synthetic" / "index.html"' in sdist_checker
    assert "SYNTHETIC_TUTORIAL_FIGURES" in sdist_checker
    assert "matplotlib>=3.8" in pyproject["project"]["optional-dependencies"]["docs"]


def test_synthetic_tutorial_is_the_first_learning_route() -> None:
    repository = _repository_root()
    tutorial = (repository / "docs" / "tutorials" / "synthetic.md").read_text(
        encoding="utf-8"
    )
    example = (
        repository / "examples" / "02_synthetic_path_selection.py"
    ).read_text(encoding="utf-8")
    renderer = (
        repository / "tools" / "render_synthetic_tutorial.py"
    ).read_text(encoding="utf-8")
    with (repository / "mkdocs.yml").open(encoding="utf-8") as stream:
        mkdocs = yaml.safe_load(stream)

    tutorials = next(item["Tutorials"] for item in mkdocs["nav"] if "Tutorials" in item)
    assert [next(iter(item.values())) for item in tutorials] == [
        "tutorials/synthetic.md",
        "tutorials/pulp.md",
    ]

    for source_path in (repository / "README.md", repository / "docs" / "index.md"):
        source = source_path.read_text(encoding="utf-8")
        assert "tutorials/synthetic.md" in source
        assert "tutorials/pulp.md" in source
        assert source.index("tutorials/synthetic.md") < source.index("tutorials/pulp.md")

    pulp = (repository / "docs" / "tutorials" / "pulp.md").read_text(encoding="utf-8")
    assert "synthetic.md" in pulp

    snippet_sections = {
        "generate-synthetic-data",
        "evaluate-synthetic-path",
        "plot-synthetic-component-path",
        "plot-synthetic-rank-profile",
        "fit-predict-synthetic-model",
    }
    for section in snippet_sections:
        assert f"examples/02_synthetic_path_selection.py:{section}" in tutorial
        assert f"# --8<-- [start:{section}]" in example
        assert f"# --8<-- [end:{section}]" in example

    for filename in FIGURE_FILENAMES:
        assert f"../assets/generated/synthetic/{filename}" in tutorial

    linked_targets = re.findall(r"\]\(([^)#]+)(?:#[^)]+)?\)", tutorial)
    assert "pulp.md" in linked_targets
    assert "../api/path.md" in linked_targets
    assert "../api/regression.md" in linked_targets
    assert "../path_analysis.md" in linked_targets

    selected_line = "selected = path.for_n_components(CHOSEN_N_COMPONENTS)"
    component_plot = '# --8<-- [start:plot-synthetic-component-path]'
    rank_profile = "rank_profile = search.predictor_rank_profile("
    fixed_fit = "model = PiPLSRegression("
    assert (
        example.index(selected_line)
        < example.index(component_plot)
        < example.index(rank_profile)
        < example.index(fixed_fit)
    )
    assert "PiPLSPathCV(refit=False).fit(train.X, train.Y)" in renderer
    assert "predictor_rank_profile(selected.n_components)" in renderer
    assert 'prediction_kind="external test predictions"' in renderer
    assert "cross_val_predict" not in renderer
