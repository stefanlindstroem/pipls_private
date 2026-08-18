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

FIGURE_FILENAMES = (
    "component_path.svg",
    "selected_component_path.svg",
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
    assert manifest["analysis"]["search_method"] == "exhaustive"
    assert manifest["analysis"]["search_is_exhaustive"] is True
    assert manifest["analysis"]["max_predictor_rank"] == 8
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
