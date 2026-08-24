from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from pipls.datasets import load_pulp

FIGURE_FILENAMES = (
    "component_path.svg",
    "selected_component_path.svg",
    "predictor_rank_profile.svg",
    "biplot.svg",
    "predictor_directions.svg",
    "weighted_response_directions.svg",
    "observed_vs_predicted.svg",
    "residuals_vs_predicted.svg",
    "oof_response_r2.svg",
    "final_fit_observed_vs_predicted.svg",
    "final_fit_r2.svg",
    "final_fit_residual_distribution.svg",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def generated_pulp_assets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repository = _repository_root()
    output_dir = tmp_path_factory.mktemp("pulp-tutorial-assets") / "pulp"
    environment = os.environ.copy()
    blocked_dependency_dir = tmp_path_factory.mktemp("pulp-no-textalloc")
    (blocked_dependency_dir / "sitecustomize.py").write_text(
        """import builtins

_original_import = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] == "textalloc":
        raise ModuleNotFoundError(
            "blocked optional dependency: textalloc",
            name="textalloc",
        )
    return _original_import(name, globals, locals, fromlist, level)


builtins.__import__ = _guarded_import
""",
        encoding="utf-8",
    )
    python_path = os.pathsep.join(
        (
            str(blocked_dependency_dir),
            str(repository / "src"),
            str(repository / "examples"),
        )
    )
    environment.update(
        {
            "PYTHONPATH": python_path,
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
            str(repository / "tools" / "render_pulp_tutorial.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repository,
        env=environment,
        check=True,
    )
    return output_dir


def test_pulp_tutorial_renderer_records_repeated_cv_and_valid_figures(
    generated_pulp_assets: Path,
) -> None:
    manifest = json.loads(
        (generated_pulp_assets / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == 2
    assert manifest["dataset"]["id"] == "pulp"
    assert manifest["dataset"]["version"] == "1"
    analysis = manifest["analysis"]
    assert analysis["chosen_n_components"] == 3
    assert analysis["chosen_predictor_rank"] == 9
    assert analysis["search_method"] == "exhaustive"
    assert analysis["search_is_exhaustive"] is True
    assert analysis["max_predictor_rank"] == 14
    assert analysis["evaluated_predictor_ranks"] == list(range(3, 15))
    assert analysis["cross_validation"] == {
        "splitter": "RepeatedKFold",
        "n_splits": 5,
        "n_repeats": 10,
        "random_state": 0,
        "materialized_splits": 50,
    }
    assert analysis["oof_predictions_per_observation"] == 10
    assert analysis["prediction_kind"] == (
        "selection-conditioned OOF predictions"
    )
    assert analysis["detailed_responses"] == list(load_pulp().target_names)
    oof_response_r2 = analysis["response_r2"]
    assert [item["response"] for item in oof_response_r2] == list(
        load_pulp().target_names
    )
    oof_r2_values = [item["value"] for item in oof_response_r2]
    assert len(oof_r2_values) == 8
    assert all(np.isfinite(value) for value in oof_r2_values)
    assert all(value <= 1.0 for value in oof_r2_values)

    final_fit = manifest["final_fit"]
    assert final_fit["prediction_kind"] == "fitted values"
    response_r2 = final_fit["response_r2"]
    assert [item["response"] for item in response_r2] == list(
        load_pulp().target_names
    )
    r2_values = [item["value"] for item in response_r2]
    assert len(r2_values) == 8
    assert all(np.isfinite(value) for value in r2_values)
    assert all(value <= 1.0 for value in r2_values)

    pooled = final_fit["pooled_standardized_residuals"]
    assert pooled["count"] == 46 * 8
    assert np.isfinite(pooled["mean"])
    assert abs(pooled["mean"]) < 1e-12
    assert np.isfinite(pooled["sample_sd"])
    assert pooled["sample_sd"] > 0.0

    figures = manifest["figures"]
    assert tuple(item["filename"] for item in figures) == FIGURE_FILENAMES
    assert set(path.name for path in generated_pulp_assets.iterdir()) == {
        *FIGURE_FILENAMES,
        "manifest.json",
    }
    for item in figures:
        figure_path = generated_pulp_assets / item["filename"]
        ET.parse(figure_path)
        assert item["sha256"] == _sha256(figure_path)
