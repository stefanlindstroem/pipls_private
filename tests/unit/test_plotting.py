from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

import pipls
import pipls.plotting as plotting
from pipls import PiPLSDecomposition
from pipls.inspection import pipls_display_factors, prediction_diagnostics

matplotlib.use("Agg")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _factors() -> object:
    decomposition = PiPLSDecomposition(
        Pi=np.eye(4, 2),
        C=np.eye(3, 2),
        W=np.eye(2),
        P=np.array(
            [
                [0.5, -0.2],
                [0.25, 0.6],
                [-0.1, 0.3],
                [0.4, -0.1],
            ]
        ),
        D=np.diag([2.0, 0.75]),
        Q=np.array(
            [
                [0.8, -0.1],
                [-0.4, 0.5],
                [0.2, 0.7],
            ]
        ),
        dilation=np.array([2.0, 0.75]),
        x_rank=2,
        x_rank_is_exact=True,
        rank_tolerance=1e-12,
        predictor_svd_solver="full",
    )
    return pipls_display_factors(decomposition)


def _diagnostics() -> object:
    observed = np.array(
        [
            [1.0, 10.0, -2.0],
            [2.0, 12.0, -1.0],
            [4.0, 14.0, 1.0],
            [5.0, 18.0, 2.0],
        ]
    )
    predicted = observed + np.array(
        [
            [0.2, -1.0, 0.1],
            [-0.1, 0.5, -0.2],
            [0.3, -0.5, 0.2],
            [-0.2, 1.0, -0.1],
        ]
    )
    return prediction_diagnostics(
        observed,
        predicted,
        prediction_kind="external test predictions",
    )


def test_plotting_names_are_submodule_exports_only() -> None:
    expected = {
        "PredictorStyle",
        "plot_pipls_decomposition",
        "plot_biplot",
        "plot_coefficients",
        "plot_observation_diagnostics",
        "plot_scores",
        "plot_x_loadings",
        "plot_y_loadings",
        "plot_prediction_diagnostics",
    }

    assert set(plotting.__all__) == expected
    assert expected.isdisjoint(pipls.__all__)


def test_plotting_submodule_import_does_not_import_matplotlib() -> None:
    script = """
import sys
import pipls.plotting
assert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_repository_root(),
        env={**os.environ, "PYTHONPATH": str(_repository_root() / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""


def test_plot_pipls_decomposition_groups_components_on_shared_axes() -> None:
    factors = _factors()
    predictor_before = factors.predictor_directions.copy()  # type: ignore[union-attr]

    figure, axes = plotting.plot_pipls_decomposition(
        factors,  # type: ignore[arg-type]
        predictor_style="bar",
        predictor_names=["Temperature", "Pressure", "Flow", "Residence time"],
        response_names=["Yield", "Purity", "Energy demand"],
        components=[0, 1],
        title="Example decomposition",
    )

    assert isinstance(figure, Figure)
    assert set(axes) == {
        "predictor_directions",
        "weighted_response_directions",
        "dilation",
    }
    predictor_axis = axes["predictor_directions"]
    response_axis = axes["weighted_response_directions"]
    assert [tick.get_text() for tick in predictor_axis.get_xticklabels()] == [
        "Temperature",
        "Pressure",
        "Flow",
        "Residence time",
    ]
    assert [tick.get_text() for tick in response_axis.get_xticklabels()] == [
        "Yield",
        "Purity",
        "Energy demand",
    ]
    assert len(predictor_axis.containers) == 2
    assert all(len(container) == 4 for container in predictor_axis.containers)
    assert len(response_axis.containers) == 2
    assert all(len(container) == 3 for container in response_axis.containers)
    assert [text.get_text() for text in predictor_axis.get_legend().get_texts()] == [
        "Component 1",
        "Component 2",
    ]
    assert figure._suptitle is not None
    assert figure._suptitle.get_text() == "Example decomposition"
    np.testing.assert_array_equal(  # type: ignore[union-attr]
        factors.predictor_directions,
        predictor_before,
    )


def test_plot_pipls_decomposition_line_mode_overlays_selected_components() -> None:
    factors = _factors()
    coordinate = np.array([1600.0, 1500.0, 1400.0, 1300.0])

    _, axes = plotting.plot_pipls_decomposition(
        factors,  # type: ignore[arg-type]
        predictor_style="line",
        predictor_axis=coordinate,
        predictor_axis_label="Wavenumber (1/cm)",
        response_names=["Yield", "Purity", "Energy demand"],
        components=[0, 1],
    )

    axis = axes["predictor_directions"]
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), coordinate)
    np.testing.assert_array_equal(axis.lines[1].get_xdata(), coordinate)
    assert axis.get_xlim() == (coordinate[0], coordinate[-1])
    assert axis.get_xlabel() == "Wavenumber (1/cm)"
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Component 1",
        "Component 2",
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"predictor_style": "points"}, "bar.*line"),
        (
            {"predictor_style": "bar", "response_names": ["a", "b", "c"]},
            "predictor_names is required",
        ),
        (
            {
                "predictor_style": "bar",
                "predictor_names": ["a", "b", "c", "d"],
            },
            "response_names is required",
        ),
        (
            {
                "predictor_style": "line",
                "response_names": ["a", "b", "c"],
            },
            "predictor_axis is required",
        ),
        (
            {
                "predictor_style": "line",
                "predictor_axis": [1.0, 2.0, 3.0, 4.0],
                "response_names": ["a", "b", "c"],
            },
            "predictor_axis_label is required",
        ),
        (
            {
                "predictor_style": "bar",
                "predictor_names": ["a", "b", "c", "d"],
                "response_names": ["u", "v", "w"],
                "predictor_axis": [1.0, 2.0, 3.0, 4.0],
            },
            "only valid",
        ),
        (
            {
                "predictor_style": "bar",
                "predictor_names": ["a"],
                "response_names": ["u", "v", "w"],
            },
            "Expected 4",
        ),
        (
            {
                "predictor_style": "bar",
                "predictor_names": ["a", "b", "c", "d"],
                "response_names": ["u", "v", "w"],
                "components": [2],
            },
            "0 <= index < 2",
        ),
        (
            {
                "predictor_style": "bar",
                "predictor_names": ["a", "b", "c", "d"],
                "response_names": ["u", "v", "w"],
                "components": [0, 0],
            },
            "duplicate",
        ),
    ],
)
def test_plot_pipls_decomposition_rejects_invalid_display_arguments(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        plotting.plot_pipls_decomposition(_factors(), **kwargs)  # type: ignore[arg-type]


def test_plot_prediction_diagnostics_returns_named_axes_and_provenance() -> None:
    diagnostics = _diagnostics()
    residual_before = diagnostics.residual.copy()  # type: ignore[union-attr]

    figure, axes = plotting.plot_prediction_diagnostics(
        diagnostics,  # type: ignore[arg-type]
        response_names=["Yield", "Purity", "Energy demand"],
        responses=[0, 2],
        title="External prediction review",
    )

    assert isinstance(figure, Figure)
    assert set(axes) == {
        "observed_vs_predicted",
        "residual_vs_predicted",
        "standardized_rmse",
    }
    assert figure._suptitle is not None
    assert figure._suptitle.get_text() == ("External prediction review\nexternal test predictions")
    assert [tick.get_text() for tick in axes["standardized_rmse"].get_xticklabels()] == [
        "Yield",
        "Energy demand",
    ]
    np.testing.assert_array_equal(diagnostics.residual, residual_before)  # type: ignore[union-attr]


def test_plot_prediction_diagnostics_writes_pdf(tmp_path: Path) -> None:
    figure, _ = plotting.plot_prediction_diagnostics(
        _diagnostics(),  # type: ignore[arg-type]
        response_names=["Yield", "Purity", "Energy demand"],
    )
    output = tmp_path / "diagnostics.pdf"

    figure.savefig(output)

    assert output.is_file()
    assert output.stat().st_size > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "response_names is required"),
        ({"response_names": ["a"]}, "Expected 3"),
        (
            {"response_names": ["a", "b", "c"], "responses": []},
            "at least one",
        ),
        (
            {"response_names": ["a", "b", "c"], "responses": [3]},
            "0 <= index < 3",
        ),
        (
            {"response_names": ["a", "b", "c"], "responses": [True]},
            "integer indices",
        ),
    ],
)
def test_plot_prediction_diagnostics_rejects_invalid_arguments(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        plotting.plot_prediction_diagnostics(_diagnostics(), **kwargs)  # type: ignore[arg-type]
