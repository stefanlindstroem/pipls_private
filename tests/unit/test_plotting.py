from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
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
        "plot_pipls_dilation",
        "plot_pipls_predictor_directions",
        "plot_pipls_response_directions",
        "plot_pipls_weighted_response_directions",
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


def test_atomic_pipls_factor_plotters_render_the_four_factor_quantities() -> None:
    factors = _factors()
    predictor_before = factors.predictor_directions.copy()  # type: ignore[union-attr]

    predictor_figure, predictor_axis = plotting.plot_pipls_predictor_directions(
        factors,  # type: ignore[arg-type]
        predictor_style="bar",
        predictor_names=["Temperature", "Pressure", "Flow", "Residence time"],
        components=[0, 1],
    )
    dilation_figure, dilation_axis = plotting.plot_pipls_dilation(
        factors,  # type: ignore[arg-type]
        components=[0, 1],
    )
    response_figure, response_axis = plotting.plot_pipls_response_directions(
        factors,  # type: ignore[arg-type]
        response_names=["Yield", "Purity", "Energy demand"],
        components=[0, 1],
    )
    weighted_figure, weighted_axis = plotting.plot_pipls_weighted_response_directions(
        factors,  # type: ignore[arg-type]
        response_names=["Yield", "Purity", "Energy demand"],
        components=[0, 1],
    )

    assert predictor_figure.axes == [predictor_axis]
    assert dilation_figure.axes == [dilation_axis]
    assert response_figure.axes == [response_axis]
    assert weighted_figure.axes == [weighted_axis]
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
    assert len(response_axis.containers) == 2
    assert len(weighted_axis.containers) == 2
    np.testing.assert_allclose(
        [patch.get_height() for patch in dilation_axis.patches],
        factors.dilation,  # type: ignore[union-attr]
    )
    for component, container in enumerate(response_axis.containers):
        np.testing.assert_allclose(
            [patch.get_height() for patch in container],
            factors.response_directions[:, component],  # type: ignore[union-attr]
        )
    for component, container in enumerate(weighted_axis.containers):
        np.testing.assert_allclose(
            [patch.get_height() for patch in container],
            factors.weighted_response_directions[:, component],  # type: ignore[union-attr]
        )
    assert predictor_axis.get_legend() is None
    assert response_axis.get_legend() is None
    assert weighted_axis.get_legend() is None
    assert predictor_axis.get_legend_handles_labels()[1] == ["Component 1", "Component 2"]
    np.testing.assert_array_equal(
        factors.predictor_directions,  # type: ignore[union-attr]
        predictor_before,
    )


def test_pipls_factor_plotters_compose_in_a_caller_owned_panel() -> None:
    import matplotlib.pyplot as plt

    factors = _factors()
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    figure_numbers = plt.get_fignums()

    calls = (
        plotting.plot_pipls_predictor_directions(
            factors,  # type: ignore[arg-type]
            predictor_style="bar",
            predictor_names=["A", "B", "C", "D"],
            ax=axes[0, 0],
        ),
        plotting.plot_pipls_dilation(factors, ax=axes[0, 1]),  # type: ignore[arg-type]
        plotting.plot_pipls_response_directions(
            factors,  # type: ignore[arg-type]
            response_names=["U", "V", "W"],
            ax=axes[1, 0],
        ),
        plotting.plot_pipls_weighted_response_directions(
            factors,  # type: ignore[arg-type]
            response_names=["U", "V", "W"],
            ax=axes[1, 1],
        ),
    )

    assert all(returned_figure is figure for returned_figure, _ in calls)
    assert [returned_axis for _, returned_axis in calls] == list(axes.flat)
    assert figure.axes == list(axes.flat)
    assert plt.get_fignums() == figure_numbers


def test_plot_pipls_predictor_directions_preserves_physical_axis_order() -> None:
    factors = _factors()
    coordinate = np.array([1600.0, 1500.0, 1400.0, 1300.0])

    _, axis = plotting.plot_pipls_predictor_directions(
        factors,  # type: ignore[arg-type]
        predictor_style="line",
        predictor_axis=coordinate,
        predictor_axis_label="Wavenumber (1/cm)",
        components=[0, 1],
    )

    np.testing.assert_array_equal(axis.lines[0].get_xdata(), coordinate)
    np.testing.assert_array_equal(axis.lines[1].get_xdata(), coordinate)
    assert axis.get_xlim() == (coordinate[0], coordinate[-1])
    assert axis.get_xlabel() == "Wavenumber (1/cm)"
    assert axis.get_legend() is None
    assert axis.get_legend_handles_labels()[1] == ["Component 1", "Component 2"]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: plotting.plot_pipls_predictor_directions(
                _factors(),  # type: ignore[arg-type]
                predictor_style="points",  # type: ignore[arg-type]
            ),
            "bar.*line",
        ),
        (
            lambda: plotting.plot_pipls_predictor_directions(
                _factors(),  # type: ignore[arg-type]
                predictor_style="bar",
            ),
            "predictor_names is required",
        ),
        (
            lambda: plotting.plot_pipls_predictor_directions(
                _factors(),  # type: ignore[arg-type]
                predictor_style="line",
            ),
            "predictor_axis is required",
        ),
        (
            lambda: plotting.plot_pipls_predictor_directions(
                _factors(),  # type: ignore[arg-type]
                predictor_style="line",
                predictor_axis=[1.0, 2.0, 3.0, 4.0],
            ),
            "predictor_axis_label is required",
        ),
        (
            lambda: plotting.plot_pipls_response_directions(
                _factors(),  # type: ignore[arg-type]
            ),
            "response_names is required",
        ),
        (
            lambda: plotting.plot_pipls_weighted_response_directions(
                _factors(),  # type: ignore[arg-type]
                response_names=["u"],
            ),
            "Expected 3",
        ),
        (
            lambda: plotting.plot_pipls_dilation(
                _factors(),  # type: ignore[arg-type]
                components=[2],
            ),
            "0 <= index < 2",
        ),
        (
            lambda: plotting.plot_pipls_dilation(
                _factors(),  # type: ignore[arg-type]
                components=[0, 0],
            ),
            "duplicate",
        ),
    ],
)
def test_atomic_pipls_factor_plotters_reject_invalid_arguments(
    call: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_atomic_pipls_factor_plotters_reject_figsize_with_supplied_axis() -> None:
    import matplotlib.pyplot as plt

    _, axis = plt.subplots()
    with pytest.raises(ValueError, match="figsize cannot be supplied"):
        plotting.plot_pipls_dilation(
            _factors(),  # type: ignore[arg-type]
            figsize=(5.0, 4.0),
            ax=axis,
        )


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
