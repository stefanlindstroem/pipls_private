from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure
from sklearn.cross_decomposition import PLSRegression

import pipls.plotting as plotting
from pipls.inspection import PLSLatentStructure, pls_latent_structure

matplotlib.use("Agg")


def _structure() -> PLSLatentStructure:
    rng = np.random.default_rng(932)
    X = rng.normal(size=(28, 5))
    Y = X[:, :3] @ np.array(
        [
            [1.0, -0.2, 0.4],
            [0.5, 0.7, -0.1],
            [-0.4, 0.3, 0.8],
        ]
    ) + 0.05 * rng.normal(size=(28, 3))
    return pls_latent_structure(PLSRegression(n_components=3).fit(X, Y))


def test_plot_pls_scores_returns_named_axis_and_selected_components() -> None:
    figure, axes = plotting.plot_pls_scores(
        _structure(),
        components=(0, 2),
        sample_names=[f"Sample {index + 1}" for index in range(28)],
        title="PLS score review",
    )

    assert isinstance(figure, Figure)
    assert set(axes) == {"scores"}
    assert axes["scores"].get_xlabel() == "X score component 1"
    assert axes["scores"].get_ylabel() == "X score component 3"
    assert axes["scores"].get_title() == "PLS score review"
    assert len(axes["scores"].texts) == 28


def test_plot_pls_x_loadings_line_mode_overlays_components_and_preserves_axis() -> None:
    coordinate = np.array([1600.0, 1500.0, 1400.0, 1300.0, 1200.0])

    _, axes = plotting.plot_pls_x_loadings(
        _structure(),
        predictor_style="line",
        predictor_axis=coordinate,
        predictor_axis_label="Wavenumber (1/cm)",
        components=[0, 2],
    )

    assert set(axes) == {"x_loadings"}
    axis = axes["x_loadings"]
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), coordinate)
    np.testing.assert_array_equal(axis.lines[1].get_xdata(), coordinate)
    assert axis.get_xlabel() == "Wavenumber (1/cm)"
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Component 1",
        "Component 3",
    ]


def test_plot_pls_x_loadings_groups_component_bars_by_predictor() -> None:
    _, axes = plotting.plot_pls_x_loadings(
        _structure(),
        predictor_style="bar",
        predictor_names=["Temperature", "Pressure", "Flow", "Density", "Viscosity"],
        components=[0, 1, 2],
    )

    axis = axes["x_loadings"]
    assert len(axis.containers) == 3
    assert all(len(container) == 5 for container in axis.containers)
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "Temperature",
        "Pressure",
        "Flow",
        "Density",
        "Viscosity",
    ]


def test_plot_pls_coefficients_line_mode_overlays_selected_responses() -> None:
    coordinate = np.array([1600.0, 1500.0, 1400.0, 1300.0, 1200.0])
    structure = _structure()

    _, axes = plotting.plot_pls_coefficients(
        structure,
        predictor_style="line",
        predictor_axis=coordinate,
        predictor_axis_label="Wavenumber (1/cm)",
        response_names=["Yield", "Purity", "Energy demand"],
        responses=[0, 2],
    )

    assert set(axes) == {"coefficients"}
    axis = axes["coefficients"]
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), coordinate)
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), structure.coefficients[0, :])
    np.testing.assert_array_equal(axis.lines[1].get_ydata(), structure.coefficients[2, :])
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Yield",
        "Energy demand",
    ]


def test_plot_pls_y_loadings_groups_components_by_named_response() -> None:
    _, axes = plotting.plot_pls_y_loadings(
        _structure(),
        response_names=["Yield", "Purity", "Energy demand"],
        components=[0, 2],
    )

    assert set(axes) == {"y_loadings"}
    axis = axes["y_loadings"]
    assert len(axis.containers) == 2
    assert all(len(container) == 3 for container in axis.containers)
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "Yield",
        "Purity",
        "Energy demand",
    ]


def test_plot_pls_coefficients_groups_responses_by_named_predictor() -> None:
    _, axes = plotting.plot_pls_coefficients(
        _structure(),
        predictor_style="bar",
        predictor_names=["Temperature", "Pressure", "Flow", "Density", "Viscosity"],
        response_names=["Yield", "Purity", "Energy demand"],
        responses=[1, 2],
    )

    axis = axes["coefficients"]
    assert len(axis.containers) == 2
    assert all(len(container) == 5 for container in axis.containers)
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Purity",
        "Energy demand",
    ]


def test_pls_plotting_writes_pdf(tmp_path: Path) -> None:
    figure, _ = plotting.plot_pls_x_loadings(
        _structure(),
        predictor_style="bar",
        predictor_names=["Temperature", "Pressure", "Flow", "Density", "Viscosity"],
    )
    output = tmp_path / "pls_x_loadings.pdf"

    figure.savefig(output)

    assert output.is_file()
    assert output.stat().st_size > 0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: plotting.plot_pls_scores(
                _structure(),
                components=[0],
            ),
            "exactly two",
        ),
        (
            lambda: plotting.plot_pls_scores(
                _structure(),
                components=[0, 0],
            ),
            "duplicate",
        ),
        (
            lambda: plotting.plot_pls_x_loadings(
                _structure(),
                predictor_style="bar",
            ),
            "predictor_names is required",
        ),
        (
            lambda: plotting.plot_pls_x_loadings(
                _structure(),
                predictor_style="line",
            ),
            "predictor_axis is required",
        ),
        (
            lambda: plotting.plot_pls_y_loadings(
                _structure(),
            ),
            "response_names is required",
        ),
        (
            lambda: plotting.plot_pls_y_loadings(
                _structure(),
                response_names=["a"],
            ),
            "Expected 3",
        ),
        (
            lambda: plotting.plot_pls_coefficients(
                _structure(),
                predictor_style="bar",
                predictor_names=["a", "b", "c", "d", "e"],
            ),
            "response_names is required",
        ),
        (
            lambda: plotting.plot_pls_coefficients(
                _structure(),
                predictor_style="bar",
                predictor_names=["a", "b", "c", "d", "e"],
                response_names=["u", "v", "w"],
                responses=[3],
            ),
            "0 <= index < 3",
        ),
    ],
)
def test_pls_plotting_rejects_invalid_arguments(
    call: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        call()
