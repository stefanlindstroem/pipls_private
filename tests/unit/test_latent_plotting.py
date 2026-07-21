from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure
from sklearn.cross_decomposition import PLSRegression

import pipls.plotting as plotting
from pipls.inspection import (
    LatentStructure,
    biplot_coordinates,
    latent_structure,
    observation_diagnostics,
)

matplotlib.use("Agg")


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    yield
    _pyplot().close("all")


def _structure() -> LatentStructure:
    rng = np.random.default_rng(932)
    X = rng.normal(size=(28, 5))
    Y = X[:, :3] @ np.array(
        [
            [1.0, -0.2, 0.4],
            [0.5, 0.7, -0.1],
            [-0.4, 0.3, 0.8],
        ]
    ) + 0.05 * rng.normal(size=(28, 3))
    return latent_structure(PLSRegression(n_components=3).fit(X, Y))


def _observation_result():
    rng = np.random.default_rng(181)
    X = rng.normal(size=(32, 6))
    Y = X[:, :2] + 0.1 * rng.normal(size=(32, 2))
    model = PLSRegression(n_components=2).fit(X, Y)
    return observation_diagnostics(model, X)


def test_plot_scores_returns_one_axis_and_selected_components() -> None:
    figure, axis = plotting.plot_scores(
        _structure(),
        components=(0, 2),
        sample_names=[f"Sample {index + 1}" for index in range(28)],
        title="PLS score review",
    )

    assert isinstance(figure, Figure)
    assert figure.axes == [axis]
    assert axis.get_xlabel() == "X score component 1"
    assert axis.get_ylabel() == "X score component 3"
    assert axis.get_title() == "PLS score review"
    assert len(axis.texts) == 28


def test_plot_x_loadings_line_mode_overlays_components_and_preserves_coordinate() -> None:
    coordinate = np.array([1600.0, 1500.0, 1400.0, 1300.0, 1200.0])

    _, axis = plotting.plot_x_loadings(
        _structure(),
        predictor_style="line",
        predictor_axis=coordinate,
        predictor_axis_label="Wavenumber (1/cm)",
        components=[0, 2],
    )

    np.testing.assert_array_equal(axis.lines[0].get_xdata(), coordinate)
    np.testing.assert_array_equal(axis.lines[1].get_xdata(), coordinate)
    assert axis.get_xlim() == (coordinate[0], coordinate[-1])
    assert axis.get_xlabel() == "Wavenumber (1/cm)"
    assert axis.get_legend() is None
    assert axis.get_legend_handles_labels()[1] == ["Component 1", "Component 3"]


def test_plot_x_loadings_groups_component_bars_by_predictor() -> None:
    _, axis = plotting.plot_x_loadings(
        _structure(),
        predictor_style="bar",
        predictor_names=["Temperature", "Pressure", "Flow", "Density", "Viscosity"],
        components=[0, 1, 2],
    )

    assert len(axis.containers) == 3
    assert all(len(container) == 5 for container in axis.containers)
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "Temperature",
        "Pressure",
        "Flow",
        "Density",
        "Viscosity",
    ]
    assert axis.get_legend() is None


def test_plot_coefficients_line_mode_overlays_selected_responses() -> None:
    coordinate = np.array([1600.0, 1500.0, 1400.0, 1300.0, 1200.0])
    structure = _structure()

    _, axis = plotting.plot_coefficients(
        structure,
        predictor_style="line",
        predictor_axis=coordinate,
        predictor_axis_label="Wavenumber (1/cm)",
        response_names=["Yield", "Purity", "Energy demand"],
        responses=[0, 2],
    )

    np.testing.assert_array_equal(axis.lines[0].get_xdata(), coordinate)
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), structure.coefficients[0, :])
    np.testing.assert_array_equal(axis.lines[1].get_ydata(), structure.coefficients[2, :])
    assert axis.get_legend() is None
    assert axis.get_legend_handles_labels()[1] == ["Yield", "Energy demand"]


def test_plot_y_loadings_groups_components_by_named_response() -> None:
    _, axis = plotting.plot_y_loadings(
        _structure(),
        response_names=["Yield", "Purity", "Energy demand"],
        components=[0, 2],
    )

    assert len(axis.containers) == 2
    assert all(len(container) == 3 for container in axis.containers)
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "Yield",
        "Purity",
        "Energy demand",
    ]
    assert axis.get_legend() is None


def test_plot_coefficients_groups_responses_by_named_predictor() -> None:
    _, axis = plotting.plot_coefficients(
        _structure(),
        predictor_style="bar",
        predictor_names=["Temperature", "Pressure", "Flow", "Density", "Viscosity"],
        response_names=["Yield", "Purity", "Energy demand"],
        responses=[1, 2],
    )

    assert len(axis.containers) == 2
    assert all(len(container) == 5 for container in axis.containers)
    assert axis.get_legend() is None
    assert axis.get_legend_handles_labels()[1] == ["Purity", "Energy demand"]


def test_plot_observation_diagnostics_returns_one_raw_scatter_axis() -> None:
    figure, axis = plotting.plot_observation_diagnostics(
        _observation_result(),
        title="Observation review",
    )

    assert isinstance(figure, Figure)
    assert figure.axes == [axis]
    assert axis.get_xlabel() == "Score distance"
    assert axis.get_ylabel() == "Squared X-reconstruction residual"
    assert axis.get_title() == "Observation review"
    assert len(axis.collections) == 1


@pytest.mark.parametrize(
    "call",
    [
        lambda axis: plotting.plot_scores(_structure(), ax=axis),
        lambda axis: plotting.plot_x_loadings(
            _structure(),
            predictor_style="bar",
            predictor_names=["A", "B", "C", "D", "E"],
            ax=axis,
        ),
        lambda axis: plotting.plot_y_loadings(
            _structure(),
            response_names=["U", "V", "W"],
            ax=axis,
        ),
        lambda axis: plotting.plot_coefficients(
            _structure(),
            predictor_style="bar",
            predictor_names=["A", "B", "C", "D", "E"],
            response_names=["U", "V", "W"],
            ax=axis,
        ),
        lambda axis: plotting.plot_biplot(
            biplot_coordinates(_structure()),
            predictor_names=["A", "B", "C", "D", "E"],
            ax=axis,
        ),
        lambda axis: plotting.plot_observation_diagnostics(
            _observation_result(),
            ax=axis,
        ),
    ],
)
def test_all_atomic_plotters_accept_a_caller_axis(
    call: Callable[[object], tuple[Figure, object]],
) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots()
    figure_numbers = plt.get_fignums()

    returned_figure, returned_axis = call(axis)

    assert returned_figure is figure
    assert returned_axis is axis
    assert figure.axes == [axis]
    assert plt.get_fignums() == figure_numbers


def test_atomic_plotter_uses_supplied_axis_without_clearing_or_creating_a_figure() -> None:
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.0))
    supplied = axes[1]
    supplied.plot([0.0, 1.0], [1.0, 0.0], label="Caller content")
    original_size = tuple(figure.get_size_inches())
    figure_numbers = plt.get_fignums()

    returned_figure, returned_axis = plotting.plot_x_loadings(
        _structure(),
        predictor_style="bar",
        predictor_names=["Temperature", "Pressure", "Flow", "Density", "Viscosity"],
        components=[0, 2],
        ax=supplied,
    )

    assert returned_figure is figure
    assert returned_axis is supplied
    assert figure.axes == list(axes)
    assert tuple(figure.get_size_inches()) == original_size
    assert plt.get_fignums() == figure_numbers
    assert supplied.lines[0].get_label() == "Caller content"
    assert supplied.get_legend() is None


def test_caller_owns_legend_and_can_replace_semantic_labels() -> None:
    _, axis = plotting.plot_x_loadings(
        _structure(),
        predictor_style="line",
        predictor_axis=np.arange(5, dtype=float),
        predictor_axis_label="Original coordinate",
        components=[0, 2],
    )

    axis.set_xlabel("Frequency")
    axis.set_ylabel("Loading coefficient")
    axis.set_title("Selected loading curves")
    legend = axis.legend(title="Latent variable")

    assert axis.get_xlabel() == "Frequency"
    assert axis.get_ylabel() == "Loading coefficient"
    assert axis.get_title() == "Selected loading curves"
    assert legend.get_title().get_text() == "Latent variable"


def test_figsize_is_rejected_with_a_supplied_axis() -> None:
    _, axis = _pyplot().subplots()

    with pytest.raises(ValueError, match="figsize cannot be supplied"):
        plotting.plot_scores(_structure(), ax=axis, figsize=(5.0, 4.0))


def test_invalid_axis_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="matplotlib.axes.Axes"):
        plotting.plot_scores(_structure(), ax=object())  # type: ignore[arg-type]


def test_shared_plotting_writes_pdf(tmp_path: Path) -> None:
    figure, _ = plotting.plot_x_loadings(
        _structure(),
        predictor_style="bar",
        predictor_names=["Temperature", "Pressure", "Flow", "Density", "Viscosity"],
    )
    output = tmp_path / "x_loadings.pdf"

    figure.savefig(output)

    assert output.is_file()
    assert output.stat().st_size > 0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: plotting.plot_scores(
                _structure(),
                components=[0],
            ),
            "exactly two",
        ),
        (
            lambda: plotting.plot_scores(
                _structure(),
                components=[0, 0],
            ),
            "duplicate",
        ),
        (
            lambda: plotting.plot_x_loadings(
                _structure(),
                predictor_style="bar",
            ),
            "predictor_names is required",
        ),
        (
            lambda: plotting.plot_x_loadings(
                _structure(),
                predictor_style="line",
            ),
            "predictor_axis is required",
        ),
        (
            lambda: plotting.plot_y_loadings(
                _structure(),
            ),
            "response_names is required",
        ),
        (
            lambda: plotting.plot_y_loadings(
                _structure(),
                response_names=["a"],
            ),
            "Expected 3",
        ),
        (
            lambda: plotting.plot_coefficients(
                _structure(),
                predictor_style="bar",
                predictor_names=["a", "b", "c", "d", "e"],
            ),
            "response_names is required",
        ),
        (
            lambda: plotting.plot_coefficients(
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
def test_shared_plotting_rejects_invalid_arguments(
    call: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_plot_biplot_uses_balanced_coordinates_and_predictor_labels() -> None:
    structure = _structure()
    coordinates = biplot_coordinates(structure, components=(0, 1))

    figure, axis = plotting.plot_biplot(
        coordinates,
        predictor_names=["Temperature", "Pressure", "Flow", "Density", "Viscosity"],
        title="Pulp-like PLS biplot",
    )

    assert isinstance(figure, Figure)
    assert figure.axes == [axis]
    assert axis.get_xlabel() == "Balanced component 1"
    assert axis.get_ylabel() == "Balanced component 2"
    assert axis.get_title() == "Pulp-like PLS biplot"
    assert axis.get_legend() is None
    assert len(axis.collections) == 1
    assert [text.get_text() for text in axis.texts] == [
        "Temperature",
        "Pressure",
        "Flow",
        "Density",
        "Viscosity",
    ]


def test_plot_biplot_accepts_optional_sample_labels() -> None:
    structure = _structure()
    coordinates = biplot_coordinates(structure)
    sample_names = [f"Sample {index + 1}" for index in range(structure.n_samples)]

    _, axis = plotting.plot_biplot(
        coordinates,
        predictor_names=["A", "B", "C", "D", "E"],
        sample_names=sample_names,
    )

    texts = [text.get_text() for text in axis.texts]
    assert texts[: structure.n_samples] == sample_names
    assert texts[structure.n_samples :] == ["A", "B", "C", "D", "E"]


def test_plot_biplot_requires_predictor_names() -> None:
    coordinates = biplot_coordinates(_structure())

    with pytest.raises(ValueError, match="Expected 5"):
        plotting.plot_biplot(
            coordinates,
            predictor_names=["A"],
        )
