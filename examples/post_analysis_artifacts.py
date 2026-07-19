"""Build canonical post-analysis CSV tables and reports derived from them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import ArrayLike

from pipls.inspection import (
    PiPLSDisplayFactors,
    PLSLatentStructure,
    PredictionDiagnostics,
    prediction_diagnostics,
)
from pipls.plotting import (
    PredictorStyle,
    plot_pipls_decomposition,
    plot_pls_coefficients,
    plot_pls_scores,
    plot_pls_x_loadings,
    plot_pls_y_loadings,
    plot_prediction_diagnostics,
)

TABLE_FILENAMES = {
    "pipls_predictor_directions": "pipls_predictor_directions.csv",
    "pipls_response_directions": "pipls_response_directions.csv",
    "predictions": "predictions.csv",
    "pls_scores": "pls_scores.csv",
    "pls_x_loadings": "pls_x_loadings.csv",
    "pls_y_loadings": "pls_y_loadings.csv",
    "pls_coefficients": "pls_coefficients.csv",
}
TABLE_COLUMNS = {
    "pipls_predictor_directions": ("predictor", "component", "p", "component_sign"),
    "pipls_response_directions": (
        "response",
        "component",
        "q",
        "dilation",
        "weighted_q",
        "component_sign",
    ),
    "predictions": (
        "model",
        "sample",
        "fold",
        "response",
        "observed",
        "predicted",
        "residual",
        "observed_standardized",
        "predicted_standardized",
        "residual_standardized",
        "prediction_kind",
    ),
    "pls_scores": ("sample", "component", "score"),
    "pls_x_loadings": ("predictor", "component", "loading"),
    "pls_y_loadings": ("response", "component", "loading"),
    "pls_coefficients": ("response", "predictor", "coefficient"),
}


def build_post_analysis_tables(
    *,
    factors: PiPLSDisplayFactors,
    diagnostics_by_model: Mapping[str, PredictionDiagnostics],
    pls_structure: PLSLatentStructure,
    predictor_names: Sequence[object],
    response_names: Sequence[object],
    sample_names: Sequence[object],
    fold_index: Sequence[int],
) -> dict[str, pd.DataFrame]:
    """Return the seven canonical long-form post-analysis tables."""

    predictors = _labels(predictor_names, factors.n_features, name="predictor_names")
    responses = _labels(response_names, factors.n_targets, name="response_names")
    samples = _labels(sample_names, pls_structure.n_samples, name="sample_names")
    folds = np.asarray(fold_index, dtype=np.int64)
    if folds.shape != (pls_structure.n_samples,) or np.any(folds < 1):
        raise ValueError("fold_index must contain one positive fold number per sample.")
    if factors.n_features != pls_structure.n_features:
        raise ValueError("Pi-PLS and PLS structures must contain the same predictors.")
    if factors.n_targets != pls_structure.n_targets:
        raise ValueError("Pi-PLS and PLS structures must contain the same responses.")

    predictor_rows: list[dict[str, Any]] = []
    for component in range(factors.n_components):
        for predictor, name in enumerate(predictors):
            predictor_rows.append(
                {
                    "predictor": name,
                    "component": component + 1,
                    "p": factors.predictor_directions[predictor, component],
                    "component_sign": int(factors.component_signs[component]),
                }
            )

    response_rows: list[dict[str, Any]] = []
    for component in range(factors.n_components):
        for response, name in enumerate(responses):
            response_rows.append(
                {
                    "response": name,
                    "component": component + 1,
                    "q": factors.response_directions[response, component],
                    "dilation": factors.dilation[component],
                    "weighted_q": factors.weighted_response_directions[response, component],
                    "component_sign": int(factors.component_signs[component]),
                }
            )

    prediction_rows: list[dict[str, Any]] = []
    reference_observed: np.ndarray | None = None
    if not diagnostics_by_model:
        raise ValueError("diagnostics_by_model must contain at least one model.")
    for model_name, diagnostics in diagnostics_by_model.items():
        if not str(model_name).strip():
            raise ValueError("Model names must be nonempty.")
        if diagnostics.n_samples != len(samples) or diagnostics.n_targets != len(responses):
            raise ValueError(
                "Prediction diagnostics must match the supplied samples and responses."
            )
        if reference_observed is None:
            reference_observed = diagnostics.observed
        elif not np.array_equal(diagnostics.observed, reference_observed):
            raise ValueError("All model diagnostics must use identical observed responses.")
        for sample, sample_name in enumerate(samples):
            for response, response_name in enumerate(responses):
                prediction_rows.append(
                    {
                        "model": str(model_name),
                        "sample": sample_name,
                        "fold": int(folds[sample]),
                        "response": response_name,
                        "observed": diagnostics.observed[sample, response],
                        "predicted": diagnostics.predicted[sample, response],
                        "residual": diagnostics.residual[sample, response],
                        "observed_standardized": diagnostics.observed_standardized[
                            sample, response
                        ],
                        "predicted_standardized": diagnostics.predicted_standardized[
                            sample, response
                        ],
                        "residual_standardized": diagnostics.residual_standardized[
                            sample, response
                        ],
                        "prediction_kind": diagnostics.prediction_kind,
                    }
                )

    score_rows = [
        {
            "sample": sample_name,
            "component": component + 1,
            "score": pls_structure.x_scores[sample, component],
        }
        for component in range(pls_structure.n_components)
        for sample, sample_name in enumerate(samples)
    ]
    x_loading_rows = [
        {
            "predictor": predictor_name,
            "component": component + 1,
            "loading": pls_structure.x_loadings[predictor, component],
        }
        for component in range(pls_structure.n_components)
        for predictor, predictor_name in enumerate(predictors)
    ]
    y_loading_rows = [
        {
            "response": response_name,
            "component": component + 1,
            "loading": pls_structure.y_loadings[response, component],
        }
        for component in range(pls_structure.n_components)
        for response, response_name in enumerate(responses)
    ]
    coefficient_rows = [
        {
            "response": response_name,
            "predictor": predictor_name,
            "coefficient": pls_structure.coefficients[response, predictor],
        }
        for response, response_name in enumerate(responses)
        for predictor, predictor_name in enumerate(predictors)
    ]

    return {
        "pipls_predictor_directions": pd.DataFrame(
            predictor_rows, columns=TABLE_COLUMNS["pipls_predictor_directions"]
        ),
        "pipls_response_directions": pd.DataFrame(
            response_rows, columns=TABLE_COLUMNS["pipls_response_directions"]
        ),
        "predictions": pd.DataFrame(prediction_rows, columns=TABLE_COLUMNS["predictions"]),
        "pls_scores": pd.DataFrame(score_rows, columns=TABLE_COLUMNS["pls_scores"]),
        "pls_x_loadings": pd.DataFrame(
            x_loading_rows, columns=TABLE_COLUMNS["pls_x_loadings"]
        ),
        "pls_y_loadings": pd.DataFrame(
            y_loading_rows, columns=TABLE_COLUMNS["pls_y_loadings"]
        ),
        "pls_coefficients": pd.DataFrame(
            coefficient_rows, columns=TABLE_COLUMNS["pls_coefficients"]
        ),
    }


def write_post_analysis_tables(
    output_dir: Path,
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, Path]:
    """Validate and write all canonical tables with deterministic filenames."""

    missing = [name for name in TABLE_FILENAMES if name not in tables]
    extra = [name for name in tables if name not in TABLE_FILENAMES]
    if missing or extra:
        raise ValueError(
            "Post-analysis tables differ from the contract: "
            f"missing={missing}, extra={extra}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, filename in TABLE_FILENAMES.items():
        table = tables[name]
        expected = TABLE_COLUMNS[name]
        if tuple(table.columns) != expected:
            raise ValueError(
                f"{name} columns must be {expected!r}; got {tuple(table.columns)!r}."
            )
        if table.empty:
            raise ValueError(f"{name} must contain at least one row.")
        path = output_dir / filename
        table.to_csv(path, index=False, float_format="%.17g")
        paths[name] = path
    return paths


def render_post_analysis_report(
    output_dir: Path,
    pdf_path: Path,
    *,
    dataset_name: str,
    predictor_style: PredictorStyle = "bar",
    predictor_axis: ArrayLike | None = None,
    predictor_axis_label: str | None = None,
    pls_score_components: Sequence[int],
    pls_loading_components: Sequence[int],
    coefficient_responses: Sequence[object],
) -> None:
    """Read the canonical CSV files and render one multipage report."""

    tables = {
        name: _read_table(output_dir / filename, name=name)
        for name, filename in TABLE_FILENAMES.items()
    }
    factors, predictor_names, response_names = _factors_from_tables(tables)
    pls_structure, sample_names = _pls_structure_from_tables(
        tables,
        predictor_names=predictor_names,
        response_names=response_names,
    )
    diagnostics_by_model = _diagnostics_from_table(
        tables["predictions"],
        sample_names=sample_names,
        response_names=response_names,
    )
    score_components = _component_indices(
        pls_score_components,
        size=pls_structure.n_components,
        argument_name="pls_score_components",
        required_count=2,
    )
    loading_components = _component_indices(
        pls_loading_components,
        size=pls_structure.n_components,
        argument_name="pls_loading_components",
    )
    selected_responses = _named_indices(
        coefficient_responses,
        labels=response_names,
        argument_name="coefficient_responses",
    )

    import matplotlib.pyplot as plt

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as report:
        figure, _ = plot_pipls_decomposition(
            factors,
            predictor_style=predictor_style,
            predictor_names=predictor_names,
            response_names=response_names,
            predictor_axis=predictor_axis,
            predictor_axis_label=predictor_axis_label,
            title=f"{dataset_name} Pi-PLS decomposition",
        )
        report.savefig(figure)
        plt.close(figure)

        for model_name, diagnostics in diagnostics_by_model.items():
            figure, _ = plot_prediction_diagnostics(
                diagnostics,
                response_names=response_names,
                title=f"{dataset_name} {model_name} prediction diagnostics",
            )
            report.savefig(figure)
            plt.close(figure)

        figure, _ = plot_pls_scores(
            pls_structure,
            components=score_components,
            title=f"{dataset_name} ordinary PLS scores",
        )
        report.savefig(figure)
        plt.close(figure)

        figure, _ = plot_pls_x_loadings(
            pls_structure,
            predictor_style=predictor_style,
            predictor_names=predictor_names,
            predictor_axis=predictor_axis,
            predictor_axis_label=predictor_axis_label,
            components=loading_components,
            title=f"{dataset_name} ordinary PLS X loadings",
        )
        report.savefig(figure)
        plt.close(figure)

        figure, _ = plot_pls_y_loadings(
            pls_structure,
            response_names=response_names,
            components=loading_components,
            title=f"{dataset_name} ordinary PLS Y loadings",
        )
        report.savefig(figure)
        plt.close(figure)

        figure, _ = plot_pls_coefficients(
            pls_structure,
            predictor_style=predictor_style,
            predictor_names=predictor_names,
            response_names=response_names,
            predictor_axis=predictor_axis,
            predictor_axis_label=predictor_axis_label,
            responses=selected_responses,
            title=f"{dataset_name} ordinary PLS coefficients",
        )
        report.savefig(figure)
        plt.close(figure)


def _read_table(path: Path, *, name: str) -> pd.DataFrame:
    expected = TABLE_COLUMNS[name]
    identifier_columns = {"model", "sample", "response", "predictor", "prediction_kind"}
    dtypes = {column: str for column in expected if column in identifier_columns}
    table = pd.read_csv(path, dtype=dtypes)
    if tuple(table.columns) != expected:
        raise ValueError(f"{path.name} columns must be {expected!r}.")
    if table.empty:
        raise ValueError(f"{path.name} must contain at least one row.")
    return table


def _factors_from_tables(
    tables: Mapping[str, pd.DataFrame],
) -> tuple[PiPLSDisplayFactors, tuple[str, ...], tuple[str, ...]]:
    predictor_table = tables["pipls_predictor_directions"]
    response_table = tables["pipls_response_directions"]
    predictor_names = tuple(predictor_table["predictor"].drop_duplicates().astype(str))
    response_names = tuple(response_table["response"].drop_duplicates().astype(str))
    components = tuple(sorted(int(value) for value in predictor_table["component"].unique()))
    if components != tuple(range(1, len(components) + 1)):
        raise ValueError("Pi-PLS component numbers must be consecutive and one-based.")

    predictor_directions = _pivot(
        predictor_table,
        index="predictor",
        columns="component",
        values="p",
        row_order=predictor_names,
        column_order=components,
    )
    response_directions = _pivot(
        response_table,
        index="response",
        columns="component",
        values="q",
        row_order=response_names,
        column_order=components,
    )
    weighted = _pivot(
        response_table,
        index="response",
        columns="component",
        values="weighted_q",
        row_order=response_names,
        column_order=components,
    )
    dilation = _one_per_component(response_table, "dilation", components).astype(np.float64)
    signs = _one_per_component(predictor_table, "component_sign", components).astype(np.int8)
    if not np.allclose(weighted, response_directions * dilation[None, :]):
        raise ValueError("weighted_q must equal q multiplied by dilation.")

    return (
        PiPLSDisplayFactors(
            predictor_directions=predictor_directions,
            dilation=dilation,
            response_directions=response_directions,
            weighted_response_directions=weighted,
            component_signs=signs,
        ),
        predictor_names,
        response_names,
    )


def _pls_structure_from_tables(
    tables: Mapping[str, pd.DataFrame],
    *,
    predictor_names: Sequence[str],
    response_names: Sequence[str],
) -> tuple[PLSLatentStructure, tuple[str, ...]]:
    score_table = tables["pls_scores"]
    sample_names = tuple(score_table["sample"].drop_duplicates().astype(str))
    components = tuple(sorted(int(value) for value in score_table["component"].unique()))
    if components != tuple(range(1, len(components) + 1)):
        raise ValueError("PLS component numbers must be consecutive and one-based.")
    x_scores = _pivot(
        score_table,
        index="sample",
        columns="component",
        values="score",
        row_order=sample_names,
        column_order=components,
    )
    x_loadings = _pivot(
        tables["pls_x_loadings"],
        index="predictor",
        columns="component",
        values="loading",
        row_order=predictor_names,
        column_order=components,
    )
    y_loadings = _pivot(
        tables["pls_y_loadings"],
        index="response",
        columns="component",
        values="loading",
        row_order=response_names,
        column_order=components,
    )
    coefficients = _pivot(
        tables["pls_coefficients"],
        index="response",
        columns="predictor",
        values="coefficient",
        row_order=response_names,
        column_order=predictor_names,
    )
    return (
        PLSLatentStructure(
            x_scores=x_scores,
            x_loadings=x_loadings,
            y_loadings=y_loadings,
            coefficients=coefficients,
        ),
        sample_names,
    )


def _diagnostics_from_table(
    table: pd.DataFrame,
    *,
    sample_names: Sequence[str],
    response_names: Sequence[str],
) -> dict[str, PredictionDiagnostics]:
    diagnostics: dict[str, PredictionDiagnostics] = {}
    for model_name in table["model"].drop_duplicates().astype(str):
        model_table = table.loc[table["model"].astype(str) == model_name]
        kinds = model_table["prediction_kind"].drop_duplicates().astype(str).tolist()
        if len(kinds) != 1:
            raise ValueError("Each model must have exactly one prediction_kind value.")
        observed = _pivot(
            model_table,
            index="sample",
            columns="response",
            values="observed",
            row_order=sample_names,
            column_order=response_names,
        )
        predicted = _pivot(
            model_table,
            index="sample",
            columns="response",
            values="predicted",
            row_order=sample_names,
            column_order=response_names,
        )
        diagnostics[model_name] = prediction_diagnostics(
            observed,
            predicted,
            prediction_kind=kinds[0],  # type: ignore[arg-type]
        )
    return diagnostics


def _pivot(
    table: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
    row_order: Sequence[object],
    column_order: Sequence[object],
) -> np.ndarray:
    if table.duplicated([index, columns]).any():
        raise ValueError(f"{values} table must contain one row per {index}/{columns} pair.")
    pivoted = table.pivot(index=index, columns=columns, values=values)
    try:
        ordered = pivoted.loc[list(row_order), list(column_order)]
    except KeyError as error:
        raise ValueError(f"{values} table does not contain the required labels.") from error
    result = ordered.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{values} table must contain only finite values.")
    return result


def _one_per_component(
    table: pd.DataFrame,
    column: str,
    components: Sequence[int],
) -> np.ndarray:
    values: list[float] = []
    for component in components:
        unique = table.loc[table["component"] == component, column].drop_duplicates()
        if len(unique) != 1:
            raise ValueError(f"{column} must contain one value per component.")
        values.append(float(unique.iloc[0]))
    return np.asarray(values)


def _labels(values: Sequence[object], size: int, *, name: str) -> tuple[str, ...]:
    labels = tuple(str(value) for value in values)
    if len(labels) != size:
        raise ValueError(f"Expected {size} values in {name}, got {len(labels)}.")
    if any(not label.strip() for label in labels):
        raise ValueError(f"{name} must contain only nonempty labels.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} must not contain duplicate labels.")
    return labels


def _component_indices(
    values: Sequence[int],
    *,
    size: int,
    argument_name: str,
    required_count: int | None = None,
) -> tuple[int, ...]:
    selected = tuple(values)
    if required_count is not None and len(selected) != required_count:
        raise ValueError(f"{argument_name} must contain exactly {required_count} components.")
    if not selected:
        raise ValueError(f"{argument_name} must contain at least one component.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in selected):
        raise ValueError(f"{argument_name} must contain one-based integer components.")
    if any(value < 1 or value > size for value in selected):
        raise ValueError(f"{argument_name} values must satisfy 1 <= component <= {size}.")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{argument_name} must not contain duplicates.")
    return tuple(value - 1 for value in selected)


def _named_indices(
    values: Sequence[object],
    *,
    labels: Sequence[str],
    argument_name: str,
) -> tuple[int, ...]:
    selected = tuple(str(value) for value in values)
    if not selected:
        raise ValueError(f"{argument_name} must contain at least one response name.")
    unknown = [value for value in selected if value not in labels]
    if unknown:
        raise ValueError(f"{argument_name} contains unknown response names: {unknown!r}.")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{argument_name} must not contain duplicates.")
    return tuple(labels.index(value) for value in selected)
