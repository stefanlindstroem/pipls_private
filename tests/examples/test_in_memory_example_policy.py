from __future__ import annotations

import ast
from pathlib import Path

_DATASET_FILENAMES = {"X.csv", "Y.csv"}
_REAL_DATA_EXAMPLES = {
    "09_pls_path_comparison.py",
    "10_pulp_real_data.py",
    "11_sugarcane_real_data.py",
    "12_tobacco_real_data.py",
}
_INSPECTION_CALLS = {
    "latent_structure",
    "pipls_display_factors",
    "prediction_diagnostics",
}
_OLD_RESULT_NAMES = {
    "component_path_results_",
    "best_predictor_rank_by_n_components_",
    "best_score_by_n_components_",
    "response_standardized_mse_path_",
    "score_path_",
}
_OLD_HELPER_NAMES = {
    "build_post_analysis_tables",
    "fixed_model_oof_predictions",
    "plot_pipls_component_path",
    "render_post_analysis_report",
    "run_pulp_workflow",
    "write_post_analysis_tables",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _numbered_examples() -> list[Path]:
    return sorted((_repository_root() / "examples").glob("[0-9][0-9]_*.py"))


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _string_literals(node: ast.AST) -> set[str]:
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def test_numbered_examples_use_csv_only_for_committed_dataset_inputs() -> None:
    for path in _numbered_examples():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        assert ".to_csv(" not in source, path
        assert "pd.DataFrame(" not in source, path

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) != "read_csv":
                continue
            assert path.name in _REAL_DATA_EXAMPLES, path
            assert len(node.args) == 1, path
            literals = _string_literals(node.args[0])
            assert literals & _DATASET_FILENAMES, path
            assert not {value for value in literals if value.endswith(".csv")} - _DATASET_FILENAMES


def test_examples_results_contain_no_tracked_csv_products() -> None:
    results = _repository_root() / "examples" / "results"
    assert list(results.rglob("*.csv")) == []


def test_real_data_examples_own_paths_figures_and_inspection_results() -> None:
    examples = _repository_root() / "examples"
    for filename in sorted(_REAL_DATA_EXAMPLES):
        path = examples / filename
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        calls = {
            name
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and (name := _call_name(node)) is not None
        }

        assert "component_path_" in source, path
        assert "subplots" in calls, path
        assert "savefig" in calls, path

        if filename != "09_pls_path_comparison.py":
            assert "cross_val_predict" in calls, path
            assert _INSPECTION_CALLS <= calls, path


def test_example_support_does_not_read_or_plot_serialized_analysis_results() -> None:
    support = _repository_root() / "examples" / "_support"
    assert not (support / "plot_component_path.py").exists()

    prohibited_calls = {"genfromtxt", "load", "loadtxt", "open", "read_csv", "read_table"}
    for path in support.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls = {
            name
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and (name := _call_name(node)) is not None
        }
        assert calls.isdisjoint(prohibited_calls), path


def test_tutorial_renderer_uses_the_direct_in_memory_public_sequence() -> None:
    source = (_repository_root() / "tools" / "render_pulp_tutorial.py").read_text(
        encoding="utf-8"
    )

    assert "PiPLSPathCV(refit=False).fit(X, Y)" in source
    assert "component_path_" in source
    assert "predictor_rank_profile(" in source
    assert "cross_val_predict(" in source
    for name in _OLD_HELPER_NAMES:
        assert name not in source
    assert ".to_csv(" not in source


def test_removed_result_attributes_and_helpers_are_absent() -> None:
    root = _repository_root()
    maintained_sources = [
        *sorted((root / "src").rglob("*.py")),
        *sorted((root / "examples").rglob("*.py")),
        *sorted((root / "tools").rglob("*.py")),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in maintained_sources)

    for name in _OLD_RESULT_NAMES | _OLD_HELPER_NAMES:
        assert name not in combined

    support = root / "examples" / "_support"
    for filename in (
        "fixed_model_oof.py",
        "plot_component_path.py",
        "post_analysis_artifacts.py",
        "pulp_workflow.py",
    ):
        assert not (support / filename).exists()
