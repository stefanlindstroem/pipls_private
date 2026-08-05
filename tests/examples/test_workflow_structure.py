from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests._source_contracts import (
    call_lines,
    call_names,
    calls_named,
    dotted_name,
    import_roots,
    imported_names,
    keyword_constant,
    module_scope,
    parse_module,
    private_pipls_imports,
    top_level_functions,
)

_RENDERING_PACKAGES = {"matplotlib", "adjustText"}
_RENDERING_METHODS = {
    "add_patch",
    "bar",
    "errorbar",
    "figure",
    "plot",
    "quiver",
    "savefig",
    "scatter",
    "subplots",
    "text",
}
_ANALYSIS_CALLS = {
    "latent_structure",
    "observation_diagnostics",
    "oof_report",
    "pipls_display_factors",
    "prediction_diagnostics",
    "predictor_rank_profile",
}
_MANUAL_WORKFLOWS = (
    "examples/02_synthetic_path_selection.py",
    "examples/05_pulp_real_data.py",
    "examples/06_sugarcane_real_data.py",
    "examples/07_tobacco_real_data.py",
    "tools/render_synthetic_tutorial.py",
    "tools/render_pulp_tutorial.py",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _numbered_examples() -> list[Path]:
    return sorted((_repository_root() / "examples").glob("[0-9][0-9]_*.py"))


def _maintained_python_files() -> list[Path]:
    root = _repository_root()
    return [
        *_numbered_examples(),
        *sorted((root / "tools").glob("render_*_tutorial.py")),
    ]


def _workflow_scope(tree: ast.Module) -> ast.AST:
    top_level = module_scope(tree)
    if "PiPLSSearchCV" in call_names(top_level):
        return top_level

    candidates = [
        function
        for function in top_level_functions(tree).values()
        if "PiPLSSearchCV" in call_names(function)
    ]
    assert len(candidates) == 1
    return candidates[0]


def _rendering_function_names(tree: ast.Module) -> set[str]:
    return {
        name
        for name, function in top_level_functions(tree).items()
        if call_names(function) & _RENDERING_METHODS
    }


def _keyword_path(call: ast.Call, keyword_name: str) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            return dotted_name(keyword.value)
    return None


@pytest.mark.parametrize(
    ("relative_path", "splitter", "expected"),
    [
        (
            "examples/02_synthetic_path_selection.py",
            "KFold",
            {"n_splits": 5, "shuffle": True, "random_state": 0},
        ),
        (
            "examples/04_pls_path_comparison.py",
            "KFold",
            {"n_splits": 5, "shuffle": True, "random_state": 0},
        ),
        (
            "examples/05_pulp_real_data.py",
            "RepeatedKFold",
            {"n_splits": 5, "n_repeats": 10, "random_state": 0},
        ),
        (
            "examples/06_sugarcane_real_data.py",
            "KFold",
            {"n_splits": 5, "shuffle": True, "random_state": 0},
        ),
        (
            "examples/07_tobacco_real_data.py",
            "KFold",
            {"n_splits": 5, "shuffle": True, "random_state": 0},
        ),
        (
            "tools/render_synthetic_tutorial.py",
            "KFold",
            {"n_splits": 5, "shuffle": True, "random_state": 0},
        ),
        (
            "tools/render_pulp_tutorial.py",
            "RepeatedKFold",
            {"n_splits": 5, "n_repeats": 10, "random_state": 0},
        ),
    ],
)
def test_randomized_cv_is_explicitly_seeded(
    relative_path: str,
    splitter: str,
    expected: dict[str, object],
) -> None:
    tree = parse_module(_repository_root() / relative_path)
    splitter_calls = calls_named(tree, splitter)

    assert splitter_calls
    for call in splitter_calls:
        assert {
            keyword: keyword_constant(call, keyword)
            for keyword in expected
        } == expected


def test_leave_one_out_example_uses_exhaustive_validation() -> None:
    tree = parse_module(
        _repository_root() / "examples" / "03_leave_one_out_validation.py"
    )
    calls = call_names(tree)

    assert "LeaveOneOut" in calls
    assert {"KFold", "RepeatedKFold", "refit"}.isdisjoint(calls)
    select_calls = calls_named(tree, "select")
    assert len(select_calls) == 1
    assert keyword_constant(select_calls[0], "rule") == "best_score"

    report_calls = calls_named(tree, "oof_report")
    assert len(report_calls) == 1
    assert _keyword_path(report_calls[0], "selection") is not None


def test_examples_import_only_public_pipls_modules_and_names() -> None:
    for path in _maintained_python_files():
        private_imports = private_pipls_imports(parse_module(path))
        assert not private_imports, f"{path}: private imports {sorted(private_imports)}"


@pytest.mark.parametrize(
    ("loader_name", "relative_paths"),
    [
        (
            "load_pulp",
            (
                "examples/01_pulp_quick_start.py",
                "examples/04_pls_path_comparison.py",
                "examples/05_pulp_real_data.py",
                "tools/render_pulp_tutorial.py",
                "tools/render_quick_start_tutorial.py",
            ),
        ),
        (
            "load_sugarcane",
            (
                "examples/04_pls_path_comparison.py",
                "examples/06_sugarcane_real_data.py",
            ),
        ),
        (
            "load_tobacco",
            (
                "examples/04_pls_path_comparison.py",
                "examples/07_tobacco_real_data.py",
            ),
        ),
    ],
)
def test_reference_workflows_use_package_owned_dataset_loaders(
    loader_name: str,
    relative_paths: tuple[str, ...],
) -> None:
    root = _repository_root()
    for relative_path in relative_paths:
        path = root / relative_path
        tree = parse_module(path)
        assert loader_name in imported_names(tree), path
        assert loader_name in call_names(tree), path


def test_ordinary_pls_is_confined_to_the_comparison_support() -> None:
    examples = _repository_root() / "examples"
    users = {
        path.relative_to(examples).as_posix()
        for path in examples.rglob("*.py")
        if "PLSRegression" in imported_names(parse_module(path))
        or "PLSRegression" in call_names(parse_module(path))
    }

    assert users == {"_support/pls_component_path.py"}
    comparison_calls = call_names(parse_module(examples / "04_pls_path_comparison.py"))
    assert "evaluate_pls_component_path" in comparison_calls
    for filename in (
        "05_pulp_real_data.py",
        "06_sugarcane_real_data.py",
        "07_tobacco_real_data.py",
    ):
        assert "evaluate_pls_component_path" not in call_names(
            parse_module(examples / filename)
        )


@pytest.mark.parametrize("relative_path", _MANUAL_WORKFLOWS)
def test_modeling_precedes_analysis_and_rendering(relative_path: str) -> None:
    tree = parse_module(_repository_root() / relative_path)
    scope = _workflow_scope(tree)

    refit_lines = call_lines(scope, {"refit"})
    analysis_lines = call_lines(scope, _ANALYSIS_CALLS)
    rendering_names = _RENDERING_METHODS | _rendering_function_names(tree)
    rendering_lines = call_lines(scope, rendering_names)

    assert len(refit_lines) == 1
    assert analysis_lines
    assert rendering_lines
    assert refit_lines[0] < min(analysis_lines)
    assert max(analysis_lines) < min(rendering_lines)


@pytest.mark.parametrize(
    "relative_path",
    [
        "examples/06_sugarcane_real_data.py",
        "examples/07_tobacco_real_data.py",
    ],
)
def test_complete_examples_separate_analysis_from_rendering(
    relative_path: str,
) -> None:
    tree = parse_module(_repository_root() / relative_path)
    scope = _workflow_scope(tree)
    functions = top_level_functions(tree)
    rendering_names = _rendering_function_names(tree)

    assert rendering_names
    assert call_names(scope).isdisjoint(_RENDERING_METHODS)
    assert rendering_names <= call_names(scope)
    for name in rendering_names:
        assert call_names(functions[name]).isdisjoint(_ANALYSIS_CALLS | {"PiPLSSearchCV", "refit"})


@pytest.mark.parametrize(
    ("relative_path", "extra_analysis_calls"),
    [
        ("examples/05_pulp_real_data.py", set()),
        ("examples/06_sugarcane_real_data.py", {"predictor_rank_profile"}),
        (
            "examples/07_tobacco_real_data.py",
            {"observation_diagnostics", "predictor_rank_profile"},
        ),
    ],
)
def test_real_data_examples_use_public_post_fit_analysis(
    relative_path: str,
    extra_analysis_calls: set[str],
) -> None:
    calls = call_names(parse_module(_repository_root() / relative_path))
    required = {
        "latent_structure",
        "oof_report",
        "pipls_display_factors",
        "prediction_diagnostics",
        *extra_analysis_calls,
    }

    assert required <= calls
    assert "select" not in calls
    assert "refit" in calls


@pytest.mark.parametrize(
    "relative_path",
    [
        "examples/06_sugarcane_real_data.py",
        "examples/07_tobacco_real_data.py",
    ],
)
def test_spectral_examples_request_rank_profile_at_a_selected_component_count(
    relative_path: str,
) -> None:
    calls = calls_named(parse_module(_repository_root() / relative_path), "predictor_rank_profile")

    assert len(calls) == 1
    assert calls[0].args
    argument = dotted_name(calls[0].args[0])
    assert argument is not None
    assert argument.endswith(".n_components")


def test_tobacco_keeps_explicit_robust_selection_and_paginated_reports() -> None:
    tree = parse_module(_repository_root() / "examples" / "07_tobacco_real_data.py")

    regression_calls = calls_named(tree, "PiPLSRegression")
    assert len(regression_calls) == 1
    assert keyword_constant(regression_calls[0], "svd_solver") == "full"

    refit_calls = calls_named(tree, "refit")
    assert len(refit_calls) == 1
    assert keyword_constant(refit_calls[0], "rule") == "minimum_cv_mse"
    assert keyword_constant(refit_calls[0], "relative_tolerance") == 0.10

    assert len(calls_named(tree, "PdfPages")) == 2
    assert "observation_diagnostics" in call_names(tree)


def test_numbered_examples_keep_data_and_analysis_in_memory() -> None:
    prohibited_calls = {"mkdir", "read_csv", "to_csv"}
    prohibited_imports = {"pandas", "subprocess"}

    for path in _numbered_examples():
        tree = parse_module(path)
        assert call_names(tree).isdisjoint(prohibited_calls), path
        assert import_roots(tree).isdisjoint(prohibited_imports), path


def test_numbered_examples_keep_rendering_caller_owned() -> None:
    examples = _numbered_examples()
    assert examples

    for path in examples:
        tree = parse_module(path)
        imports = import_roots(tree)
        calls = call_names(tree)

        if path.name == "03_leave_one_out_validation.py":
            assert imports.isdisjoint(_RENDERING_PACKAGES)
            assert calls.isdisjoint(_RENDERING_METHODS)
        else:
            assert "matplotlib" in imports, path
            assert {"savefig", "subplots"} <= calls, path

        assert all(not name.startswith("plot_") for name in calls), path


def test_example_support_is_numerical_only() -> None:
    support = _repository_root() / "examples" / "_support"
    prohibited_calls = {
        *_RENDERING_METHODS,
        "genfromtxt",
        "load",
        "loadtxt",
        "open",
        "read_csv",
        "read_table",
        "to_csv",
    }

    for path in support.glob("*.py"):
        tree = parse_module(path)
        assert import_roots(tree).isdisjoint(_RENDERING_PACKAGES), path
        assert call_names(tree).isdisjoint(prohibited_calls), path
