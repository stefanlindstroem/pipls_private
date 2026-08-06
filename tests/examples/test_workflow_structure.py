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
_SEARCH_EVIDENCE_CALLS = {
    "oof_report",
    "predictor_rank_profile",
}
_SELECTION_EVIDENCE_ANALYSIS_CALLS = {"prediction_diagnostics"}
_FITTED_MODEL_ANALYSIS_CALLS = {
    "latent_structure",
    "observation_diagnostics",
    "pipls_display_factors",
}
_ANALYSIS_CALLS = (
    _SEARCH_EVIDENCE_CALLS
    | _SELECTION_EVIDENCE_ANALYSIS_CALLS
    | _FITTED_MODEL_ANALYSIS_CALLS
)
_SYNTHETIC_SELECTION_WORKFLOWS = (
    "examples/02_synthetic_path_selection.py",
    "tools/render_synthetic_tutorial.py",
)
_SELECTION_DRIVEN_REAL_DATA_WORKFLOWS = (
    "examples/05_pulp_real_data.py",
    "examples/06_sugarcane_real_data.py",
    "examples/07_tobacco_real_data.py",
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


@pytest.mark.parametrize("relative_path", _SYNTHETIC_SELECTION_WORKFLOWS)
def test_synthetic_selection_review_precedes_refit(
    relative_path: str,
) -> None:
    tree = parse_module(_repository_root() / relative_path)
    scope = _workflow_scope(tree)

    select_lines = call_lines(scope, {"select"})
    profile_lines = call_lines(scope, {"predictor_rank_profile"})
    refit_lines = call_lines(scope, {"refit"})
    prediction_lines = call_lines(scope, {"prediction_diagnostics"})
    rendering_names = _RENDERING_METHODS | _rendering_function_names(tree)
    rendering_lines = call_lines(scope, rendering_names)

    assert len(select_lines) == 1
    assert len(profile_lines) == 1
    assert len(refit_lines) == 1
    assert len(prediction_lines) == 1
    assert rendering_lines
    assert select_lines[0] < profile_lines[0] < refit_lines[0]
    assert refit_lines[0] < prediction_lines[0]

    if relative_path.startswith("tools/"):
        assert min(rendering_lines) < select_lines[0]
        assert any(profile_lines[0] < line < refit_lines[0] for line in rendering_lines)
        assert prediction_lines[0] < max(rendering_lines)
    else:
        assert prediction_lines[0] < min(rendering_lines)

    select_call = calls_named(scope, "select")[0]
    component_keyword = next(
        keyword
        for keyword in select_call.keywords
        if keyword.arg == "n_components"
    )
    assert isinstance(component_keyword.value, ast.Name)
    assert component_keyword.value.id in {"CHOSEN_N_COMPONENTS", "chosen_n_components"}

    refit_call = calls_named(scope, "refit")[0]
    assert _keyword_path(refit_call, "selection") == "selection"
    assert _keyword_path(refit_call, "rule") is None
    assert _keyword_path(refit_call, "n_components") is None


@pytest.mark.parametrize("relative_path", _SELECTION_DRIVEN_REAL_DATA_WORKFLOWS)
def test_real_data_selection_review_precedes_refit(
    relative_path: str,
) -> None:
    path = _repository_root() / relative_path
    tree = parse_module(path)
    scope = _workflow_scope(tree)

    select_lines = call_lines(scope, {"select"})
    profile_lines = call_lines(scope, {"predictor_rank_profile"})
    report_lines = call_lines(scope, {"oof_report"})
    diagnostic_lines = call_lines(scope, _SELECTION_EVIDENCE_ANALYSIS_CALLS)
    refit_lines = call_lines(scope, {"refit"})
    fitted_analysis_lines = call_lines(scope, _FITTED_MODEL_ANALYSIS_CALLS)
    rendering_names = _RENDERING_METHODS | _rendering_function_names(tree)
    rendering_lines = call_lines(scope, rendering_names)

    assert len(select_lines) == 1
    assert len(profile_lines) == 1
    assert len(report_lines) == 1
    assert len(diagnostic_lines) == 1
    assert len(refit_lines) == 1
    assert fitted_analysis_lines
    assert rendering_lines
    assert (
        select_lines[0]
        < profile_lines[0]
        < report_lines[0]
        < diagnostic_lines[0]
        < refit_lines[0]
        < min(fitted_analysis_lines)
    )

    if relative_path.startswith("tools/"):
        assert min(rendering_lines) < select_lines[0]
        assert any(profile_lines[0] < line < refit_lines[0] for line in rendering_lines)
        assert max(fitted_analysis_lines) < max(rendering_lines)
    else:
        assert max(fitted_analysis_lines) < min(rendering_lines)

    report_call = calls_named(scope, "oof_report")[0]
    refit_call = calls_named(scope, "refit")[0]
    assert _keyword_path(report_call, "selection") == "selection"
    assert _keyword_path(refit_call, "selection") == "selection"
    assert _keyword_path(refit_call, "rule") is None
    assert _keyword_path(refit_call, "n_components") is None
    assert all(
        not (isinstance(node, ast.Attribute) and node.attr == "selection_")
        for node in ast.walk(scope)
    ), path


@pytest.mark.parametrize(
    ("relative_path", "first_stage", "selection_stage", "review_stage"),
    [
        (
            "examples/02_synthetic_path_selection.py",
            "inspect-synthetic-component-path",
            "choose-synthetic-selection",
            "inspect-synthetic-selected-evidence",
        ),
        (
            "examples/05_pulp_real_data.py",
            "inspect-pulp-component-path",
            "choose-pulp-selection",
            "inspect-pulp-selected-evidence",
        ),
    ],
)
def test_manual_tutorial_sources_separate_path_selection_and_review(
    relative_path: str,
    first_stage: str,
    selection_stage: str,
    review_stage: str,
) -> None:
    text = (_repository_root() / relative_path).read_text(encoding="utf-8")

    first = text.index(f"# --8<-- [start:{first_stage}]")
    selection = text.index(f"# --8<-- [start:{selection_stage}]")
    selection_end = text.index(f"# --8<-- [end:{selection_stage}]")
    review = text.index(f"# --8<-- [start:{review_stage}]")
    chosen = text.index("CHOSEN_N_COMPONENTS")
    select_call = text.index("selection = search.select")

    assert first < selection < review
    assert selection < chosen < selection_end
    assert selection < select_call < selection_end


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
def test_real_data_examples_use_public_selection_and_inspection(
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
    assert {"select", "refit"} <= calls


@pytest.mark.parametrize(
    "relative_path",
    [
        "examples/05_pulp_real_data.py",
        "examples/06_sugarcane_real_data.py",
    ],
)
def test_manual_real_data_workflows_select_the_declared_component_count(
    relative_path: str,
) -> None:
    tree = parse_module(_repository_root() / relative_path)
    select_calls = calls_named(tree, "select")

    assert len(select_calls) == 1
    component_keyword = next(
        keyword
        for keyword in select_calls[0].keywords
        if keyword.arg == "n_components"
    )
    assert isinstance(component_keyword.value, ast.Name)
    assert component_keyword.value.id == "CHOSEN_N_COMPONENTS"


@pytest.mark.parametrize(
    "relative_path",
    [
        "examples/03_leave_one_out_validation.py",
        "examples/04_pls_path_comparison.py",
    ],
)
def test_validation_and_comparison_routes_do_not_refit_a_final_model(
    relative_path: str,
) -> None:
    calls = call_names(parse_module(_repository_root() / relative_path))
    assert "refit" not in calls


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


def test_tobacco_keeps_two_explicit_tolerance_decisions_and_paginated_reports() -> None:
    tree = parse_module(_repository_root() / "examples" / "07_tobacco_real_data.py")

    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert constants["PREDICTOR_RANK_RELATIVE_TOLERANCE"] == 0.10
    assert constants["COMPONENT_RELATIVE_TOLERANCE"] == 0.10

    regression_calls = calls_named(tree, "PiPLSRegression")
    assert len(regression_calls) == 1
    assert keyword_constant(regression_calls[0], "svd_solver") == "full"

    search_calls = calls_named(tree, "PiPLSSearchCV")
    assert len(search_calls) == 1
    predictor_keyword = next(
        keyword
        for keyword in search_calls[0].keywords
        if keyword.arg == "predictor_rank_relative_tolerance"
    )
    assert isinstance(predictor_keyword.value, ast.Name)
    assert predictor_keyword.value.id == "PREDICTOR_RANK_RELATIVE_TOLERANCE"

    select_calls = calls_named(tree, "select")
    assert len(select_calls) == 1
    assert keyword_constant(select_calls[0], "rule") == "minimum_cv_mse"
    component_keyword = next(
        keyword
        for keyword in select_calls[0].keywords
        if keyword.arg == "relative_tolerance"
    )
    assert isinstance(component_keyword.value, ast.Name)
    assert component_keyword.value.id == "COMPONENT_RELATIVE_TOLERANCE"

    refit_calls = calls_named(tree, "refit")
    assert len(refit_calls) == 1
    assert _keyword_path(refit_calls[0], "selection") == "selection"
    assert _keyword_path(refit_calls[0], "rule") is None
    assert all(
        keyword.arg != "relative_tolerance" for keyword in refit_calls[0].keywords
    )

    text_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    rendered_text = "\n".join(text_literals)
    assert "predictor-rank threshold" in rendered_text
    assert "component-count threshold" in rendered_text
    assert "exact optimum r_pi=" in rendered_text
    assert "exact minimum h=" in rendered_text

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
