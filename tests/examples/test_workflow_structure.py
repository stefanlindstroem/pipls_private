from __future__ import annotations

import ast
from pathlib import Path

import pytest

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
_INSPECTION_CALLS = {
    "latent_structure",
    "pipls_display_factors",
    "prediction_diagnostics",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _numbered_examples() -> list[Path]:
    return sorted((_repository_root() / "examples").glob("[0-9][0-9]_*.py"))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _attribute_path(node: ast.expr) -> str | None:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _call_names(tree: ast.AST) -> set[str]:
    return {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (name := _call_name(node)) is not None
    }


def _attribute_paths(tree: ast.AST) -> set[str]:
    return {
        path
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and (path := _attribute_path(node)) is not None
    }


def _import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names


def _string_literals(node: ast.AST) -> set[str]:
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _assigned_call_path(tree: ast.AST, target_name: str) -> str | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != target_name:
            continue
        if isinstance(node.value, ast.Call):
            return _attribute_path(node.value.func)
    return None


def _assigned_call_argument_path(
    tree: ast.AST,
    target_name: str,
    argument_index: int,
) -> str | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != target_name:
            continue
        if not isinstance(node.value, ast.Call):
            return None
        if argument_index >= len(node.value.args):
            return None
        return _attribute_path(node.value.args[argument_index])
    return None


def _assigned_value_path(tree: ast.AST, target_name: str) -> str | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != target_name:
            continue
        if isinstance(node.value, ast.Call):
            return _attribute_path(node.value.func)
        if isinstance(node.value, ast.Attribute):
            return _attribute_path(node.value)
    return None


def _calls_with_name(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == name
    ]


def _keyword_string(call: ast.Call, keyword_name: str) -> str | None:
    for keyword in call.keywords:
        if keyword.arg != keyword_name:
            continue
        if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return keyword.value.value
    return None


def _keyword_constant(call: ast.Call, keyword_name: str) -> object | None:
    for keyword in call.keywords:
        if keyword.arg != keyword_name:
            continue
        if isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def _uses_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def _top_level_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _assigned_name_lineno(scope: ast.AST, target_name: str) -> int:
    lines = [
        node.lineno
        for node in ast.walk(scope)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in node.targets
        )
    ]
    assert lines, target_name
    return min(lines)


def _first_call_lineno(scope: ast.AST, name: str) -> int:
    lines = [
        node.lineno
        for node in ast.walk(scope)
        if isinstance(node, ast.Call) and _call_name(node) == name
    ]
    assert lines, name
    return min(lines)


@pytest.mark.parametrize(
    ("relative_path", "required_attribute"),
    [
        ("examples/02_synthetic_path_selection.py", "search.component_path_"),
        ("examples/03_leave_one_out_validation.py", "search.validation_report"),
        ("examples/04_pls_path_comparison.py", "search.component_path_"),
        ("examples/05_pulp_real_data.py", "search.component_path_"),
        ("examples/06_sugarcane_real_data.py", "search.component_path_"),
        ("examples/07_tobacco_real_data.py", "search.component_path_"),
        ("tools/render_synthetic_tutorial.py", "search.component_path_"),
        ("tools/render_pulp_tutorial.py", "search.component_path_"),
    ],
)
def test_retained_search_objects_use_search_name(
    relative_path: str,
    required_attribute: str,
) -> None:
    tree = _tree(_repository_root() / relative_path)
    assert required_attribute in _attribute_paths(tree)


@pytest.mark.parametrize(
    "relative_path",
    [
        "examples/02_synthetic_path_selection.py",
        "examples/04_pls_path_comparison.py",
        "examples/05_pulp_real_data.py",
        "examples/06_sugarcane_real_data.py",
        "examples/07_tobacco_real_data.py",
        "tools/render_synthetic_tutorial.py",
        "tools/render_pulp_tutorial.py",
    ],
)
def test_kfold_examples_use_seeded_shuffled_folds(relative_path: str) -> None:
    tree = _tree(_repository_root() / relative_path)
    kfold_calls = _calls_with_name(tree, "KFold")
    assert kfold_calls
    for call in kfold_calls:
        assert _keyword_constant(call, "n_splits") == 5
        assert _keyword_constant(call, "shuffle") is True
        assert _keyword_constant(call, "random_state") == 0


def test_leave_one_out_example_uses_its_exhaustive_splitter() -> None:
    tree = _tree(_repository_root() / "examples" / "03_leave_one_out_validation.py")
    assert "LeaveOneOut" in _call_names(tree)
    assert "KFold" not in _call_names(tree)


def test_ordinary_pls_is_confined_to_the_comparison_helper() -> None:
    examples = _repository_root() / "examples"
    users = {
        path.relative_to(examples).as_posix()
        for path in examples.rglob("*.py")
        if "PLSRegression" in _imported_names(_tree(path))
        or "PLSRegression" in _call_names(_tree(path))
    }
    assert users == {"_support/pls_component_path.py"}

    comparison_tree = _tree(examples / "04_pls_path_comparison.py")
    comparison_calls = _call_names(comparison_tree)
    comparison_attributes = _attribute_paths(comparison_tree)
    assert "evaluate_pls_component_path" in comparison_calls
    assert "search.component_path_" in comparison_attributes
    assert {"pulp", "sugarcane", "tobacco"} <= _string_literals(comparison_tree)

    for filename in (
        "05_pulp_real_data.py",
        "06_sugarcane_real_data.py",
        "07_tobacco_real_data.py",
    ):
        assert "evaluate_pls_component_path" not in _call_names(_tree(examples / filename))


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
def test_maintained_reference_consumers_use_package_loaders(
    loader_name: str,
    relative_paths: tuple[str, ...],
) -> None:
    repository = _repository_root()
    for relative_path in relative_paths:
        path = repository / relative_path
        tree = _tree(path)
        assert loader_name in _imported_names(tree), path
        assert loader_name in _call_names(tree), path


@pytest.mark.parametrize(
    (
        "filename",
        "extra_calls",
        "required_attributes",
        "expected_pdfs",
        "loader_name",
        "response_assignment",
        "response_source",
        "coordinate_assignment",
    ),
    [
        (
            "05_pulp_real_data.py",
            {"predictor_rank_profile"},
            {
                "model.selection_",
                "search.component_path_",
                "factors.predictor_directions",
                "structure.x_scores",
                "diagnostics.observed_standardized",
            },
            {
                "component_path.pdf",
                "predictor_rank_profile.pdf",
                "pipls_factors.pdf",
                "prediction_diagnostics.pdf",
                "latent_structure.pdf",
                "coefficients.pdf",
            },
            "load_pulp",
            "data.target_names",
            None,
            None,
        ),
        (
            "06_sugarcane_real_data.py",
            {"predictor_rank_profile"},
            {
                "model.selection_",
                "search.component_path_",
                "rank_profile.selected_result",
                "factors.predictor_directions",
                "structure.x_scores",
                "diagnostics.observed_standardized",
            },
            {
                "component_path.pdf",
                "predictor_rank_profile.pdf",
                "pipls_factors.pdf",
                "prediction_diagnostics.pdf",
                "latent_structure.pdf",
                "coefficients.pdf",
            },
            "load_sugarcane",
            "list",
            "data.target_names",
            ("wavelengths", "np.asarray", "data.feature_names"),
        ),
        (
            "07_tobacco_real_data.py",
            {"observation_diagnostics", "predictor_rank_profile"},
            {
                "search.component_path_",
                "rank_profile.selected_result",
                "factors.predictor_directions",
                "structure.x_scores",
                "diagnostics.observed_standardized",
                "observations.score_distance",
                "observations.x_reconstruction_residual",
            },
            {
                "component_path.pdf",
                "predictor_rank_profile.pdf",
                "pipls_factors.pdf",
                "prediction_diagnostics.pdf",
                "latent_structure.pdf",
                "coefficients.pdf",
            },
            "load_tobacco",
            "list",
            "data.target_names",
            ("wavenumbers", "np.asarray", "data.feature_names"),
        ),
    ],
)
def test_real_data_examples_use_direct_public_results(
    filename: str,
    extra_calls: set[str],
    required_attributes: set[str],
    expected_pdfs: set[str],
    loader_name: str,
    response_assignment: str,
    response_source: str | None,
    coordinate_assignment: tuple[str, str, str] | None,
) -> None:
    path = _repository_root() / "examples" / filename
    tree = _tree(path)
    calls = _call_names(tree)
    attributes = _attribute_paths(tree)

    required_calls = {
        "PiPLSSearchCV",
        "refit",
        "savefig",
        "subplots",
        *_INSPECTION_CALLS,
        *extra_calls,
    }
    if filename in {"05_pulp_real_data.py", "06_sugarcane_real_data.py"}:
        required_calls.add("oof_report")
        assert {"select", "validation_report"}.isdisjoint(calls)
    else:
        required_calls.update({"select", "validation_report"})
    assert required_calls <= calls
    assert required_attributes <= attributes
    assert loader_name in calls
    assert _assigned_value_path(tree, "response_names") == response_assignment
    if response_source is not None:
        assert _assigned_call_argument_path(tree, "response_names", 0) == response_source
    if coordinate_assignment is not None:
        variable, call_path, source_path = coordinate_assignment
        assert _assigned_call_path(tree, variable) == call_path
        assert _assigned_call_argument_path(tree, variable, 0) == source_path
    assert {
        value
        for value in _string_literals(tree)
        if value.endswith(".pdf")
    } == expected_pdfs


@pytest.mark.parametrize(
    ("filename", "loader_name", "extra_analysis_calls"),
    [
        (
            "06_sugarcane_real_data.py",
            "load_sugarcane",
            {"predictor_rank_profile"},
        ),
        (
            "07_tobacco_real_data.py",
            "load_tobacco",
            {"observation_diagnostics", "predictor_rank_profile"},
        ),
    ],
)
def test_complete_examples_separate_analysis_from_same_file_rendering(
    filename: str,
    loader_name: str,
    extra_analysis_calls: set[str],
) -> None:
    path = _repository_root() / "examples" / filename
    tree = _tree(path)
    functions = _top_level_functions(tree)
    main = functions["main"]

    analysis_calls = {
        "PiPLSSearchCV",
        loader_name,
        "refit",
        *_INSPECTION_CALLS,
        *extra_analysis_calls,
    }
    if filename == "06_sugarcane_real_data.py":
        analysis_calls.add("oof_report")
        assert {"select", "validation_report"}.isdisjoint(_call_names(main))
    else:
        analysis_calls.update({"select", "validation_report"})
    assert analysis_calls <= _call_names(main)
    assert _call_names(main).isdisjoint(_RENDERING_METHODS)

    rendering_functions = [
        function
        for name, function in functions.items()
        if name != "main"
        and {"savefig", "subplots"} <= _call_names(function)
    ]
    assert rendering_functions
    assert {function.name for function in rendering_functions} <= _call_names(main)

    for function in rendering_functions:
        calls = _call_names(function)
        assert function.name.startswith("_")
        assert calls.isdisjoint(analysis_calls)
        assert {"savefig", "subplots"} <= calls

    module_scope = ast.Module(
        body=[
            node
            for node in tree.body
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ],
        type_ignores=[],
    )
    module_calls = _call_names(module_scope)
    assert "main" in module_calls
    assert module_calls.isdisjoint({*analysis_calls, *_RENDERING_METHODS})



@pytest.mark.parametrize(
    (
        "relative_path",
        "scope_name",
        "path_name",
        "report_expected",
        "render_call",
    ),
    [
        (
            "examples/02_synthetic_path_selection.py",
            None,
            "path",
            False,
            "subplots",
        ),
        (
            "examples/05_pulp_real_data.py",
            None,
            "path",
            True,
            "subplots",
        ),
        (
            "examples/06_sugarcane_real_data.py",
            "main",
            "path",
            True,
            "_plot_component_path",
        ),
        (
            "tools/render_synthetic_tutorial.py",
            "render_synthetic_tutorial_assets",
            "path",
            False,
            "subplots",
        ),
        (
            "tools/render_pulp_tutorial.py",
            "render_pulp_tutorial_assets",
            "component_path",
            True,
            "_render_component_path",
        ),
    ],
)
def test_manual_workflows_complete_modeling_before_analysis_and_rendering(
    relative_path: str,
    scope_name: str | None,
    path_name: str,
    report_expected: bool,
    render_call: str,
) -> None:
    tree = _tree(_repository_root() / relative_path)
    scope: ast.AST = (
        tree if scope_name is None else _top_level_functions(tree)[scope_name]
    )

    search_line = _assigned_name_lineno(scope, "search")
    model_line = _assigned_name_lineno(scope, "model")
    selection_line = _assigned_name_lineno(scope, "selection")
    path_line = _assigned_name_lineno(scope, path_name)
    rank_profile_line = _assigned_name_lineno(scope, "rank_profile")
    diagnostics_line = _assigned_name_lineno(scope, "diagnostics")
    render_line = _first_call_lineno(scope, render_call)

    assert search_line < model_line < selection_line
    assert selection_line < path_line < rank_profile_line
    if report_expected:
        report_line = _assigned_name_lineno(scope, "report")
        assert rank_profile_line < report_line < diagnostics_line
    else:
        assert "oof_report" not in _call_names(scope)
        assert rank_profile_line < diagnostics_line
    assert diagnostics_line < render_line


def test_numbered_examples_keep_dataset_io_and_analysis_in_memory() -> None:
    for path in _numbered_examples():
        tree = _tree(path)
        calls = _call_names(tree)
        imports = _import_roots(tree)
        assert "mkdir" not in calls, path
        assert "read_csv" not in calls, path
        assert "to_csv" not in calls, path
        assert "pandas" not in imports, path
        assert "subprocess" not in imports, path



def test_manual_workflows_use_refitted_model_selection() -> None:
    repository = _repository_root()
    paths = (
        repository / "examples" / "02_synthetic_path_selection.py",
        repository / "examples" / "05_pulp_real_data.py",
        repository / "examples" / "06_sugarcane_real_data.py",
        repository / "tools" / "render_synthetic_tutorial.py",
        repository / "tools" / "render_pulp_tutorial.py",
    )
    for path in paths:
        tree = _tree(path)
        assert "model.selection_" in _attribute_paths(tree), path
        assert "select" not in _call_names(tree), path


@pytest.mark.parametrize(
    ("filename", "selection_name"),
    [
        ("06_sugarcane_real_data.py", "selection"),
        ("07_tobacco_real_data.py", "selected"),
    ],
)
def test_spectral_examples_extract_rank_profile_at_selected_component_count(
    filename: str,
    selection_name: str,
) -> None:
    tree = _tree(_repository_root() / "examples" / filename)

    assert _assigned_call_path(tree, "rank_profile") == "search.predictor_rank_profile"
    assert (
        _assigned_call_argument_path(tree, "rank_profile", 0)
        == f"{selection_name}.n_components"
    )


def test_tobacco_owns_full_svd_selection_and_paginated_reports() -> None:
    tree = _tree(_repository_root() / "examples" / "07_tobacco_real_data.py")
    calls = _call_names(tree)

    regression_calls = _calls_with_name(tree, "PiPLSRegression")
    assert len(regression_calls) == 1
    assert _keyword_string(regression_calls[0], "svd_solver") == "full"

    select_calls = _calls_with_name(tree, "select")
    assert len(select_calls) == 2
    assert {_keyword_string(call, "rule") for call in select_calls} == {
        "minimum_cv_mse",
        "one_standard_error",
    }

    refit_calls = _calls_with_name(tree, "refit")
    assert len(refit_calls) == 1
    assert _keyword_string(refit_calls[0], "rule") == "one_standard_error"

    report_calls = _calls_with_name(tree, "validation_report")
    assert len(report_calls) == 1
    assert _keyword_string(report_calls[0], "rule") == "one_standard_error"

    search_calls = _calls_with_name(tree, "PiPLSSearchCV")
    assert any(_keyword_string(call, "search_method") == "auto" for call in search_calls)
    assert _assigned_call_path(tree, "response_names") == "list"
    assert (
        _assigned_call_argument_path(tree, "response_names", 0)
        == "data.target_names"
    )
    assert "sorted" not in calls
    assert "observation_diagnostics" in calls

    functions = _top_level_functions(tree)
    report_function_names: set[str] = set()
    for name, function in functions.items():
        if name == "main":
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.With):
                continue
            if not any(
                isinstance(item.context_expr, ast.Call)
                and _call_name(item.context_expr) == "PdfPages"
                for item in node.items
            ):
                continue
            assert any(
                isinstance(child, ast.For) and _uses_name(child.iter, "response_pages")
                for child in node.body
            )
            report_function_names.add(name)

    assert len(report_function_names) == 2
    paginated_reports = {
        value
        for node in ast.walk(functions["main"])
        if isinstance(node, ast.Call) and _call_name(node) in report_function_names
        for value in _string_literals(node)
        if value.endswith(".pdf")
    }
    assert paginated_reports == {"prediction_diagnostics.pdf", "coefficients.pdf"}



def test_numbered_examples_keep_rendering_caller_owned() -> None:
    examples = _numbered_examples()
    assert examples

    for path in examples:
        tree = _tree(path)
        imports = _import_roots(tree)
        calls = _call_names(tree)

        if path.name == "03_leave_one_out_validation.py":
            assert imports.isdisjoint(_RENDERING_PACKAGES)
            assert calls.isdisjoint(_RENDERING_METHODS)
            continue

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
        tree = _tree(path)
        assert _import_roots(tree).isdisjoint(_RENDERING_PACKAGES), path
        assert _call_names(tree).isdisjoint(prohibited_calls), path
