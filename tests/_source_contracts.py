from __future__ import annotations

import ast
from pathlib import Path


def parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def dotted_name(node: ast.expr) -> str | None:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def call_name(node: ast.Call) -> str | None:
    name = dotted_name(node.func)
    return None if name is None else name.rsplit(".", 1)[-1]


def call_names(scope: ast.AST) -> set[str]:
    return {
        name
        for node in ast.walk(scope)
        if isinstance(node, ast.Call) and (name := call_name(node)) is not None
    }


def calls_named(scope: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Call) and call_name(node) == name
    ]


def call_lines(scope: ast.AST, names: set[str]) -> list[int]:
    return sorted(
        node.lineno
        for node in ast.walk(scope)
        if isinstance(node, ast.Call) and call_name(node) in names
    )


def attribute_paths(scope: ast.AST) -> set[str]:
    return {
        path
        for node in ast.walk(scope)
        if isinstance(node, ast.Attribute) and (path := dotted_name(node)) is not None
    }


def import_roots(scope: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".", 1)[0])
    return roots


def imported_names(scope: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names


def private_pipls_imports(scope: ast.AST) -> set[str]:
    private: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Import):
            private.update(alias.name for alias in node.names if alias.name.startswith("pipls._"))
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.startswith("pipls._"):
                private.add(node.module)
            if node.module == "pipls" or node.module.startswith("pipls."):
                private.update(alias.name for alias in node.names if alias.name.startswith("_"))
    return private


def keyword_constant(call: ast.Call, keyword_name: str) -> object | None:
    for keyword in call.keywords:
        if keyword.arg == keyword_name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def top_level_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def module_scope(tree: ast.Module) -> ast.Module:
    return ast.Module(
        body=[
            node
            for node in tree.body
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ],
        type_ignores=[],
    )
