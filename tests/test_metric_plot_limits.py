from __future__ import annotations

import ast
from pathlib import Path

_METRIC_ATTRIBUTES = {"standardized_rmse"}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _plot_sources(root: Path) -> tuple[Path, ...]:
    examples = tuple(sorted((root / "examples").glob("*.py")))
    renderers = tuple(sorted((root / "tools").glob("render_*tutorial.py")))
    return examples + renderers


def _contains_metric(node: ast.AST) -> bool:
    return any(
        isinstance(candidate, ast.Attribute)
        and candidate.attr in _METRIC_ATTRIBUTES
        for candidate in ast.walk(node)
    )


def _receiver(call: ast.Call, method: str) -> str | None:
    function = call.func
    if not isinstance(function, ast.Attribute) or function.attr != method:
        return None
    return ast.dump(function.value, include_attributes=False)


def _is_constant(node: ast.AST, value: float) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


def _has_unit_ylim(function: ast.FunctionDef, receiver: str) -> bool:
    for candidate in ast.walk(function):
        if not isinstance(candidate, ast.Call):
            continue
        if _receiver(candidate, "set_ylim") != receiver:
            continue
        if len(candidate.args) < 2:
            continue
        if _is_constant(candidate.args[0], 0.0) and _is_constant(
            candidate.args[1], 1.0
        ):
            return True
    return False


def test_standardized_rmse_bar_plots_use_unit_interval() -> None:
    root = _repository_root()
    violations: list[str] = []
    metric_bars = 0

    for path in _plot_sources(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function in (
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ):
            for candidate in ast.walk(function):
                if not isinstance(candidate, ast.Call) or len(candidate.args) < 2:
                    continue
                receiver = _receiver(candidate, "bar")
                if receiver is None or not _contains_metric(candidate.args[1]):
                    continue
                metric_bars += 1
                if not _has_unit_ylim(function, receiver):
                    violations.append(
                        f"{path.relative_to(root)}:{candidate.lineno} ({function.name})"
                    )

    assert metric_bars == 2
    assert violations == []
