from __future__ import annotations

import re
from pathlib import Path

import pipls

_DIRECTIVE = re.compile(r"^::: pipls\.([A-Za-z_][A-Za-z0-9_]*)$", re.MULTILINE)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_core_api_pages_cover_the_top_level_public_objects() -> None:
    api_directory = _repository_root() / "docs" / "api"
    documented = {
        name
        for path in api_directory.glob("*.md")
        for name in _DIRECTIVE.findall(path.read_text(encoding="utf-8"))
    }
    expected = set(pipls.__all__) - {"__version__"}

    assert documented == expected


def test_core_api_pages_reference_only_public_import_paths() -> None:
    api_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((_repository_root() / "docs" / "api").glob("*.md"))
    )

    assert "::: pipls._" not in api_text
    assert "show_source: true" not in api_text
