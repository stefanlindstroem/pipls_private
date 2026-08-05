from __future__ import annotations

from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dilation_notation_text() -> str:
    root = _repository_root()
    paths = [
        root / "CHANGELOG.md",
        *(root / ".llm").glob("*.md"),
        *(root / "docs").rglob("*.md"),
        root / "src" / "pipls" / "decomposition.py",
        root / "src" / "pipls" / "inspection.py",
        root / "examples" / "05_pulp_real_data.py",
        root / "examples" / "06_sugarcane_real_data.py",
        root / "examples" / "07_tobacco_real_data.py",
        root / "tools" / "render_pulp_tutorial.py",
    ]
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_dilation_notation_matches_the_companion_manuscript() -> None:
    text = _dilation_notation_text()

    assert "d_k" not in text
    assert r"\operatorname{diag}(d_1" not in text
    assert "$D_k=D_{kk}$" in text
    assert r"$d_{\mathrm{p}}$" in text
