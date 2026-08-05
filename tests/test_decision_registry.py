from __future__ import annotations

import re
from pathlib import Path

_DECISION_ROW = re.compile(r"^\| `(?P<filename>\d{4}-[a-z0-9-]+\.md)` \|", re.MULTILINE)
_RETIREMENT_ROW = re.compile(
    r"^\| `(?P<filename>\d{4}-[a-z0-9-]+\.md)` \| "
    r"(?P<replacement>.*?) \|",
    re.MULTILINE,
)
_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]\n]+\]\((?P<target>[^)]+)\)")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _decision_files() -> set[str]:
    decisions = _repository_root() / "docs" / "decisions"
    return {
        path.name
        for path in decisions.glob("[0-9][0-9][0-9][0-9]-*.md")
    }


def _heading_anchors(text: str) -> set[str]:
    return {
        re.sub(r"[^a-z0-9 -]", "", heading.lower()).replace(" ", "-")
        for heading in re.findall(r"^#{1,6} (.+)$", text, re.MULTILINE)
    }


def test_current_decision_indexes_match_shipped_records() -> None:
    root = _repository_root()
    decisions = root / "docs" / "decisions"
    shipped = _decision_files()

    public_index = (decisions / "index.md").read_text(encoding="utf-8")
    public_entries = re.findall(r"\((\d{4}-[a-z0-9-]+\.md)\)", public_index)
    maintainer_index = (root / ".llm" / "decisions.md").read_text(encoding="utf-8")
    maintainer_entries = _DECISION_ROW.findall(maintainer_index)

    assert shipped
    assert len(public_entries) == len(set(public_entries))
    assert len(maintainer_entries) == len(set(maintainer_entries))
    assert set(public_entries) == shipped
    assert set(maintainer_entries) == shipped
    assert "[Compact development history](history.md)" in public_index
    assert "[Explicit retirement map](retirements.md)" in public_index

    for filename in shipped:
        text = (decisions / filename).read_text(encoding="utf-8")
        assert text.startswith("# Decision"), filename
        assert "\x00" not in text, filename


def test_retirement_map_is_complete_and_nonconflicting() -> None:
    decisions = _repository_root() / "docs" / "decisions"
    retirement_text = (decisions / "retirements.md").read_text(encoding="utf-8")
    rows = list(_RETIREMENT_ROW.finditer(retirement_text))
    retired_files = [match.group("filename") for match in rows]
    shipped_files = _decision_files()

    assert rows
    assert len(retired_files) == len(set(retired_files))
    assert set(retired_files).isdisjoint(shipped_files)
    assert {filename[:4] for filename in retired_files}.isdisjoint(
        {filename[:4] for filename in shipped_files}
    )

    history_text = (decisions / "history.md").read_text(encoding="utf-8")
    history_anchors = _heading_anchors(history_text)

    for match in rows:
        replacements = re.findall(r"\]\(([^)]+)\)", match.group("replacement"))
        assert replacements, match.group("filename")
        for replacement in replacements:
            target, _, anchor = replacement.partition("#")
            assert (decisions / target).is_file(), replacement
            if anchor:
                assert target == "history.md", replacement
                assert anchor in history_anchors, replacement


def test_active_maintainer_records_do_not_reference_retired_numbers() -> None:
    root = _repository_root()
    decisions = root / "docs" / "decisions"
    retirement_text = (decisions / "retirements.md").read_text(encoding="utf-8")
    retired_numbers = {
        match.group("filename")[:4]
        for match in _RETIREMENT_ROW.finditer(retirement_text)
    }
    retired_number_pattern = re.compile(
        rf"\b(?:{'|'.join(sorted(retired_numbers))})\b"
    )

    active_paths = sorted(decisions.glob("[0-9][0-9][0-9][0-9]-*.md"))
    active_paths.extend(sorted((root / ".llm").glob("*.md")))
    for path in active_paths:
        assert retired_number_pattern.search(path.read_text(encoding="utf-8")) is None, path


def test_all_local_decision_links_resolve() -> None:
    decisions = _repository_root() / "docs" / "decisions"
    documents = sorted(decisions.glob("*.md"))
    anchors = {
        path.name: _heading_anchors(path.read_text(encoding="utf-8"))
        for path in documents
    }

    for path in documents:
        text = path.read_text(encoding="utf-8")
        for match in _MARKDOWN_LINK.finditer(text):
            target = match.group("target").strip()
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            filename, _, anchor = target.partition("#")
            linked = path.parent / filename
            assert linked.is_file(), (path, target)
            if anchor:
                assert anchor in anchors[linked.name], (path, target)
