from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import yaml

import pipls

if importlib.util.find_spec("tomllib") is not None:
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 in CI
    import tomli as tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REPOSITORY = "https://github.com/stefanlindstroem/pipls"
EXPECTED_DOCUMENTATION = "https://stefanlindstroem.github.io/pipls/"


def _project_metadata() -> dict[str, object]:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def _citation_metadata() -> dict[str, object]:
    return yaml.safe_load((REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8"))


def test_release_version_is_consistent_across_public_metadata() -> None:
    project = _project_metadata()
    citation = _citation_metadata()
    version = str(project["version"])
    citation_guide = (REPOSITORY_ROOT / "docs" / "citation.md").read_text(encoding="utf-8")
    changelog = (REPOSITORY_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert version != "0.0.0"
    assert pipls.__version__ == version
    assert citation["version"] == version
    assert f"version {version}." in citation_guide

    release_heading = re.search(
        rf"^## {re.escape(version)} - (?P<date>\d{{4}}-\d{{2}}-\d{{2}})$",
        changelog,
        flags=re.MULTILINE,
    )
    assert release_heading is not None
    assert str(citation["date-released"]) == release_heading.group("date")


def test_release_urls_are_consistent_across_public_metadata() -> None:
    project = _project_metadata()
    citation = _citation_metadata()
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    project_urls = project["urls"]
    assert isinstance(project_urls, dict)

    assert project_urls["Source"] == EXPECTED_REPOSITORY
    assert project_urls["Issues"] == f"{EXPECTED_REPOSITORY}/issues"
    assert project_urls["Documentation"] == EXPECTED_DOCUMENTATION
    assert citation["repository-code"] == EXPECTED_REPOSITORY
    assert citation["url"] == EXPECTED_DOCUMENTATION
    assert EXPECTED_DOCUMENTATION in readme
