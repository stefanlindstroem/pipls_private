from __future__ import annotations

import importlib.util
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
    assert version != "0.0.0"
    assert pipls.__version__ == version
    assert citation["version"] == version


def test_release_urls_are_consistent_across_public_metadata() -> None:
    project = _project_metadata()
    citation = _citation_metadata()
    project_urls = project["urls"]
    assert isinstance(project_urls, dict)

    assert project_urls["Source"] == EXPECTED_REPOSITORY
    assert project_urls["Issues"] == f"{EXPECTED_REPOSITORY}/issues"
    assert project_urls["Documentation"] == EXPECTED_DOCUMENTATION
    assert citation["repository-code"] == EXPECTED_REPOSITORY
    assert citation["url"] == EXPECTED_DOCUMENTATION


def test_annotation_layout_extra_is_textalloc_only() -> None:
    project = _project_metadata()
    optional = project["optional-dependencies"]
    assert isinstance(optional, dict)

    for group in ("dev", "examples", "docs"):
        requirements = [str(requirement) for requirement in optional[group]]
        assert any(requirement.startswith("textalloc>=") for requirement in requirements)
        assert all(not requirement.lower().startswith("adjusttext") for requirement in requirements)
