"""Build the documentation from a clean source-distribution installation."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from _artifact_support import (
    clean_subprocess_environment,
    run,
    safe_extract_sdist,
    single_artifact,
    venv_python,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_generated_manifests(generated_root: Path) -> None:
    manifests = sorted(generated_root.rglob("manifest.json"))
    if not manifests:
        raise RuntimeError("Documentation build did not generate figure manifests.")

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        figures = manifest.get("figures")
        if not isinstance(figures, list) or not figures:
            raise RuntimeError(f"Invalid figure manifest: {manifest_path}")

        filenames = [record.get("filename") for record in figures]
        if len(filenames) != len(set(filenames)):
            raise RuntimeError(f"Duplicate figure filenames in {manifest_path}")

        for record in figures:
            filename = record.get("filename")
            expected_hash = record.get("sha256")
            if not isinstance(filename, str) or not isinstance(expected_hash, str):
                raise RuntimeError(f"Invalid figure record in {manifest_path}")
            figure = manifest_path.parent / filename
            if not figure.is_file():
                raise RuntimeError(f"Missing manifest-declared figure: {figure}")
            ET.parse(figure)
            if _sha256(figure) != expected_hash:
                raise RuntimeError(f"Figure hash disagrees with manifest: {figure}")


def _validate_site(source: Path) -> None:
    if not (source / "site" / "index.html").is_file():
        raise RuntimeError("Documentation build did not create site/index.html.")
    _validate_generated_manifests(source / "docs" / "assets" / "generated")


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    make = shutil.which("make")
    if make is None:
        raise RuntimeError("The documentation distribution check requires make.")
    environment_variables = clean_subprocess_environment()

    with tempfile.TemporaryDirectory(prefix="pipls-docs-dist-") as temporary:
        workspace = Path(temporary)
        environment_variables["PIP_CACHE_DIR"] = str(workspace / "pip-cache")
        artifacts = workspace / "artifacts"
        artifacts.mkdir()

        run(
            [
                sys.executable,
                "-m",
                "build",
                "--quiet",
                "--sdist",
                "--outdir",
                str(artifacts),
            ],
            cwd=repository,
            env=environment_variables,
        )
        source_distribution = single_artifact(
            artifacts,
            "*.tar.gz",
            "source distribution",
        )
        source = safe_extract_sdist(source_distribution, workspace / "extracted")

        environment = workspace / "venv"
        run(
            [sys.executable, "-m", "venv", str(environment)],
            env=environment_variables,
        )
        python = venv_python(environment)
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--quiet",
                "--no-input",
                "--retries",
                "5",
                "--timeout",
                "60",
                f"{source}[docs]",
            ],
            env=environment_variables,
        )
        run(
            [make, "docs", f"PYTHON={python}"],
            cwd=source,
            env=environment_variables,
        )
        _validate_site(source)

        pages_config = source / ".mkdocs-pages.yml"
        run(
            [
                str(python),
                str(source / "tools" / "configure_pages_docs.py"),
                "--repository",
                "example/pipls",
                "--server-url",
                "https://github.com",
                "--output",
                str(pages_config),
            ],
            cwd=source,
            env=environment_variables,
        )
        run(
            [
                str(python),
                "-m",
                "mkdocs",
                "build",
                "--strict",
                "--config-file",
                str(pages_config),
            ],
            cwd=source,
            env=environment_variables,
        )
        _validate_site(source)

    print("Source-distribution documentation build passed.")


if __name__ == "__main__":
    main()
