"""Build the documentation from a clean installation of the source distribution."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> Path:
    members = archive.getmembers()
    if not members:
        raise RuntimeError("The source-distribution archive is empty.")

    roots = {Path(member.name).parts[0] for member in members if Path(member.name).parts}
    if len(roots) != 1:
        raise RuntimeError("The source distribution must contain one top-level directory.")

    destination_resolved = destination.resolve()
    for member in members:
        member_path = Path(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise RuntimeError(f"Unsafe archive member: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Archive links are not supported: {member.name}")
        target = (destination / member_path).resolve()
        if not target.is_relative_to(destination_resolved):
            raise RuntimeError(f"Archive member escapes extraction directory: {member.name}")

    if sys.version_info >= (3, 12):
        archive.extractall(destination, filter="data")
    else:
        archive.extractall(destination)
    return destination / roots.pop()


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    make = shutil.which("make")
    if make is None:
        raise RuntimeError("The documentation distribution check requires make.")

    with tempfile.TemporaryDirectory(prefix="pipls-docs-dist-") as temporary:
        workspace = Path(temporary)
        artifacts = workspace / "artifacts"
        extracted = workspace / "extracted"
        environment = workspace / "venv"
        artifacts.mkdir()
        extracted.mkdir()

        _run(
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
        )
        archives = sorted(artifacts.glob("*.tar.gz"))
        if len(archives) != 1:
            raise RuntimeError(f"Expected one source distribution, found {len(archives)}.")

        with tarfile.open(archives[0], mode="r:gz") as archive:
            source = _safe_extract(archive, extracted)

        required = [
            source / "Makefile",
            source / "mkdocs.yml",
            source / "docs" / "index.md",
            source / "docs" / "api" / "index.md",
            source / "docs" / "javascripts" / "mathjax.js",
            source / "src" / "pipls" / "__init__.py",
        ]
        missing = [path.relative_to(source).as_posix() for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Source distribution is missing documentation inputs: {missing}")

        _run([sys.executable, "-m", "venv", str(environment)])
        python = _venv_python(environment)
        _run(
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
            ]
        )
        _run([make, "docs", f"PYTHON={python}"], cwd=source)

        rendered = [
            source / "site" / "index.html",
            source / "site" / "api" / "regression" / "index.html",
            source / "site" / "api" / "inspection" / "index.html",
        ]
        missing_rendered = [
            path.relative_to(source).as_posix() for path in rendered if not path.is_file()
        ]
        if missing_rendered:
            raise RuntimeError(
                f"Documentation build did not create expected pages: {missing_rendered}"
            )

    print("Source-distribution documentation build passed.")


if __name__ == "__main__":
    main()
