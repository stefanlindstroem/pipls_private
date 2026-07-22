"""Build the documentation from a clean installation of the source distribution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

SYNTHETIC_TUTORIAL_FIGURES = (
    "component_path.svg",
    "predictor_rank_profile.svg",
    "observed_vs_predicted.svg",
)
PULP_TUTORIAL_FIGURES = (
    "component_path.svg",
    "predictor_rank_profile.svg",
    "biplot.svg",
    "predictor_directions.svg",
    "observed_vs_predicted.svg",
    "standardized_rmse.svg",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _validate_figure_manifest(
    generated_dir: Path,
    expected_figures: tuple[str, ...],
    *,
    tutorial_name: str,
) -> dict[str, object]:
    manifest_path = generated_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Documentation build did not generate the {tutorial_name} manifest.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = tuple(item["filename"] for item in manifest.get("figures", []))
    if declared != expected_figures:
        raise RuntimeError(
            f"{tutorial_name} manifest does not declare the expected figures: {declared!r}."
        )
    figure_records = {item["filename"]: item for item in manifest["figures"]}
    for filename in expected_figures:
        figure_path = generated_dir / filename
        if not figure_path.is_file():
            raise RuntimeError(f"Missing generated {tutorial_name} figure: {filename}")
        ET.parse(figure_path)
        if figure_records[filename]["sha256"] != _sha256(figure_path):
            raise RuntimeError(f"Generated {tutorial_name} figure hash disagrees: {filename}")
    return manifest


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
            source / "docs" / "tutorials" / "synthetic.md",
            source / "docs" / "tutorials" / "pulp.md",
            source / "docs" / "api" / "index.md",
            source / "docs" / "troubleshooting.md",
            source / "docs" / "javascripts" / "mathjax.js",
            source / "tools" / "render_synthetic_tutorial.py",
            source / "tools" / "render_pulp_tutorial.py",
            source / "examples" / "02_synthetic_path_selection.py",
            source / "examples" / "10_pulp_real_data.py",
            source / "datasets" / "pulp" / "X.csv",
            source / "datasets" / "pulp" / "Y.csv",
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

        synthetic_dir = source / "docs" / "assets" / "generated" / "synthetic"
        synthetic_manifest = _validate_figure_manifest(
            synthetic_dir,
            SYNTHETIC_TUTORIAL_FIGURES,
            tutorial_name="synthetic tutorial",
        )
        synthetic_analysis = synthetic_manifest.get("analysis", {})
        if synthetic_analysis.get("chosen_n_components") != 2:
            raise RuntimeError("Synthetic tutorial component selection changed unexpectedly.")
        if synthetic_analysis.get("chosen_predictor_rank") != 4:
            raise RuntimeError("Synthetic tutorial predictor-rank selection changed unexpectedly.")
        if not math.isfinite(float(synthetic_analysis.get("external_test_r2", math.nan))):
            raise RuntimeError("Synthetic tutorial external-test R2 must be finite.")

        pulp_dir = source / "docs" / "assets" / "generated" / "pulp"
        pulp_manifest = _validate_figure_manifest(
            pulp_dir,
            PULP_TUTORIAL_FIGURES,
            tutorial_name="Pulp tutorial",
        )
        dataset_hashes = pulp_manifest.get("dataset", {}).get("files", {})
        for filename in ("X.csv", "Y.csv"):
            dataset_path = source / "datasets" / "pulp" / filename
            if dataset_hashes.get(filename) != _sha256(dataset_path):
                raise RuntimeError(f"Generated Pulp tutorial dataset hash disagrees: {filename}")

        rendered = [
            source / "site" / "index.html",
            source / "site" / "tutorials" / "synthetic" / "index.html",
            source / "site" / "tutorials" / "pulp" / "index.html",
            source / "site" / "api" / "regression" / "index.html",
            source / "site" / "troubleshooting" / "index.html",
            source / "site" / "api" / "inspection" / "index.html",
            source / "site" / "assets" / "generated" / "synthetic" / "component_path.svg",
            source
            / "site"
            / "assets"
            / "generated"
            / "synthetic"
            / "observed_vs_predicted.svg",
            source / "site" / "assets" / "generated" / "pulp" / "component_path.svg",
            source / "site" / "assets" / "generated" / "pulp" / "standardized_rmse.svg",
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
