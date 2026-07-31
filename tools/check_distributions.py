"""Build and validate clean installations of the wheel and source distribution."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_SMOKE_TEST = """\
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import pipls
import pipls.datasets
import pipls.inspection
import pipls.metrics
import pipls.search
from pipls import PiPLSRegression, PiPLSSearchCV

repository = Path(sys.argv[1]).resolve()
artifact_label = sys.argv[2]
package_file = Path(pipls.__file__).resolve()
source_root = (repository / "src").resolve()
environment_root = Path(sys.prefix).resolve()

try:
    package_file.relative_to(source_root)
except ValueError:
    pass
else:
    raise AssertionError(f"pipls imported from the repository checkout: {package_file}")

try:
    package_file.relative_to(environment_root)
except ValueError as error:
    raise AssertionError(
        f"pipls did not import from the clean virtual environment: {package_file}"
    ) from error

assert version("pipls") == pipls.__version__
assert PiPLSSearchCV.__name__ == "PiPLSSearchCV"

rng = np.random.default_rng(0)
X = rng.normal(size=(12, 4))
Y = np.column_stack((X[:, 0] + X[:, 1], X[:, 2] - X[:, 3]))
model = PiPLSRegression(n_components=1, predictor_rank=2).fit(X, Y)
prediction = model.predict(X[:2])

assert prediction.shape == (2, 2)
assert np.isfinite(prediction).all()
assert pipls.__version__

print(
    f"{artifact_label} installation passed: "
    f"pipls {pipls.__version__} from {package_file}"
)
"""


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


def _venv_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _single_artifact(artifacts: Path, pattern: str, label: str) -> Path:
    matches = sorted(artifacts.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {label}, found {len(matches)}.")
    return matches[0]


def _extract_source_distribution(artifact: Path, destination: Path) -> Path:
    destination.mkdir()
    destination_resolved = destination.resolve()
    with tarfile.open(artifact, mode="r:gz") as archive:
        members = archive.getmembers()
        roots = {Path(member.name).parts[0] for member in members if Path(member.name).parts}
        if len(roots) != 1:
            raise RuntimeError("The source distribution must contain one top-level directory.")
        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe source-distribution member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Source-distribution links are unsupported: {member.name}")
            target = (destination / member_path).resolve()
            if not target.is_relative_to(destination_resolved):
                raise RuntimeError(f"Source-distribution member escapes extraction: {member.name}")
        if sys.version_info >= (3, 12):
            archive.extractall(destination, filter="data")
        else:
            archive.extractall(destination)
    return destination / roots.pop()


def _check_source_distribution_example(
    *,
    artifact: Path,
    workspace: Path,
    python: Path,
    environment_variables: dict[str, str],
) -> None:
    source = _extract_source_distribution(artifact, workspace / "sdist-source")
    result_directories = (
        source / "examples" / "results",
        source / "examples" / "results" / "pls_path_comparison",
        source / "examples" / "results" / "synthetic_tutorial",
        source / "examples" / "results" / "pulp_post_analysis",
        source / "examples" / "results" / "sugarcane_post_analysis",
        source / "examples" / "results" / "tobacco_post_analysis",
    )
    missing = [
        path.relative_to(source).as_posix()
        for path in result_directories
        if not path.is_dir()
    ]
    if missing:
        raise RuntimeError(f"Source distribution is missing example result directories: {missing}")

    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-compile",
            "--disable-pip-version-check",
            "--no-input",
            "--retries",
            "5",
            "--timeout",
            "60",
            f"pipls[examples] @ {artifact.resolve().as_uri()}",
        ],
        cwd=source,
        env=environment_variables,
    )
    example_environment = environment_variables.copy()
    example_environment["MPLBACKEND"] = "Agg"
    _run(
        [str(python), str(source / "examples" / "01_minimal_fit_and_plot.py")],
        cwd=source,
        env=example_environment,
    )
    output = source / "examples" / "results" / "minimal_fit_and_plot.pdf"
    if not output.is_file():
        raise RuntimeError("The source-distribution minimal example did not create its PDF output.")


def _check_installation(
    *,
    label: str,
    artifact: Path,
    workspace: Path,
    smoke_test: Path,
    repository: Path,
    environment_variables: dict[str, str],
    check_source_example: bool,
) -> None:
    environment = workspace / f"{label}-venv"
    run_directory = workspace / f"{label}-run"
    run_directory.mkdir()

    _run([sys.executable, "-m", "venv", str(environment)], env=environment_variables)
    python = _venv_python(environment)
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-compile",
            "--disable-pip-version-check",
            "--no-input",
            "--retries",
            "5",
            "--timeout",
            "60",
            str(artifact),
        ],
        cwd=run_directory,
        env=environment_variables,
    )
    _run(
        [str(python), str(smoke_test), str(repository), label],
        cwd=run_directory,
        env=environment_variables,
    )
    if check_source_example:
        _check_source_distribution_example(
            artifact=artifact,
            workspace=workspace,
            python=python,
            environment_variables=environment_variables,
        )


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    environment_variables = _subprocess_environment()

    with tempfile.TemporaryDirectory(prefix="pipls-dist-check-") as temporary:
        workspace = Path(temporary)
        environment_variables["PIP_CACHE_DIR"] = str(workspace / "pip-cache")
        artifacts = workspace / "artifacts"
        artifacts.mkdir()
        smoke_test = workspace / "installed_smoke_test.py"
        smoke_test.write_text(_SMOKE_TEST, encoding="utf-8")

        _run(
            [sys.executable, "-m", "build", "--quiet", "--outdir", str(artifacts)],
            cwd=repository,
            env=environment_variables,
        )
        wheel = _single_artifact(artifacts, "*.whl", "wheel")
        source_distribution = _single_artifact(artifacts, "*.tar.gz", "source distribution")

        checks = (
            ("wheel", wheel, False),
            ("sdist", source_distribution, True),
        )
        with ThreadPoolExecutor(max_workers=len(checks)) as executor:
            futures = [
                executor.submit(
                    _check_installation,
                    label=label,
                    artifact=artifact,
                    workspace=workspace,
                    smoke_test=smoke_test,
                    repository=repository,
                    environment_variables=environment_variables,
                    check_source_example=check_source_example,
                )
                for label, artifact, check_source_example in checks
            ]
            for future in futures:
                future.result()

    print("Wheel and source-distribution installation checks passed.")


if __name__ == "__main__":
    main()
