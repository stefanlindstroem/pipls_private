"""Smoke-test a clean installation of Pi-PLS outside the repository checkout."""

from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

import pipls
from pipls import PiPLSRegression, PiPLSSearchCV
from pipls.datasets import load_pulp, load_sugarcane, load_tobacco


def main() -> None:
    repository = Path(sys.argv[1]).resolve()
    artifact_label = sys.argv[2]
    package_file = Path(pipls.__file__).resolve()
    source_root = (repository / "src").resolve()
    environment_root = Path(sys.prefix).resolve()

    if package_file.is_relative_to(source_root):
        raise AssertionError(
            f"pipls imported from the repository checkout: {package_file}"
        )
    if not package_file.is_relative_to(environment_root):
        raise AssertionError(
            f"pipls did not import from the clean environment: {package_file}"
        )
    if version("pipls") != pipls.__version__:
        raise AssertionError("Installed metadata and package versions disagree.")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4))
    Y = np.column_stack((X[:, 0] + X[:, 1], X[:, 2] - X[:, 3]))

    model = PiPLSRegression(n_components=1, predictor_rank=2).fit(X, Y)
    prediction = model.predict(X[:2])
    if prediction.shape != (2, 2) or not np.isfinite(prediction).all():
        raise AssertionError("Installed fixed-estimator prediction failed.")

    search = PiPLSSearchCV(
        n_components_values=(1,),
        predictor_rank_values=(1, 2),
        search_method="exhaustive",
        cv=2,
        n_jobs=1,
    ).fit(X, Y)
    selection = search.select(n_components=1)
    refitted = search.refit(X, Y, selection=selection)
    if not np.isfinite(refitted.predict(X[:2])).all():
        raise AssertionError("Installed search and refit workflow failed.")

    expected_shapes = {
        "pulp": ((46, 14), (46, 8)),
        "sugarcane": ((57, 1721), (57, 4)),
        "tobacco": ((347, 1557), (347, 13)),
    }
    for name, loader in (
        ("pulp", load_pulp),
        ("sugarcane", load_sugarcane),
        ("tobacco", load_tobacco),
    ):
        dataset = loader()
        if (dataset.X.shape, dataset.Y.shape) != expected_shapes[name]:
            raise AssertionError(f"Installed {name} dataset has unexpected shapes.")

    print(
        f"{artifact_label} installation passed: "
        f"pipls {pipls.__version__} from {package_file}"
    )


if __name__ == "__main__":
    main()
