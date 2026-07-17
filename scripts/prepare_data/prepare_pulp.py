"""Prepare the repository pulp X and Y tables from Vishal's source CSV."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_SHA256 = "e5aeb8dacc4ae938b30b109da1cbcc1fef290f5cf7774295a2612e44ddc6a468"
SOURCE_COLUMNS = [
    "FZ gap",
    "CZ gap",
    "Dilution screw",
    "Dilution CZ",
    "Dos. screw",
    "SRE",
    "Shives",
    "Fines B",
    "L (arith)",
    "L (lw)",
    "L (llw)",
    "W (arith)",
    "W (lw)",
    "W (llw)",
    "C (arith)",
    "C (lw)",
    "C (llw)",
    "F (arith)",
    "F (lw)",
    "F (llw)",
    "CSF",
    "Density",
    "TI",
    "Elongation",
    "TEA",
    "TSI",
    "Tear index",
    "s",
    "k",
]
PREDICTOR_COLUMNS = SOURCE_COLUMNS[6:20]
RESPONSE_COLUMNS = SOURCE_COLUMNS[20:28]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_pulp(source: Path, output_dir: Path, *, verify_source: bool = True) -> None:
    """Validate the supplied source table and write transparent X/Y CSV files."""

    if verify_source and _sha256(source) != SOURCE_SHA256:
        raise ValueError(
            "Unexpected pulp source checksum. Use the unmodified "
            "PiPLSR_v0.1/data/pulp.csv file."
        )

    table = pd.read_csv(source)
    if list(table.columns) != SOURCE_COLUMNS:
        raise ValueError("Unexpected pulp source columns or column order.")
    if table.shape != (46, 29):
        raise ValueError(f"Unexpected pulp source shape: {table.shape!r}.")

    numeric = table.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Pulp source data must contain only finite numeric values.")

    output_dir.mkdir(parents=True, exist_ok=True)
    numeric[PREDICTOR_COLUMNS].to_csv(
        output_dir / "X.csv", index=False, lineterminator="\n"
    )
    numeric[RESPONSE_COLUMNS].to_csv(
        output_dir / "Y.csv", index=False, lineterminator="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to PiPLSR_v0.1/data/pulp.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/pulp"),
        help="Directory receiving X.csv and Y.csv",
    )
    args = parser.parse_args()
    prepare_pulp(args.source, args.output_dir)
    print(f"X.csv sha256: {_sha256(args.output_dir / 'X.csv')}")
    print(f"Y.csv sha256: {_sha256(args.output_dir / 'Y.csv')}")


if __name__ == "__main__":
    main()
