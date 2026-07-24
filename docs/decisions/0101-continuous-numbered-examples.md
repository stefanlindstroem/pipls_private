# Decision 0101: continuous numbered examples

## Context

The maintained example sequence jumped from 03 to 09. The gap was a historical artifact from
removed scripts, but the package has not been released and new users reasonably interpreted the
missing numbers as absent examples. The first example also printed an unlabeled prediction matrix
and gave no terminal confirmation of its generated figure.

## Decision

1. Number the seven maintained examples continuously from 01 through 07 in increasing workflow
   complexity.
2. Rename the comparison and real-data examples from 09--12 to 04--07 and migrate every active
   documentation, dataset, tool, test, and guide-layer reference in the same patch.
3. Preserve result-directory names, generated PDF filenames, numerical calculations, datasets, and
   workflow ordering.
4. Keep historical decision records unchanged; this decision supersedes their old user-facing
   example numbers.
5. Make example 01 label its prediction output and report the PDF path it writes.
6. Test both the exact continuous sequence and the order used by `make examples`.

## Consequences

The example catalogue now reads as a complete sequence rather than a partially deleted one. Existing
repository users must use the new script filenames, but no compatibility copies or aliases are
retained because the package has not been released. Tutorial snippet extraction and source-
distribution documentation checks follow the renamed Pulp example directly.
