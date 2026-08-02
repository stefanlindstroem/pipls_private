# Archived Pulp repository layout

This directory preserves the former repository-facing Pulp dataset layout exactly as it existed
before Decision 0138 completed the transition to the package-owned `load_pulp()` dataset.

The archived files are development history only. They are not the canonical runtime dataset and
must not be read by package code, maintained examples, tests, served documentation, wheels, or
source distributions. The active resources are under `src/pipls/_data/pulp/` and are accessed
through `pipls.datasets.load_pulp()`.

The five inherited files are retained byte-for-byte for temporary historical reference. This
archive may be removed in a later owner-authorized cleanup after the loader contract has
stabilized.
