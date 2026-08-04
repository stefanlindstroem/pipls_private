# Decision 0033: remove the Linnerud reference integration

## Status

Accepted after the first real-data smoke-check review.

## Context

The Linnerud dataset was introduced as a compact demonstration of transparent `X.csv` and `Y.csv`
reading. Subsequent review showed that it does not provide a useful package-level scientific or
validation example: it has only 20 observations, its selection-conditioned out-of-fold diagnostics
are uninformative, and it was not part of the original Pi-PLS study. The Pulp integration already
demonstrates the same ordinary pandas-to-`fit(X, Y)` workflow on a more relevant multivariate
dataset.

Keeping an example that programming users can reasonably interpret as evidence for model quality
would give the wrong impression. The transparent input contract does not depend on this particular
dataset.

## Decision

- Remove `datasets/linnerud/` and all of its redistributed files.
- Remove `examples/09_linnerud_real_data.py` and its dataset-specific execution test.
- Remove Linnerud from active dataset, example, reproducibility, roadmap, and product navigation.
- Keep the general repository dataset-layout contract and direct pandas I/O policy unchanged.
- Keep Pulp, sugarcane, and tobacco as the current transparent real-data suite.
- Record the retired first-dataset decision in the Decision 0147 retirement map; Git preserves
  its full text.

## Consequences

- Source distributions no longer contain the Linnerud tables, license, metadata, or example.
- The package no longer presents a weak small-sample result as a representative real-data use case.
- Pulp is the compact ordinary component-path example and the first implemented real-data smoke
  check.
- Dataset numbering is not renumbered; existing Pulp, sugarcane, and tobacco example filenames
  remain stable.
