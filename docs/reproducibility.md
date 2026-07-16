# Reproducibility

## Real-data integrations

Each repository dataset records its source, citation, redistribution terms, analysis-facing files,
row-order contract, missing-value policy, and integrity hashes when appropriate. These assets
support repository review; they do not become runtime requirements for fitting user-supplied
`X` and `Y`.

The first integration is `datasets/linnerud/`. Its upstream values come from scikit-learn 1.8.0
and are stored as comma-delimited `X.csv` and `Y.csv`. `metadata.yaml` records the conversion,
source, license, dimensions, variables, row alignment, missingness, and SHA-256 hashes. The
executable example reads only the two model tables directly rather than calling a package loader
or metadata parser.

Manuscript-specific datasets will be migrated separately after their source, licensing, and
scientific preparation choices are resolved.
