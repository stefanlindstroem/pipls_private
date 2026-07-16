# Reproducibility

## Real-data integrations

Each repository dataset records its source, citation, redistribution terms, analysis-facing files,
row-order contract, missing-value policy, and integrity hashes when appropriate. These assets
support repository review; they do not become runtime requirements for fitting user-supplied
`X` and `Y`.

The first integration is `datasets/linnerud/`. Its two data files are copied verbatim from
scikit-learn 1.8.0, and `checksums.sha256` freezes their repository contents. The executable example
reads both files directly rather than calling a package loader.

Manuscript-specific datasets will be migrated separately after their source, licensing, and
scientific preparation choices are resolved.
