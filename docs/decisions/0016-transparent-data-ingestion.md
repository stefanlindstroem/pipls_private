# Decision 0016: transparent real-data ingestion and examples

## Status

Accepted before the first real-dataset migration; clarified by Decision 0020.

## Context

Pi-PLS accepts predictor and response matrices, while real data may originate from many file
formats, databases, instruments, and domain-specific systems. A public registry or generic loader
would make repository examples look convenient at the cost of obscuring the actual programming
contract and expanding the package into general data-management infrastructure.

## Decision

- The primary real-data contract is `fit(X, Y)` with arrays or data frames prepared by the user.
- No metadata file, registry, checksum manifest, or `PiPLSDataset` instance is required for fitting.
- `PiPLSDataset` remains optional and is appropriate for package-owned synthetic data or structured
  experiments.
- Examples and paper-reproduction scripts read data, align rows, select columns, and form `X` and
  `Y` explicitly in the script using ordinary NumPy, pandas, or domain-specific user code.
- Package or example helper functions must not hide analysis-time data reading.
- Repository provenance assets do not become runtime requirements for external users.
- Committed dataset documentation uses only public or included sources; private development
  lineage and preparation-only scripts remain outside the public repository.
- No generic public registry or real-data loader is planned.

## Consequences

- Users can adopt Pi-PLS without conforming their data sources to a package-specific metadata
  system.
- Examples teach the same transparent workflow users are expected to follow.
- Real-dataset patches remain dataset-specific and reviewable.
- Public provenance and analysis-time simplicity are maintained without exposing internal project
  inputs.
