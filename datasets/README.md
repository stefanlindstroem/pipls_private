# Datasets

No research dataset is bundled yet. Real data enters this directory one dataset at a time after
source, citation, licensing, redistribution, and deterministic preparation choices are reviewed.

The installed package already provides the Phase E1 in-memory dataset container and deterministic
synthetic generator under `pipls.datasets`; see `docs/datasets.md`. Synthetic data are generated
at runtime and are not committed as dataset files.


Analysis examples must read their predictor and response files explicitly and form `X` and `Y`
without a generic package loader. Human-readable provenance or preparation metadata may accompany a
repository dataset, but it is not required for external users fitting their own data.
