# Retired decisions

Decision 0147 permits a numbered record to leave the maintained tree when a later canonical
record fully replaces its active contract. Git remains the authoritative archive, decision numbers
are never reused, and this map records where maintainers should look for the current policy.

To recover a retired record, locate its last revision with:

```bash
git log --all -- docs/decisions/<retired-filename>
git show <revision>:docs/decisions/<retired-filename>
```

## Decision 0147 Patch 3

| Retired record | Current canonical replacement | Reason for retirement |
|---|---|---|
| `0017-first-real-dataset.md` | [0033](0033-remove-linnerud-integration.md) | Linnerud and its example were removed; the retained decision defines the current dataset boundary. |
| `0027-synthetic-benchmark-contract.md` | [0125](0125-retire-benchmark-layer.md) | The benchmark manifest and schema no longer exist. |
| `0028-synthetic-ci-benchmark-runner.md` | [0125](0125-retire-benchmark-layer.md) | The benchmark runner and its CI contract were removed. |
| `0029-human-and-machine-readable-results.md` | [0125](0125-retire-benchmark-layer.md) | The benchmark result-output contract was retired with the benchmark layer. |
| `0030-focused-benchmark-design.md` | [0125](0125-retire-benchmark-layer.md) | The later decision removes the complete focused benchmark layer. |
| `0035-tobacco-randomized-auto-path.md` | [0036](0036-real-data-pls-path-comparison.md) | The maintained Tobacco workflow uses the full solver; randomized-SVD behavior is tested separately. |
| `0106-fold-based-cv-standard-error.md` | [0146](0146-cv-mse-tolerance-selection.md) | Split SD is descriptive only and the public standard-error surface was removed. |
| `0107-component-path-recommendation-methods.md` | [0140](0140-search-owned-path-selection.md), [0146](0146-cv-mse-tolerance-selection.md) | Selection moved from path methods to search-owned tolerance rules. |
| `0108-tobacco-one-standard-error-workflow.md` | [0146](0146-cv-mse-tolerance-selection.md) | Tobacco now demonstrates 10% relative-tolerance minimum-CV-MSE selection. |
| `0109-tobacco-one-standard-error-threshold-figure.md` | [0146](0146-cv-mse-tolerance-selection.md) | The maintained figure now shows split SD and the tolerance threshold. |
| `0111-explicit-path-selection-rules.md` | [0137](0137-post-fit-inspect-decide-refit-lifecycle.md), [0140](0140-search-owned-path-selection.md), [0143](0143-model-selection-provenance-and-oof-reporting.md), [0146](0146-cv-mse-tolerance-selection.md) | Constructor-owned refitting, selected-search state, and the 1-SE rule were replaced by post-fit selection, refitting, provenance, and tolerance selection. |
| `0116-composed-validation-report-result.md` | [0143](0143-model-selection-provenance-and-oof-reporting.md), [0144](0144-pre-release-public-surface-cleanup.md) | The former validation report was replaced by `PiPLSOOFReport` carrying one immutable selection. |
| `0129-remove-pre-release-search-aliases.md` | [0137](0137-post-fit-inspect-decide-refit-lifecycle.md), [0144](0144-pre-release-public-surface-cleanup.md), [0145](0145-final-implementation-surface-cleanup.md) | The selected-estimator search state and its aliases were removed entirely. |
