# Project map

## Purpose

Pi-PLS is a PLS-family method for multivariate regression. This repository is the long-lived home of
the installable `pipls` package: numerical construction, scikit-learn-compatible estimators,
selection and validation results, synthetic and reference datasets, numerical inspection,
documentation, examples, tests, packaging, and releases.

The repository is not a paper-reproduction environment. Read `.llm/product_scope.md` for the
normative product/publication boundary and `.llm/state.md` for the current handoff and active
maintenance increment.

## Current package

The top-level public surface is deliberately narrow:

```python
from pipls import PiPLSRegression, PiPLSSearchCV, PredictorRankSupportWarning
```

Focused modules expose immutable result records, response-standardized metrics, datasets, synthetic
generators, validation results, and pure numerical inspection.

`PiPLSRegression` fits one explicit `(n_components, predictor_rank)` pair. `PiPLSSearchCV` evaluates
the triangular path and supports explicit post-search `select()`, `refit()`, and `oof_report()`
operations. The supported named selection rules are `best_score` and tolerance-based
`minimum_cv_mse`. Refitted models retain the exact immutable `selection_`.

The package owns no plotting module. Maintained examples and tutorial renderers build Matplotlib
figures directly from immutable numerical results. Optional `textalloc` may reposition annotated
biplot labels while avoiding predictor-arrow shafts, but is not part of the numerical contract.

## Data and product assets

Pulp, Sugarcane, and Tobacco are the closed set of package-owned reference datasets. Their sole
active matrix copies are language-neutral resources under `src/pipls/_data/<dataset>/`, with
metadata, README, and license files. Public loaders are available from `pipls.datasets`.

Arbitrary user data remains ordinary array-like `X` and `y`. The package has no generic registry,
network downloader, DataFrame requirement, or hidden preprocessing layer. Corn, the legacy
Citrination Steel table, SARCOS, and FRED-MD remain excluded because the exact candidate materials
do not have sufficiently clear redistribution rights.

The maintained numbered examples are:

1. Pulp automatic fit and fitted-value diagnostic;
2. synthetic inspect-decide-refit workflow with external-test prediction;
3. matched-fold PLS-family component-path comparison across Pulp, Sugarcane, Tobacco, and one
   deterministic near-saturated synthetic stress case, covering the peer-reviewed cross-covariance
   Pi-PLS policy, the least-squares software extension, and ordinary PLS without final refitting;
4. repeated-CV Pulp analysis;
5. Sugarcane analysis with EPV-fixed predictor rank at `samples_per_predictor_rank=5.0` and a
   separate component-count choice;
6. Tobacco analysis with optimized predictor rank and separate 20% predictor-rank and
   component-count tolerances.

`make examples` owns complete application validation. The default test suite does not duplicate the
full real-data workflows.

## Runtime ownership

- `src/pipls/_core.py`: fixed-rank numerical construction.
- `src/pipls/_cv_engine.py`: fold-local candidate evaluation, scoring, timing, and OOF support.
- `src/pipls/_model_selection.py`: private rank limits, split materialization, and rank-search
  orchestration.
- `src/pipls/_result_validation.py`: shared immutable-result validation.
- `src/pipls/_sklearn_compat.py`: supported scikit-learn validation and tag compatibility.
- `src/pipls/regression.py`: `PiPLSRegression`.
- `src/pipls/search.py`: `PiPLSSearchCV`.
- `src/pipls/component_path.py`: component path, scalar selection, and predictor-rank profile.
- `src/pipls/decomposition.py`: immutable Pi-PLS decomposition.
- `src/pipls/validation.py`: immutable OOF report.
- `src/pipls/inspection.py`: fitted-model numerical inspection.
- `src/pipls/metrics.py`: response-standardized metrics.
- `src/pipls/datasets.py`: stable public dataset façade and `__all__`.
- `src/pipls/_dataset_types.py`: dataset record plus shared matrix, label, and metadata validation.
- `src/pipls/_dataset_resources.py`: packaged Pulp, Sugarcane, and Tobacco loading.
- `src/pipls/_synthetic_data.py`: deterministic synthetic-data generator.
- `src/pipls/exceptions.py`: package warning types.
- `src/pipls/__init__.py`: deliberate top-level exports.

## Test ownership

- `tests/unit/`: local behavior, boundary conditions, search orchestration, and leakage prevention.
- `tests/invariants/`: mathematical identities, dimensions, orthogonality, and subspace properties.
- `tests/api/`: public validation, estimator compatibility, pipelines, and validation protocols.
- `tests/regression/`: frozen comparisons with trusted implementations.
- `tests/examples/`: focused workflow, rendering, and application-structure contracts.
- root-level tests: repository, documentation, distribution, generated-asset, and decision
  integrity.

Tests protect durable behavior and policy. They do not preserve every historical implementation
arrangement or exact explanatory sentence.

## Documentation and distribution ownership

- `docs/`: self-contained served documentation plus maintainer decisions under `docs/decisions/`.
- `docs/decisions/index.md`: current numbered decisions.
- `docs/decisions/history.md`: compact completed-era summary.
- `docs/decisions/retirements.md`: exhaustive retired-filename replacement map.
- `.llm/`: current maintainer contracts and workflow helpers; not served or installed.
- `mkdocs.yml`: strict Material site; decision records are excluded from served navigation.
- `constraints/minimum.txt`: maintainer-only minimum dependency test input.
- `.github/workflows/`: compatibility, build, documentation, and deployment validation.
- `tools/`: distribution and documentation validation and deterministic tutorial rendering.

Python 3.10--3.14 is supported within the guarded NumPy, scikit-learn, and joblib ranges declared in
`pyproject.toml`. Compatibility validation separates the minimum stack, normal resolution on every
supported Python, and latest-compatible upgrades. Wheel and source-distribution installations are
checked in clean environments.

Snapshots are root-relative archives of a clean committed Git tree. Generated `site/`, build
outputs, caches, bytecode, and example artifacts are not package state.

## Contract ownership

- `.llm/state.md`: current handoff, exclusions, and roadmap.
- `.llm/product_scope.md`: package and publication boundary.
- `.llm/strategy.md`: current development principles and active sequence.
- `.llm/decisions.md`: current decision registry.
- `.llm/theory.md` and `.llm/mathematics.md`: conceptual and normative mathematical contracts.
- `.llm/numerical_contracts.md`: numerical and degeneracy policy.
- `.llm/public_api.md`: constructors, methods, results, defaults, and exclusions.
- `.llm/data_io.md` and `.llm/dataset_layout.md`: data ownership and resource layout.
- `.llm/analysis.md`: interpretation, OOF provenance, rendering, and artifact boundaries.
- `.llm/testing.md`: durable testing obligations.
- `.llm/development.md`: implementation, patch, and validation procedure.

## Architectural invariants

- Runtime code does not import from `.llm`, tests, examples, docs, or tools.
- The fixed core owns no preprocessing, cross-validation, datasets, or publication workflow.
- Centering and optional scaling are learned inside every fit and every search training fold.
- Fixed fitting and path search do not wrap each other.
- Search retains evidence but not training matrices or an implicit final estimator.
- OOF reporting reuses the exact materialized search splits and accepts an existing compatible
  selection; evidence-retaining workflows pass that same object to final refitting.
- Numerical inspection is package-owned; plotting and report composition are caller-owned.
- Public behavior changes include focused tests and synchronized documentation and decisions.
- Paper-specific experiments and reporting belong downstream of tagged package releases.
- Changes are bounded, reviewable, and returned as root-relative Git patches.

## Validation

```bash
make check
make docs
make examples
make build
make dist-check
```

Run only the targets applicable to the patch, but record each as passed, failed, or unavailable.
