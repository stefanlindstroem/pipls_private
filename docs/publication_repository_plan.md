# Pi-PLS publication repository plan, revision 5

**Status:** specified architecture after author decisions on fold-safe rank bounds, leave-one-out performance reporting, response-standardized model selection, and an integrated LLM-assisted development workflow  
**Primary inputs reviewed:** `pipls.tar.gz`, `PiPLSR_v0.1.tar.gz`, and `pipls.tex`  
**Recommended release scope:** the basic Pi-PLS method described in the manuscript, a PLS-style scikit-learn estimator with automatic predictor-rank selection, fold-safe cross-validation bounds, response-standardized model selection, leave-one-out performance reporting, common-format datasets, and paper-reproduction scripts

**Post-plan decision:** decision 0007 refines the predictor-rank search vocabulary. Exhaustive
conditional CV search is now targeted as `predictor_rank="optimal"`; `predictor_rank="auto"` is
reserved for deterministic adaptive coarse-to-fine search. Where revision 5 describes exhaustive
search under the name `"auto"`, decision 0007 supersedes the name while preserving the fold-safe
bound, fold-local preprocessing, scoring, and tie-breaking contracts. Randomized SVD is governed
by a separate future solver policy.

## 1. Executive recommendation

Create a clean repository rather than publishing either supplied archive directly. The new repository should combine:

- the Pi-PLS core algorithm from Vishal's implementation, because it matches the manuscript construction;
- the `src/` package layout, stricter parameter validation, scalar `score()` behavior, and substantial test suite from the other implementation;
- a public estimator whose ordinary use mirrors `sklearn.cross_decomposition.PLSRegression`;
- a model-selection layer built on scikit-learn splitters, scorers, `Pipeline`, and `GridSearchCV` conventions;
- a versioned dataset area in which every analysis-ready dataset has `X.csv`, `Y.csv`, and `metadata.yaml`, with optional `samples.csv` and `splits.csv`;
- deterministic preparation scripts that convert the original source files into the common dataset format;
- separate examples for PLS-style fitting, automatic predictor-rank selection, complete two-parameter path analysis, leave-one-out selection, custom preprocessing, rule-fixed rank, explicitly fixed rank, and paper reproduction;
- a tracked but non-installable `.llm/` layer containing the repository map, mathematical and numerical contracts, development rules, reusable request templates, snapshot tooling, and guarded root-relative patch application.

The principal public call should be:

```python
from pipls import PiPLSRegression

model = PiPLSRegression(n_components=3)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
```

The programming parameter `n_components` denotes the theoretical parameter $h$. The second theoretical parameter $r_\pi$ should not be required in ordinary use. It should be represented publicly as `predictor_rank`, default to `"auto"`, and be selected by cross-validation conditional on `n_components`. The rule parameter $c$ should be represented publicly as `samples_per_predictor_rank`. Its role is to determine the upper admissible predictor rank, not normally to determine the final fitted rank.

The default procedure for a fixed value of `n_components` is therefore:

\begin{equation}
r_{\pi,\max}
=
\min\left[p,n_{\mathrm{train,min}},
\left\lceil
\frac{n_{\mathrm{train,min}}}
{\texttt{samples\_per\_predictor\_rank}}
\right\rceil
\right],
\end{equation}

followed by

\begin{equation}
r_\pi^*(h)
=
\operatorname*{arg\,min}_{r_\pi\in\{h,\ldots,r_{\pi,\max}\}}
\widehat{\operatorname{MSE}}_{\mathrm{CV,response\text{-}std}}(h,r_\pi).
\end{equation}

Here $n_{\mathrm{train,min}}$ is the smallest training-fold size produced by the supplied splitter. Under `LeaveOneOut()`, $n_{\mathrm{train,min}}=n-1$. The selection loss is response-standardized MSE: each held-out residual is divided by the corresponding response scale estimated from that fold's training observations before squaring and aggregation.

The general API and the paper-reproduction protocol are intentionally separated. The API always uses the fold-safe $n_{\mathrm{train,min}}$ rule. The reproduction scripts preserve the manuscript's current sample-size-dependent rule and reported rank grid explicitly, including any full-$n$ calculation needed to reproduce published values; that paper-specific choice is passed as explicit rank limits or rank values and is not hidden in the estimator.

The fitted value should be exposed as `predictor_rank_`. Two nondefault modes should remain available:

```python
PiPLSRegression(n_components=3, predictor_rank="max")
PiPLSRegression(n_components=3, predictor_rank=8)
```

The first uses the rule-derived upper rank directly and scans only over `n_components` when an external search is used. The second fixes the rank explicitly and is intended for reproduction, simulation, testing, or externally controlled searches.

The paper release should contain only the standard response-subspace construction based on the right singular vectors of $\mathbf{Z}^{\mathsf{T}}\mathbf{Y}$. The trial response-subspace alternatives and block-scaling implementation should not be part of the stable paper API. The architecture should permit a future `BlockScaler` to be inserted as an ordinary scikit-learn transformer without altering `PiPLSRegression`.

The public integration target should be described as the **SciPy ecosystem with a scikit-learn estimator API**. The familiar `PLSRegression` class, `Pipeline`, cross-validation splitters, and parameter-search classes are supplied by scikit-learn, not by SciPy itself.

## 2. Findings from the supplied material

### 2.1 Agreement on the mathematical core

Vishal's `pipls/core.py` follows the manuscript algorithm:

1. Compute the leading $r_\pi$ right singular vectors of centered $\mathbf{X}$ and collect them in $\mathbf{\Pi}$.
2. Form $\mathbf{Z}=\mathbf{X}\mathbf{\Pi}$.
3. Obtain $\mathbf{C}$ from the leading $h$ right singular vectors of $\mathbf{Z}^{\mathsf{T}}\mathbf{Y}$.
4. Solve $\mathbf{Z}\mathbf{W}\approx\mathbf{Y}\mathbf{C}$ by least squares.
5. Compute $\mathbf{W}=\mathbf{M}\mathbf{D}\mathbf{N}^{\mathsf{T}}$.
6. Return $\mathbf{P}=\mathbf{\Pi}\mathbf{M}$, $\mathbf{D}$, and $\mathbf{Q}=\mathbf{C}\mathbf{N}$.

The `response_subspace="xcov"` path in the other implementation is numerically equivalent to this construction. On a deterministic synthetic comparison, the two implementations produced a maximum prediction difference of approximately $8\times 10^{-13}$. This is useful: Vishal's implementation can define the theory oracle, while the more developed estimator can supply part of the API and test infrastructure.

### 2.2 Useful elements in the independently developed package

The smaller `pipls` archive has several features worth retaining:

- standard `src/pipls/` packaging;
- 90 passing tests in the supplied environment;
- explicit enforcement of $h\le r_\pi$;
- a scalar mean-$R^2$ implementation of `score()`;
- affine/core export utilities that may be retained in a secondary, documented module after the primary estimator is settled;
- clean separation between the model and demonstrations.

The following parts should not enter the stable paper estimator:

- `response_subspace="var"` and `response_subspace="lstsq"`, because they are not the Pi-PLS method defined in the manuscript;
- allowing $h=0$ as a standard fitted model;
- zero-padding components for $h>q$;
- naming $\mathbf{P}$ and $\mathbf{Q}$ as PLS loadings unless actual loadings are computed by their standard definitions;
- an approximate `inverse_transform()` presented as an exact inverse;
- stale references to a missing `blocks.py` and `test_blocks.py`.

### 2.3 Problems in Vishal's current estimator and CV layer

The mathematical core is suitable, but the current public wrapper needs replacement or substantial refactoring:

- `score()` returns `{"mse": ..., "r2": ...}`. A scikit-learn regressor's `score()` must return one scalar; the conventional default is $R^2$.
- `coef_` is stored with shape $(p,q)$. Current scikit-learn `PLSRegression` exposes `coef_` with shape $(q,p)$ and predicts with `X @ coef_.T + intercept_`.
- `predict()` does not use fitted-feature validation through `validate_data(..., reset=False)`.
- `scale=False` performs no centering, although the manuscript core assumes centered $\mathbf{X}$ and $\mathbf{Y}$. To match `PLSRegression`, `scale=False` should mean “center but do not divide by standard deviations.”
- $h\le r_\pi$ and $r_\pi\le\min(n,p)$ are not consistently enforced.
- requested component counts are silently clamped. Invalid model specifications should normally raise an error so that failed grid points are visible.
- the estimator stores full training matrices only to support plotting. Plotting functions should accept data explicitly rather than requiring every fitted estimator to retain the training set.
- the custom CV framework catches broad exceptions and skips failed folds, which can conceal invalid parameter combinations.
- the CV documentation says errors are reported in original units, while the implementation computes MSE after fold-specific target scaling.
- the leave-one-out implementation combines targets transformed by different fold-specific scalers. Those values do not share one common coordinate system.
- the current `PiPLSRegressionCV` scans $h$ for one fixed $r_\pi$; it does not implement the triangular two-parameter scan requested for the public workflow.
- the data-dependent `variance` rule is resolved on the full dataset before CV, which introduces information from validation folds into parameter determination.

The existing CV code is still valuable as a behavioral reference for fixed-$r_\pi$ experiments. Its tested results should be captured in regression fixtures before it is replaced by the scikit-learn-native implementation.

### 2.4 Release-blocking specification inconsistencies

There are three rank-rule statements that are not currently identical:

1. The manuscript states a one-in-ten rule, with a one-in-five rule for $n\le 50$, and reports $r_\pi=10$ for Pulp at $n=46$.
2. Vishal's code uses `ceil(n / 10)` for $n>60$ and `ceil(n / 5)` otherwise.
3. The requested workflow originally referred to `round(n / c)` and treats this quantity as an upper bound for a scan.

The paper's Pulp value rules out ordinary rounding: $\operatorname{round}(46/5)=9$, whereas $\lceil46/5\rceil=10$. Revision 4 resolves this at two levels:

- the general API uses the ceiling operation with $n_{\mathrm{train,min}}$ and a numeric user-supplied `samples_per_predictor_rank`;
- the paper-reproduction scripts preserve the manuscript's current one-in-five rule for $n\le50$ and one-in-ten rule otherwise, together with the currently reported full-$n$ rank calculation where required for exact reproduction.

The paper-specific rule is therefore explicit orchestration metadata, not an adaptive hidden default in `PiPLSRegression`. Vishal's $n\le60$ code branch remains a historical implementation discrepancy to be tested and documented, not copied into the public API.

The public naming and workflow are now settled at the architectural level:

```text
theoretical h       -> n_components
theoretical r_pi    -> predictor_rank
theoretical c       -> samples_per_predictor_rank
selected r_pi       -> predictor_rank_
rule-derived limit  -> max_predictor_rank_
```

The symbols $h$, $r_\pi$, and $c$ should remain in the manuscript, equations, private core routine, and paper-reproduction output. They should not be public constructor aliases. The descriptive names are less ambiguous in scikit-learn parameter grids and avoid using “component” for both $h$ and the predictor truncation rank.

Three distinct rank procedures must remain explicit:

- **automatic bounded scan:** `predictor_rank="auto"`; search all admissible $r_\pi$ up to the rule-derived upper bound for the specified `n_components`;
- **rule-fixed rank:** `predictor_rank="max"`; set $r_\pi$ equal to the rule-derived upper bound without scanning it;
- **explicit fixed rank:** `predictor_rank=<int>`; use a user-specified value.

The automatic bounded scan is the normal estimator behavior. The other two are specialist alternatives. A full diagnostic sweep up to the fold-admissible matrix rank belongs to `PiPLSPathCV`, not to the default estimator.

## 3. Scope of version 0.1.0

### Included

- standard Pi-PLS core exactly as defined in `pipls.tex`;
- a `PiPLSRegression` estimator whose minimal call mirrors ordinary `PLSRegression` use;
- `n_components` as the public name for $h$;
- `predictor_rank="auto"` as the normal conditional rank-selection mode;
- `samples_per_predictor_rank` as the public control of the admissible rank bound;
- rule-fixed and explicitly fixed predictor-rank alternatives;
- a two-parameter `PiPLSPathCV` search and diagnostic utility;
- standard scikit-learn CV splitters and scoring, including an explicit `LeaveOneOut()` paper protocol;
- response-standardized MSE as the default automatic-selection and path-analysis objective;
- LOO errors reported as the paper's performance results, with their selection-conditioned status stated explicitly;
- fold-local built-in centering and scaling;
- pipeline-aware model selection for arbitrary preprocessing;
- common-format real datasets and deterministic converters;
- synthetic generators and fixed reproducibility seeds;
- examples and scripts reproducing all paper figures and reported tables;
- unit, invariant, estimator-contract, integration, dataset, and paper-regression tests.

### Explicitly excluded

- block scaling as a fitted method;
- alternative response-subspace definitions;
- nonlinear Pi-PLS variants;
- undocumented rank heuristics based on full-data singular values;
- public aliases named `h`, `r_pi`, or `c`;
- arbitrary preprocessing objects embedded as constructor parameters of `PiPLSRegression`;
- dataset-specific plotting functions inside the estimator;
- generated result files, pickles, Python bytecode, editor backups, `.DS_Store`, AppleDouble `._*` files, embedded Git repositories, and compiled documentation artifacts.

The excluded research variants can remain in a private development branch or a separate experimental repository until their theory and validation are ready for the follow-up paper.

## 4. Proposed repository layout

```text
pipls/
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
├── pyproject.toml
├── Makefile
├── MANIFEST.in                       # only when needed for sdist contents
├── .gitignore
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       ├── tests.yml
│       ├── build.yml
│       └── data-validation.yml
│
├── .llm/
│   ├── README.md
│   ├── project.md
│   ├── mathematics.md
│   ├── numerical_contracts.md
│   ├── development.md
│   ├── public_api.md
│   ├── snapshot.sh
│   ├── create_patch.sh
│   ├── strategy.md
│   ├── prompts/
│   │   ├── bugfix.md
│   │   ├── feature.md
│   │   ├── refactor.md
│   │   ├── mathematical-change.md
│   │   └── numerical-stability.md
│   └── templates/
│       ├── PATCH_REQUEST.md
│       ├── TEST_PROTOCOL.md
│       └── REVIEW_CHECKLIST.md
│
├── src/
│   └── pipls/
│       ├── __init__.py
│       ├── _core.py                  # private, theory-faithful numerical core
│       ├── _validation.py
│       ├── regression.py             # PiPLSRegression
│       ├── model_selection.py        # PiPLSPathCV and grid builders
│       ├── metrics.py                # explicit MSE/RMSE/R2 helpers and scorers
│       ├── datasets.py               # common-schema loader; no converters
│       ├── plotting.py               # optional plotting functions
│       └── preprocessing.py          # reserved public namespace for future transformers
│
├── tests/
│   ├── unit/
│   │   ├── test_core.py
│   │   ├── test_validation.py
│   │   ├── test_regression.py
│   │   ├── test_model_selection.py
│   │   └── test_metrics.py
│   ├── invariants/
│   │   ├── test_orthogonality.py
│   │   ├── test_prediction_factorization.py
│   │   └── test_affine_equivalence.py
│   ├── estimator_checks/
│   │   └── test_sklearn_checks.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   ├── test_grid_search.py
│   │   ├── test_internal_scaling.py
│   │   ├── test_preprocessing_leakage.py
│   │   ├── test_target_transform.py
│   │   ├── test_leave_one_out.py
│   │   └── test_official_splits.py
│   ├── data/
│   │   ├── test_schema.py
│   │   ├── test_checksums.py
│   │   └── test_dataset_shapes.py
│   ├── regression/
│   │   ├── test_reference_core.py
│   │   ├── test_corn_results.py
│   │   └── test_pulp_results.py
│   └── conftest.py
│
├── datasets/
│   ├── README.md
│   ├── registry.yaml
│   ├── schema/
│   │   └── dataset-metadata.schema.json
│   ├── corn/
│   │   ├── metadata.yaml
│   │   ├── X.csv
│   │   ├── Y.csv
│   │   ├── samples.csv               # optional sample metadata
│   │   └── checksums.sha256
│   ├── pulp/
│   │   └── ...
│   ├── tobacco/
│   │   └── ...
│   ├── sugarcane/
│   │   └── ...
│   ├── sarcos/
│   │   ├── metadata.yaml
│   │   ├── X.csv
│   │   ├── Y.csv
│   │   ├── splits.csv                # official train/test role
│   │   └── checksums.sha256
│   ├── fredmd/
│   │   └── ...
│   ├── steel/
│   │   └── ...
│   └── synthetic_reference/
│       ├── metadata.yaml
│       ├── X.csv
│       ├── Y.csv
│       └── checksums.sha256
│
├── scripts/
│   ├── prepare_data/
│   │   ├── common.py
│   │   ├── prepare_corn.py
│   │   ├── prepare_pulp.py
│   │   ├── prepare_tobacco.py
│   │   ├── prepare_sugarcane.py
│   │   ├── prepare_sarcos.py
│   │   ├── prepare_fredmd.py
│   │   ├── prepare_steel.py
│   │   └── prepare_synthetic_reference.py
│   ├── validate_datasets.py
│   └── reproduce_paper/
│       ├── common.py
│       ├── synthetic_studies.py
│       ├── corn_analysis.py
│       ├── pulp_analysis.py
│       ├── predictor_rank_sweeps.py
│       └── build_all.py
│
├── examples/
│   ├── 01_pls_style_estimator.py
│   ├── 02_automatic_predictor_rank.py
│   ├── 03_two_parameter_path.py
│   ├── 04_rule_fixed_predictor_rank.py
│   ├── 05_explicit_fixed_predictor_rank.py
│   ├── 06_custom_preprocessing_path.py
│   ├── 07_custom_cv_splitter.py
│   ├── 08_leave_one_out.py
│   └── 09_official_train_test_split.py
│
├── docs/
│   ├── index.md
│   ├── theory.md
│   ├── estimator_api.md
│   ├── parameter_selection.md
│   ├── preprocessing.md
│   ├── datasets.md
│   ├── reproducibility.md
│   └── decisions/
│       ├── 0001-core-definition.md
│       ├── 0002-preprocessing-semantics.md
│       ├── 0003-predictor-rank-selection.md
│       ├── 0004-response-standardized-mse.md
│       ├── 0005-leave-one-out-protocol.md
│       └── 0006-paper-versus-api-rank-rule.md
│
└── paper/
    ├── pipls.tex
    ├── references.bib
    ├── reproduction-manifest.yaml
    └── README.md
```

The `datasets/` directory should not be included in the built Python wheel. It belongs to the research repository and release archive. The package itself should depend only on what is required for estimation. Dataset conversion dependencies belong in an optional dependency group.

## 5. LLM-assisted development layer

The `.llm/` directory is part of the tracked repository but is not part of the installable package. Its purpose is to make tarball-based review and root-relative patch exchange reproducible across separate LLM sessions and human contributors. It must not be imported by `src/pipls`, included as runtime package data, or treated as an alternative source of truth to the tested implementation and public documentation.

### 5.1 Authoritative files

The following files should have distinct responsibilities:

- `.llm/strategy.md`: LLM-maintained operational phase plan, next increment, acceptance conditions, ownership boundaries, and maintenance protocol;
- `.llm/project.md`: concise repository map, public entry points, module responsibilities, test locations, generated-file boundaries, and the preferred commands for validation;
- `.llm/mathematics.md`: notation, dimensions, defining equations, model identities, admissibility conditions, equivalences to established methods, and the distinction between mathematical identities and implementation conventions;
- `.llm/numerical_contracts.md`: policies for SVD and eigenproblems, rank tolerances, pseudoinverses, degeneracy, repeated eigenvalues, projection-based comparisons, sign indeterminacy, finite-value checks, and avoidance of unnecessarily large matrices;
- `.llm/development.md`: scikit-learn estimator conventions, test requirements, dependency policy, documentation obligations, patch format, generated-file exclusions, and validation reporting;
- `.llm/public_api.md`: constructor parameters, fitted attributes, array shapes, orientation conventions, compatibility commitments, and names intentionally excluded from the public API;
- `.llm/README.md`: how the files fit together and which document is authoritative when guidance overlaps.

The mathematical file should include a dimension table for at least $\mathbf{X}_{\mathrm{cs}}$, $\mathbf{Y}_{\mathrm{cs}}$, $\mathbf{\Pi}$, $\mathbf{Z}$, $\mathbf{C}$, $\mathbf{W}$, $\mathbf{P}$, $\mathbf{D}$, and $\mathbf{Q}$. It should state explicitly that repeated or nearly repeated eigenvalues identify an invariant subspace rather than intrinsically numbered vectors. Tests of such objects must compare projection matrices, principal angles, singular values, regression maps, or predictions rather than raw basis vectors.

### 5.2 Stable command surface

A root `Makefile` should expose a small set of commands used by humans, CI, and LLM-assisted work:

```text
make test
make lint
make typecheck
make docs
make build
make snapshot
```

These commands should delegate to the same tools used in CI. The `.llm` instructions should refer to these targets rather than duplicating long command lines.

### 5.3 Snapshot contract

`.llm/snapshot.sh` should create a deterministic repository tarball suitable for upload. The archive should contain the contents of the repository root, including `.llm`, but no enclosing project-name directory. This keeps archive member paths identical to root-relative Git paths and permits extraction directly into an existing project directory. It should exclude at least:

```text
.git/
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
coverage.xml
build/
dist/
*.egg-info/
docs/_build/
notebooks/.ipynb_checkpoints/
generated paper figures and tables
editor backups and operating-system metadata
```

The script should fail on an unexpected repository state rather than silently producing an ambiguous archive. It should generate snapshot metadata inside the archive without modifying tracked files. The metadata should record the project name, commit hash, branch, dirty-worktree status, snapshot time, Python version, and package version when available.

The snapshot script is the canonical way to prepare an LLM upload. CI should include a lightweight archive-content test that rejects caches, embedded Git repositories, compiled Python files, generated documentation, and other forbidden artifacts.

### 5.4 Patch contract

LLM-produced changes should be returned as a unified Git patch relative to the repository root. Paths must have the form:

```diff
diff --git a/src/pipls/regression.py b/src/pipls/regression.py
```

They must not contain absolute paths, temporary extraction prefixes, or archive-specific directory names.

Patch application and commits should use direct Git commands from the repository root rather than project wrapper scripts. The documented sequence is:

```bash
git status --short
git apply --check ~/Downloads/proposed-change.patch
git apply ~/Downloads/proposed-change.patch
make check
git diff
git add -A
git diff --cached --check
git diff --cached
git commit -m "Describe the completed increment"
```

The worktree should normally be clean before application. `git apply --check` must succeed before `git apply` is run. The user reviews both the unstaged and staged diffs; no project script applies, stages, or commits changes implicitly. `.llm/create_patch.sh` may remain as an optional convenience for exporting local changes, but it is not required for accepting an LLM-produced patch.

### 5.5 Request and review templates

The prompt templates should distinguish ordinary feature work from mathematical and numerical changes.

A mathematical-change request should require:

- the exact mathematical object being changed;
- all affected dimensions;
- identities and invariants that must remain true;
- whether equality is basis-dependent or only subspace-dependent;
- references or derivations supporting the change;
- focused tests of the defining equations;
- documentation changes in both `.llm/mathematics.md` and user-facing theory documentation when appropriate.

A numerical-stability request should require:

- the conditioning or rank-deficiency problem;
- the tolerance definition and its scale dependence;
- expected behavior for constant, singular, nearly singular, and repeated-eigenvalue cases;
- avoidance of explicit inverses where a solve, SVD, or pseudoinverse is appropriate;
- deterministic tests near the numerical boundary.

Every completed patch review should report validation in a fixed form, for example:

```text
Validation:
- `make test`: passed
- `make lint`: passed
- `make typecheck`: not run; mypy unavailable
- `make docs`: passed
```

“Validated” must never mean only that the patch was inspected.

### 5.6 Normal workflow

The intended workflow is:

1. produce the upload with `make snapshot`;
2. upload the tarball with a request based on `.llm/templates/PATCH_REQUEST.md`;
3. require inspection of the relevant `.llm` contracts before source modification;
4. receive one root-relative unified patch plus a concise validation report;
5. save the patch locally and run `git apply --check proposed-change.patch` followed by `git apply proposed-change.patch`;
6. run the Makefile validation targets and inspect `git diff`;
7. stage with `git add -A`, inspect `git diff --cached`, and commit with `git commit -m "MESSAGE"`;
8. create the next clean snapshot only after the commit succeeds.

The `.llm` layer standardizes communication; it does not replace code review, tests, Git history, release notes, or scientific review.

## 6. Package API

### 6.1 Top-level exports

```python
from pipls import PiPLSPathCV, PiPLSRegression
from pipls.datasets import load_dataset
from pipls.metrics import (
    neg_response_standardized_mean_squared_error,
    response_standardized_mean_squared_error,
)
```

Only the two principal classes should be imported at `pipls` top level in version 0.1.0. Low-level numerical functions should live in `pipls._core` and be treated as private until their direct API is intentionally stabilized.

### 6.2 Public naming contract

Use descriptive scikit-learn-style names in the public API and preserve the manuscript notation internally:

```text
Public name                         Mathematical notation
----------------------------------------------------------------
n_components                       h
predictor_rank                      r_pi
samples_per_predictor_rank          c
predictor_rank_                     selected r_pi
max_predictor_rank_                 rule-derived upper bound
```

`predictor_rank` is preferred to `retained_predictor_matrix_rank`: it is shorter and does not suggest that the value is the numerical rank of the original input matrix. `samples_per_predictor_rank` is preferred to `observations_per_predictor_component`: scikit-learn uses “samples,” and “component” is already reserved for `n_components`.

Do not expose `h`, `r_pi`, or `c` as aliases. Aliases would duplicate parameter-grid names, complicate `get_params()` and `set_params()`, and create unnecessary compatibility obligations before the first public release.

### 6.3 Base estimator

Recommended constructor:

```python
PiPLSRegression(
    n_components=2,
    *,
    scale=True,
    copy=True,
    predictor_rank="auto",
    samples_per_predictor_rank=10,
    cv=5,
    scoring="neg_response_standardized_mean_squared_error",
    n_jobs=None,
)
```

The minimal constructor deliberately resembles `PLSRegression(n_components=2)`. Pi-PLS-specific controls are optional and keyword-only.

Semantics:

- `n_components` is the public name for the theoretical parameter $h$.
- `predictor_rank="auto"` searches the admissible predictor ranks for the specified `n_components` and selects by cross-validation.
- `predictor_rank="max"` uses the rule-derived upper rank directly, without scanning predictor rank.
- `predictor_rank=<int>` fixes the predictor rank explicitly.
- `samples_per_predictor_rank` is the public name for the theoretical rule parameter $c$.
- `cv`, `scoring`, and `n_jobs` control the internal search only when `predictor_rank="auto"`. `cv` should accept an integer, a scikit-learn splitter object such as `LeaveOneOut()`, or an iterable of `(train, test)` index pairs. The default scorer identifier resolves to the package's response-standardized MSE scorer; users may instead supply any compatible scikit-learn scorer callable or supported scorer name.
- `scale=True` centers and standardizes both $X$ and $y$.
- `scale=False` centers both $X$ and $y$ but does not divide by standard deviations. This mirrors the conceptual behavior of scikit-learn's `PLSRegression`.
- all constructor arguments are stored unchanged; validation and data-dependent resolution occur in `fit()`.
- `max_iter` and `tol` should not be copied from `PLSRegression` unless Pi-PLS actually uses a corresponding iterative algorithm.

For automatic selection, define

\begin{equation}
r_{\pi,\max}
=
\min\left[
p,n_{\mathrm{train,min}},
\left\lceil
\frac{n_{\mathrm{train,min}}}
{\texttt{samples\_per\_predictor\_rank}}
\right\rceil
\right].
\end{equation}

This is now the fixed public-API rule. The estimator derives $n_{\mathrm{train,min}}$ from the actual materialized splits before constructing the candidate ranks. It never substitutes the complete development-set size. The paper-reproduction layer may preserve the manuscript's existing full-$n$ rank grid by passing explicit limits or values; the estimator itself must not infer the paper's small-sample threshold or alter a user-supplied numeric `samples_per_predictor_rank`. The implementation must also enforce

\begin{equation}
1
\le
\texttt{n\_components}
\le
\min(\texttt{predictor\_rank\_},q),
\end{equation}

and

\begin{equation}
\texttt{n\_components}
\le
\texttt{predictor\_rank\_}
\le
\min(n,p).
\end{equation}

Do not silently clamp invalid values. Raise a `ValueError` that identifies the violated inequality.

### 6.4 Default PLS-style use

The principal introductory example should contain no explicit predictor-rank argument:

```python
from pipls import PiPLSRegression

model = PiPLSRegression(n_components=3)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
r2 = model.score(X_test, Y_test)

print(model.predictor_rank_)
```

For comparison, an ordinary PLS analysis uses the same top-level pattern:

```python
from sklearn.cross_decomposition import PLSRegression

model = PLSRegression(n_components=3)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
```

The extra Pi-PLS rank exists mathematically, but it does not intrude into the normal constructor call.

### 6.5 Automatic, rule-fixed, and explicit rank modes

Normal conditional selection:

```python
model = PiPLSRegression(
    n_components=3,
    predictor_rank="auto",
    samples_per_predictor_rank=10,
)
```

For the specified $h$, the estimator selects

\begin{equation}
r_\pi^*(h)
=
\operatorname*{arg\,min}_{r_\pi\in\{h,\ldots,r_{\pi,\max}\}}
\widehat{\operatorname{MSE}}_{\mathrm{CV,response\text{-}std}}(h,r_\pi).
\end{equation}

Rule-fixed rank:

```python
model = PiPLSRegression(
    n_components=3,
    predictor_rank="max",
    samples_per_predictor_rank=10,
)
```

Explicit fixed rank:

```python
model = PiPLSRegression(
    n_components=3,
    predictor_rank=8,
)
```

The integer form is an advanced interface for paper reproduction, simulations, unit tests, known-model refits, and external parameter searches. It should not be used in the first example.

### 6.6 Fitted attributes

Recommended fitted attributes:

```text
n_features_in_
feature_names_in_                    # when input names are available
n_targets_
predictor_rank_
max_predictor_rank_
predictor_rank_values_               # automatic mode
predictor_rank_cv_results_            # automatic mode
best_response_standardized_mse_         # positive loss, automatic mode
x_mean_, x_scale_
y_mean_, y_scale_
Pi_                                  # p x predictor_rank_
C_                                   # q x n_components
W_                                   # predictor_rank_ x n_components
P_                                   # p x n_components
D_                                   # n_components x n_components
dilation_                            # n_components-vector convenience view
Q_                                   # q x n_components
x_rotations_                         # compatibility alias for P_
y_rotations_                         # compatibility alias for Q_
x_scores_
y_scores_
coef_                                # q x p, scikit-learn convention
coef_matrix_                         # p x q, manuscript convention
intercept_                           # q-vector
```

The constructor parameter `n_components` remains available unchanged through `get_params()`. A separate fitted `n_components_` is unnecessary unless fitting can resolve it to a different value, which is not recommended.

Do not define `x_loadings_` or `y_loadings_` as aliases for $\mathbf{P}$ and $\mathbf{Q}$. Either compute loadings by an explicitly documented definition or omit those attributes in version 0.1.0.

`score(X, y)` should return scalar $R^2$, with uniform averaging across responses. A separate `evaluate_regression()` utility may return a dictionary of MSE, RMSE, and $R^2$ values.

`transform(X, y=None)` should follow the PLS-style behavior:

- return predictor scores when only `X` is supplied;
- return `(x_scores, y_scores)` when both are supplied;
- use the fitted centering/scaling parameters;
- validate the number and order of input features.

Omit `inverse_transform()` from version 0.1.0 unless an actual reconstruction definition is implemented and tested. It is not required for regression compatibility.

### 6.7 Built-in standardization

`scale` should provide the same simple control expected by users of `PLSRegression`:

```python
model = PiPLSRegression(
    n_components=3,
    scale=True,
)
```

The implementation must fit centering and scaling separately inside every internal predictor-rank CV training fold. A validation observation must never contribute to the means or standard deviations used to fit its candidate model.

Conceptually, each internal split should perform:

```text
1. Fit X and y centering/scaling on the inner training fold.
2. Transform the inner training data.
3. Fit the fixed-(n_components, predictor_rank) Pi-PLS core.
4. Transform the inner validation predictors with training-fold statistics.
5. Predict and return responses to the coordinate system required by the scorer.
6. Compute the validation criterion.
```

After selecting `predictor_rank_`, refit the scaling and Pi-PLS model on all data supplied to `fit()`.

This built-in `scale` route is the normal API. It requires no lower-level access.

### 6.8 General preprocessing and leakage boundary

Arbitrary preprocessing should use scikit-learn composition rather than additional constructor branches in `PiPLSRegression`. Examples include robust scaling, derivatives, filtering, imputation, block scaling, column selection, and target transformations.

A pipeline such as

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pipls import PiPLSRegression

pipeline = Pipeline([
    ("preprocess", StandardScaler()),
    ("regression", PiPLSRegression(
        n_components=1,
        predictor_rank=1,
        scale=False,
    )),
])
```

is a suitable fixed-model template. However, it is not statistically valid to replace `predictor_rank=1` with `predictor_rank="auto"` and let the final estimator conduct an internal search after the preceding transformer has been fitted on all data passed to `pipeline.fit()`. The inner validation folds would then be transformed using preprocessing parameters influenced by those same validation observations.

For custom preprocessing, rank selection must therefore be performed by `PiPLSPathCV` around the complete pipeline:

```python
from pipls import PiPLSPathCV

search = PiPLSPathCV(
    estimator=pipeline,
    samples_per_predictor_rank=10,
    cv=10,
)
search.fit(X, Y)
```

`PiPLSPathCV` should clone and fit the entire pipeline for every fold and candidate pair. It should detect the final `PiPLSRegression` step and set nested parameters such as:

```python
{
    "regression__n_components": h,
    "regression__predictor_rank": predictor_rank,
}
```

This is a higher-level, pipeline-aware API, not a lower-level numerical API.

Target transformations beyond built-in centering and scaling should be included in the CV-controlled composite estimator, for example with `TransformedTargetRegressor`. They must not be fitted once before predictor-rank selection. For a deeply nested estimator, `PiPLSPathCV` should use an explicit `pipls_param_prefix`, such as `regressor__regression`, to set the Pi-PLS parameters.

### 6.9 External search over `n_components`

Existing PLS model-selection code should require only a small change:

```python
from sklearn.model_selection import GridSearchCV
from pipls import PiPLSRegression
from pipls.metrics import neg_response_standardized_mean_squared_error

search = GridSearchCV(
    PiPLSRegression(),
    param_grid={
        "n_components": range(1, 11),
    },
    scoring=neg_response_standardized_mean_squared_error,
    cv=10,
)

search.fit(X, Y)
```

For every candidate `n_components`, the estimator profiles `predictor_rank` internally. When this search is itself used inside an outer performance-estimation split, rank selection remains inside the outer training data.

For complete inspection of the triangular error surface, or for custom preprocessing, use `PiPLSPathCV` instead.

### 6.10 Future block scaling

The future extension should be a transformer in `pipls.preprocessing`, not a parameter branch inside the regression algorithm:

```python
pipeline = Pipeline([
    ("block_scale", BlockScaler(
        blocks=block_schema,
        method="...",
    )),
    ("regression", PiPLSRegression(
        n_components=1,
        predictor_rank=1,
        scale=False,
    )),
])

search = PiPLSPathCV(
    estimator=pipeline,
    samples_per_predictor_rank=10,
    cv=10,
)
```

Design constraints for future `BlockScaler`:

- implement `fit`, `transform`, and `get_feature_names_out`;
- learn all scale factors from training folds only;
- accept column names, index arrays, or slices as block definitions;
- preserve deterministic output-column order;
- store fitted block statistics with trailing underscores;
- remain independent of Pi-PLS, so it can also precede PLS, ridge regression, or other estimators;
- carry its own tests and paper citation when introduced.

No placeholder block classes are required in version 0.1.0. Reserving the module and maintaining transformer compatibility is sufficient preparation.

## 7. Two-parameter model selection

### 7.1 Roles of the two public classes

`PiPLSRegression` and `PiPLSPathCV` should serve different but consistent purposes:

- `PiPLSRegression(n_components=h)` is the ordinary PLS-style estimator. It selects `predictor_rank` conditionally for that one value of `n_components`.
- `PiPLSPathCV` evaluates and stores the complete triangular surface over `n_components` and `predictor_rank`.
- `PiPLSPathCV` is also the required route when arbitrary learned preprocessing must be refitted inside every rank-selection fold.

Both layers must use the same private fixed-parameter numerical kernel and the same admissibility checks.

### 7.2 Admissible grid

Let $n_{\mathrm{train,min}}$ be the smallest training-fold size generated by the chosen CV splitter. For an upper predictor rank $r_{\pi,\max}$, define

\begin{equation}
\mathcal{G}
=
\{(h,r_\pi):
1\le h\le h_{\max},
\quad
h\le r_\pi\le r_{\pi,\max}\},
\end{equation}

where

\begin{equation}
h_{\max}
=
\min(q,r_{\pi,\max}),
\end{equation}

and

\begin{equation}
r_{\pi,\max}
\le
\min(p,n_{\mathrm{train,min}}).
\end{equation}

This fold-aware bound prevents a candidate that is valid on the complete dataset but invalid in a smaller training fold.

For the normal bounded scan, use

\begin{equation}
r_{\pi,\max}
=
\min\left[
p,n_{\mathrm{train,min}},
\left\lceil
\frac{n_{\mathrm{train,min}}}
{\texttt{samples\_per\_predictor\_rank}}
\right\rceil
\right].
\end{equation}

This fold-safe formula is fixed for ordinary and nested CV. The splitter is materialized first, $n_{\mathrm{train,min}}$ is computed from those splits, and one common candidate grid is then reused for all folds and parameter pairs. Any paper-specific switch from 10 to 5 samples per predictor rank, or any full-$n$ rank limit retained for exact manuscript reproduction, must be encoded explicitly by the reproduction script.

### 7.3 Conditional predictor-rank selection

For every fixed $h$, compute

\begin{equation}
r_\pi^*(h)
=
\operatorname*{arg\,min}_{r_\pi:(h,r_\pi)\in\mathcal{G}}
\widehat{\operatorname{MSE}}_{\mathrm{CV,response\text{-}std}}(h,r_\pi).
\end{equation}

Store the complete error surface and the conditional path

\begin{equation}
\{[h,r_\pi^*(h),
\widehat{\operatorname{MSE}}_{\mathrm{CV,response\text{-}std}}(h,r_\pi^*(h))]\}_{h=1}^{h_{\max}}.
\end{equation}

Tie-breaking should be deterministic:

1. lower mean CV-MSE;
2. smaller predictor rank when scores are equal within a stated tolerance;
3. for the global best pair, smaller `n_components`, then smaller predictor rank.

An optional one-standard-error rule can be added later, but it should not be the default unless it is used in the paper.

### 7.4 `PiPLSPathCV` interface

Recommended constructor:

```python
PiPLSPathCV(
    estimator=None,
    *,
    pipls_param_prefix=None,
    n_components_values=None,
    predictor_rank_values=None,
    max_predictor_rank="rule",
    samples_per_predictor_rank=10,
    cv=None,
    scoring="neg_response_standardized_mean_squared_error",
    refit=True,
    n_jobs=None,
    error_score="raise",
    return_train_score=False,
    return_oof_predictions=False,
)
```

Recommended behavior:

- default `estimator` is a fixed-parameter `PiPLSRegression` template with built-in scaling;
- if `estimator` is a `Pipeline` whose final step is a unique `PiPLSRegression`, infer the nested parameter prefix from that step;
- for deeper composites, such as a `TransformedTargetRegressor` containing a pipeline, accept `pipls_param_prefix` using standard double-underscore parameter notation;
- raise a clear error when the Pi-PLS parameter location is ambiguous;
- materialize one set of CV splits and reuse it for every parameter pair;
- construct a list of one-point parameter dictionaries or an equivalent triangular grid so that combinations with `predictor_rank < n_components` are never fitted;
- set `predictor_rank` to an integer for every candidate, thereby avoiding nested automatic rank selection inside the path search;
- clone and fit the complete estimator or pipeline within each fold;
- use scikit-learn scorer conventions; the package default is negative response-standardized MSE, with `best_score_` retaining the negative scorer orientation and positive-loss convenience attributes exposed separately;
- expose the standard `cv_results_`, `best_params_`, `best_score_`, and `best_estimator_` attributes;
- additionally expose `best_predictor_rank_by_n_components_`, `best_score_by_n_components_`, `response_standardized_mse_path_`, `best_response_standardized_mse_`, `n_components_values_`, and `predictor_rank_values_`;
- when `return_oof_predictions=True`, expose `oof_predictions_` for the selected parameterization in original sample order and document that the resulting paper performance is selection-conditioned;
- refit the globally selected pair on the complete input when `refit=True`;
- provide `predict`, `transform`, and `score` only when a fitted `best_estimator_` exists;
- accept relevant split metadata, such as groups, through `fit`;
- never suppress failed grid points silently.

A list of dictionaries is supported directly by `GridSearchCV`, so the implementation can remain close to scikit-learn rather than rebuilding the complete CV engine.

### 7.5 Standard estimator workflow

For one specified number of components:

```python
from pipls import PiPLSRegression

model = PiPLSRegression(
    n_components=3,
    samples_per_predictor_rank=10,
)
model.fit(X, Y)

print(model.predictor_rank_)
Y_pred = model.predict(X_new)
```

This is the default user workflow.

### 7.6 Complete path workflow

```python
from sklearn.model_selection import RepeatedKFold
from pipls import PiPLSPathCV

cv = RepeatedKFold(
    n_splits=5,
    n_repeats=20,
    random_state=2026,
)

search = PiPLSPathCV(
    samples_per_predictor_rank=10,
    cv=cv,
    scoring="neg_response_standardized_mean_squared_error",
    n_jobs=-1,
)
search.fit(X, Y)

print(search.best_predictor_rank_by_n_components_)
print(search.best_params_)
Y_pred = search.predict(X_new)
```

This is the preferred scientific diagnostic when the analyst wants the complete response-standardized CV-MSE surface and the profiled curve versus `n_components`.

### 7.7 Rule-fixed and explicit alternatives

Rule-fixed predictor rank:

```python
model = PiPLSRegression(
    n_components=3,
    predictor_rank="max",
    samples_per_predictor_rank=10,
)
```

Explicitly fixed predictor rank:

```python
model = PiPLSRegression(
    n_components=3,
    predictor_rank=8,
)
```

A helper may expose the exact rank-bound rule:

```python
from pipls.model_selection import predictor_rank_limit

rank_limit = predictor_rank_limit(
    n_samples=X.shape[0],
    n_features=X.shape[1],
    samples_per_predictor_rank=10,
)
```

The helper must implement one exact, documented ceiling-based rule for the numeric value supplied by the user. It must not switch between 5 and 10 through a hidden sample-size threshold.

### 7.8 Custom preprocessing workflow

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pipls import PiPLSPathCV, PiPLSRegression

pipeline = Pipeline([
    ("preprocess", StandardScaler()),
    ("regression", PiPLSRegression(
        n_components=1,
        predictor_rank=1,
        scale=False,
    )),
])

search = PiPLSPathCV(
    estimator=pipeline,
    samples_per_predictor_rank=10,
    cv=10,
)
search.fit(X, Y)
```

The complete pipeline must be cloned inside every CV fold. This is essential for future block scaling and for any learned preprocessing that would otherwise contaminate the predictor-rank validation folds.

### 7.9 Cross-validation and test-set policy

Training/test behavior belongs to splitters and orchestration scripts, not to the private numerical core.

Recommended patterns:

- small independent datasets: `LeaveOneOut` only when the paper protocol requires it;
- repeated random folds: `RepeatedKFold`;
- official train/test roles: `PredefinedSplit` or explicit train/test arrays;
- grouped measurements: `GroupKFold` or a related group-aware splitter;
- temporal data such as FRED-MD: a time-respecting split, normally `TimeSeriesSplit` or a documented fixed historical split;
- unbiased performance after tuning: nested CV or a held-out external test set.

The paper-reproduction scripts may implement the manuscript's sample-size-dependent protocol. The package API should accept a user-supplied splitter rather than hard-code that policy.

### 7.10 Leave-one-out protocol

Leave-one-out cross-validation should be represented by the ordinary scikit-learn splitter `LeaveOneOut()`. It must not be implemented as a Boolean flag or as a special branch in the private Pi-PLS core. This keeps the API compatible with other splitters and makes the training and validation indices explicit.

For automatic predictor-rank selection at one specified value of `n_components`, the paper-style call is:

```python
from sklearn.model_selection import LeaveOneOut
from pipls import PiPLSRegression

model = PiPLSRegression(
    n_components=3,
    cv=LeaveOneOut(),
)
model.fit(X, Y)

print(model.predictor_rank_)
```

For the complete triangular path over `n_components` and `predictor_rank`, use:

```python
from sklearn.model_selection import LeaveOneOut
from pipls import PiPLSPathCV
from pipls.metrics import neg_response_standardized_mean_squared_error

search = PiPLSPathCV(
    cv=LeaveOneOut(),
    scoring=neg_response_standardized_mean_squared_error,
    return_oof_predictions=True,
)
search.fit(X, Y)

print(search.best_predictor_rank_by_n_components_)
print(search.best_params_)
```

With $n$ observations, `LeaveOneOut()` creates $n$ splits. Every validation set contains one observation and every training set contains $n-1$ observations. The general fold-safe predictor-rank limit is therefore

\begin{equation}
r_{\pi,\max}^{\mathrm{LOO}}
=
\min\left[
p,n-1,
\left\lceil
\frac{n-1}{\texttt{samples\_per\_predictor\_rank}}
\right\rceil
\right].
\end{equation}

This is the fixed general API rule because every candidate is valid in every LOO training fold. The manuscript's current rank rule is preserved separately for paper reproduction. When exact reproduction requires a rank derived from the complete sample size $n$, the reproduction script passes an integer `max_predictor_rank` or explicit `predictor_rank_values`. The public estimator never hides that full-$n$ convention inside `LeaveOneOut()`.

The fixed selection and paper-performance criterion is response-standardized MSE. Foldwise $R^2$ is undefined for LOO because each validation fold contains only one response observation. Let $s_{(-i),j}$ be the scale of response $j$ estimated from the $n-1$ observations used to train the model that omits observation $i$. Then

\begin{equation}
\operatorname{MSE}_{\mathrm{LOO,response\text{-}std}}
=
\frac{1}{nq}
\sum_{i=1}^{n}
\sum_{j=1}^{q}
\left[
\frac{y_{ij}-\widehat{y}_{(-i),ij}}
{s_{(-i),j}}
\right]^2.
\end{equation}

The response scales are fold-specific and are never estimated from the held-out observation. Uniform averaging across responses is the default. Since all validation folds have equal size, averaging the singleton-fold response-standardized MSE values is identical to pooling all standardized held-out squared residuals.

A pooled diagnostic $R^2$ may be computed from the complete set of ordered out-of-fold predictions:

```python
from sklearn.metrics import r2_score

r2_loo = r2_score(
    Y,
    search.oof_predictions_,
    multioutput="uniform_average",
)
```

This quantity must be described as $R^2$ calculated from the pooled LOO predictions. It must not be called mean foldwise $R^2$. The optional `oof_predictions_` attribute should be produced only when explicitly requested, retain the original row order, and identify the parameterization that generated it.

Centering and scaling must be fitted independently on each set of $n-1$ training observations. The omitted observation must not contribute to predictor means, predictor scales, response means, response scales, the truncated predictor basis, or predictor-rank selection. Predictions are returned to original response units; the scorer then divides each residual by the response scale learned from the corresponding training fold. Transformed responses from different folds must never be concatenated and treated as if they share one standardized coordinate system. To mirror the built-in PLS scaling convention, response scales should use sample standard deviation with `ddof=1`; a zero training-fold scale is replaced by 1.0 and recorded so the loss remains finite and deterministic.

For custom learned preprocessing, `PiPLSPathCV` must wrap the complete pipeline and clone it inside every LOO fold:

```python
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pipls import PiPLSPathCV, PiPLSRegression
from pipls.metrics import neg_response_standardized_mean_squared_error

pipeline = Pipeline([
    ("preprocess", StandardScaler()),
    ("regression", PiPLSRegression(
        n_components=1,
        predictor_rank=1,
        scale=False,
    )),
])

search = PiPLSPathCV(
    estimator=pipeline,
    cv=LeaveOneOut(),
    scoring=neg_response_standardized_mean_squared_error,
)
search.fit(X, Y)
```

After selecting the parameterization, `refit=True` should fit the selected estimator once on all $n$ observations. This final refit is distinct from the LOO models used to estimate the selection criterion.

The paper protocol is now explicit: LOO is used both to select the Pi-PLS parameterization and to produce the reported performance result. The reported quantity is the minimum response-standardized LOO-MSE for the selected parameterization, together with the selected `n_components` and `predictor_rank`. Because the same LOO results drive selection and reporting, the repository must label this value as a **selection-conditioned LOO performance estimate**. It may be used as the paper's performance result, but it must not be described as an unbiased external-test or nested-CV estimate. General-purpose documentation should still recommend a held-out test set or nested CV when an unbiased post-selection estimate is required.

LOO is computationally expensive. For a grid containing $G$ admissible parameter pairs, the path analysis requires approximately $nG$ candidate fits, followed by one full-data refit when `refit=True`. The paper-reproduction manifest should record the splitter, explicit paper rank grid, scoring rule, preprocessing convention, selected parameterization, and that LOO is used for both tuning and the reported performance result.

## 8. Scoring and scale conventions

“CV-MSE” is insufficiently specific for a multivariate model. Revision 4 fixes the selection and paper-performance quantity as response-standardized MSE. The release must record the response-scale convention, uniform response weighting, split aggregation, and—where uncertainty is displayed—whether error bars describe split variability, repeat variability, or variability across synthetic realizations.

Fixed public conventions:

- `PiPLSRegression.score`: scalar $R^2$ in original units, uniform average over responses, preserving normal regressor behavior;
- automatic predictor-rank selection and `PiPLSPathCV`: negative response-standardized MSE by default;
- paper performance reporting: positive response-standardized LOO-MSE for the selected parameterization;
- multiresponse aggregation: uniform weighting across responses unless an explicit scorer callable implements documented weights.

For validation fold $k$, with training-fold response scales $s_{k,j}$ and validation index set $\mathcal{V}_k$, define

\begin{equation}
\operatorname{MSE}_{k,\mathrm{response\text{-}std}
}
=
\frac{1}{|\mathcal{V}_k|q}
\sum_{i\in\mathcal{V}_k}\sum_{j=1}^{q}
\left(
\frac{y_{ij}-\widehat{y}_{ij}}
{s_{k,j}}
\right)^2.
\end{equation}

Following scikit-learn split-score conventions, the CV score is the unweighted mean of the split scores unless the package explicitly documents another aggregation mode. Under LOO, all splits have one validation observation, so the split mean equals the pooled response-standardized MSE. The scorer callable should be named `neg_response_standardized_mean_squared_error`; the corresponding positive utility should be `response_standardized_mean_squared_error`.

Predictions remain in original response units. Only residuals are divided by the response scales learned from the matching training fold. This avoids combining targets transformed by different fold-specific scalers. Foldwise $R^2$ must not be requested for LOO; pooled out-of-fold $R^2$ may be reported only as a secondary diagnostic and must not replace the response-standardized MSE performance result.

## 9. Dataset standard

### 9.1 Definition of “unstandardized”

For this repository, an analysis-ready dataset is **unstandardized** when neither stored $X$ nor stored $Y$ has been centered or divided by a sample standard deviation as part of model fitting.

Domain preprocessing is allowed when it is intrinsic to the defined dataset, but it must be explicit and reproducible. Examples include:

- the first spectral derivative used for Corn;
- selection of wavelengths at or above 780 nm for Sugarcane;
- lag and lead construction for FRED-MD;
- exclusion of samples with missing reference measurements;
- selection of specified predictor and response columns.

The metadata should therefore distinguish:

```yaml
representation: analysis_ready_unscaled
centered: false
scaled: false
domain_preprocessing_applied: true
```

### 9.2 Required files

Each dataset directory must contain:

- `X.csv`: `sample_id` followed by numeric predictor columns;
- `Y.csv`: the same `sample_id` values, in the same order, followed by numeric response columns;
- `metadata.yaml`: source, licensing, schema, processing, dimensions, variables, and recommended splitting information;
- `checksums.sha256`: checksums for all committed dataset files.

Optional files:

- `samples.csv`: groups, dates, batches, source row identifiers, or other non-model sample metadata;
- `splits.csv`: named official roles such as `train`, `test`, or `calibration`;
- `features.csv`: extended feature descriptions when metadata would otherwise become unwieldy;
- `targets.csv`: extended target descriptions and units.

### 9.3 CSV rules

- UTF-8 encoding;
- comma delimiter and period decimal separator;
- one header row;
- unique, stable, machine-readable column names;
- `sample_id` as the first column in both files;
- no implicit pandas index;
- sufficient numeric precision for round-trip reconstruction, normally `%.17g` for floating-point values;
- no `NaN` or infinite values in released analysis matrices;
- identical sample IDs and row order in `X.csv` and `Y.csv`;
- units and display labels stored in metadata, not encoded only in column names.

### 9.4 Example `metadata.yaml`

```yaml
schema_version: 1
id: corn
title: Corn NIR composition dataset
dataset_version: 1.0.0

representation:
  stage: analysis_ready_unscaled
  centered: false
  scaled: false
  domain_preprocessing_applied: true

source:
  landing_page: "TO_BE_VERIFIED"
  files:
    - name: corn.mat
      sha256: "TO_BE_COMPUTED"
  retrieved: "YYYY-MM-DD"
  citation:
    text: "TO_BE_COMPLETED"
    doi: null
  license:
    name: "TO_BE_VERIFIED"
    url: null
    redistribution_permitted: null

samples:
  n: 80
  id_column: sample_id
  alignment: exact

predictors:
  file: X.csv
  n_features: 696
  dtype: float64
  description: first derivative of NIR absorbance spectrum
  units: null
  axis:
    kind: wavelength
    unit: nm
    minimum: 1104
    maximum: 2494
    step: 2

responses:
  file: Y.csv
  n_targets: 4
  columns:
    - name: moisture
      display_name: Moisture
      unit: "TO_BE_VERIFIED"
    - name: oil
      display_name: Oil
      unit: "TO_BE_VERIFIED"
    - name: protein
      display_name: Protein
      unit: "TO_BE_VERIFIED"
    - name: starch
      display_name: Starch
      unit: "TO_BE_VERIFIED"

processing:
  script: scripts/prepare_data/prepare_corn.py
  software:
    python: ">=3.10"
    scipy: "RECORDED_AT_BUILD"
  steps:
    - extract raw spectra and composition values from corn.mat
    - retain wavelengths 1104 through 2494 nm
    - apply the manuscript-specified first-derivative Savitzky-Golay treatment
    - do not center or standardize predictors
    - do not center or standardize responses
  random_seed: null

splitting:
  official_split: null
  recommended_cv: repeated_kfold
  groups_column: null
  time_column: null

integrity:
  x_sha256: "TO_BE_COMPUTED"
  y_sha256: "TO_BE_COMPUTED"
```

A JSON Schema should validate required keys, allowed values, and basic types. Dataset-specific scientific assertions remain in Python tests.

## 10. Dataset migration audit

| Dataset | Current analysis dimensions | Required conversion | Main issue before release |
|---|---:|---|---|
| Corn | $X:80\times696$, $Y:80\times4$ | regenerate from `corn.mat` | current `dataset_corn.mat` has standardized responses; the exact predictor conversion is not documented in code, while the manuscript specifies a first-derivative Savitzky-Golay treatment |
| Pulp | $X:46\times14$, $Y:46\times8$ | select documented columns from the 29-column source CSV | six leading source columns and the final `k` column are excluded by position; the scientific reason must be stated and column selection must use names |
| Tobacco | $X:347\times1557$, $Y:347\times13$ | merge workbooks by the shared sample ID | source, citation, license, units, translated display names, and redistribution rights are missing |
| Sugarcane | $X:57\times1721$, $Y:57\times4$ | merge 60 spectra with response data, remove incomplete rows, retain wavelengths $\ge780$ nm | the three excluded samples and wavelength rule must be recorded; source and license need verification |
| SARCOS | train $44484\times21/7$, test $4449\times21/7$ | combine into common X/Y with a split-role file | preserve the official split and source terms; do not replace it with random splitting in the principal example |
| FRED-MD | $X:386\times882$, $Y:386\times5$ | export lagged predictors and selected lead responses | the loader deletes the second response column by list position as “noisy”; response selection and temporal validation must be explicit |
| Steel | $X:267\times13$, $Y:267\times3$ | export compositions and three mechanical responses | Charpy is silently excluded from modeling; source, cleaning, units, and exclusion rationale must be documented |
| Synthetic | parameter dependent | retain generator plus fixed snapshot(s) and seeds | generator parameters, noise definition, random-number policy, and paper scenario IDs must be versioned |

### Corn-specific release action

The supplied raw `corn.mat` contains spectra and unscaled composition values. The current `dataset_corn.mat` cannot be presented as the required unstandardized export because its response columns have sample standard deviation approximately one and match a sample-standardized version of the raw composition values. The conversion script should reconstruct the exact first-derivative predictor representation described in `pipls.tex`, verify the wavelength interval, and export raw composition responses.

### FRED-MD-specific release action

A temporal economic panel should not default to shuffled repeated K-fold validation. The metadata and example should define the time column and use a time-respecting split. Any omitted response must be named in the preparation script and metadata; it must not be removed by list position.

### Licensing gate

No dataset should be committed merely because a source file is present in the archive. For each dataset, establish:

- canonical source and retrieval date;
- required citation;
- license or terms of use;
- whether redistribution of converted CSV files is permitted;
- whether only a preparation/download script may be distributed.

When redistribution is not permitted, keep the common directory structure but replace X/Y with a documented fetch-and-prepare command, checksum expectations, and a small non-sensitive fixture for tests.

## 11. Data loader

The package loader should read the common format rather than contain one handwritten function per dataset.

Recommended interface:

```python
from pipls.datasets import load_dataset

data = load_dataset("datasets/corn", as_frame=False)
X = data.data
y = data.target
print(data.feature_names)
print(data.target_names)
print(data.metadata)
```

Return a scikit-learn `Bunch` with at least:

```text
data
target
feature_names
target_names
sample_ids
metadata
frame                  # when as_frame=True
```

The loader should:

- validate X/Y sample alignment;
- reject nonnumeric model columns;
- preserve feature names;
- make no centering, standardization, imputation, or row filtering decisions;
- expose optional sample metadata and split roles;
- permit a direct path, so the Python package is not tied to the repository checkout;
- avoid downloading data implicitly in version 0.1.0.

Dataset preparation belongs under `scripts/prepare_data`, not in `src/pipls`.

## 12. Synthetic data

Retain Vishal's latent-structure generator, but convert it into a side-effect-free function using a local random generator:

```python
def make_pipls_regression(
    *,
    n_samples,
    n_features,
    n_targets,
    n_predictor_specific,
    n_shared,
    n_response_specific,
    noise,
    random_state=None,
):
    ...
```

Use `numpy.random.default_rng(random_state)` rather than changing NumPy's global seed. Return a `Bunch` containing $X$, $Y$, the latent matrices, and the true dimensions. Tests should verify shapes, reproducibility, zeroed loading blocks, and the noise-free rank relations.

The paper scripts should define scenario tables in data files such as YAML or CSV rather than repeating parameter combinations in many scripts. Each figure and table should have a scenario ID and fixed seed list.

## 13. Testing strategy

### 13.1 Core theory tests

- compare the regression map $\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf{T}}$ against a frozen trusted result from Vishal's core;
- compare predictions, not raw singular-vector signs;
- verify $\mathbf{P}^{\mathsf{T}}\mathbf{P}=\mathbf{I}$ and $\mathbf{Q}^{\mathsf{T}}\mathbf{Q}=\mathbf{I}$;
- verify diagonal entries of $\mathbf{D}$ are nonnegative and nonincreasing;
- verify dimensions for all admissible $(n,p,q,h,r_\pi)$ regimes;
- verify the no-truncation/full-response case against the corresponding multivariate least-squares fitted values where the manuscript states equivalence;
- test rank-deficient and $p\gg n$ matrices;
- test deterministic behavior.

SVD signs are not stable identifiers. Compare projection matrices, singular values, regression maps, and predictions.

### 13.2 Estimator-contract tests

Run scikit-learn's estimator checks in CI. The supplied versions presently fail those checks for different reasons, so passing the checks should be a release criterion.

Also test:

- `PiPLSRegression()` can be instantiated without arguments;
- `PiPLSRegression(n_components=k)` mirrors the ordinary PLS fitting pattern;
- cloning and parameter introspection;
- `get_params()` exposes `n_components`, `predictor_rank`, and `samples_per_predictor_rank`, but no `h`, `r_pi`, or `c` aliases;
- one-dimensional and multioutput targets;
- pandas feature names and reordered columns;
- wrong feature counts at prediction;
- fitted-state validation;
- `score()` returns a Python float;
- `coef_` orientation;
- `Pipeline`, `TransformedTargetRegressor`, and `GridSearchCV` interoperability;
- no mutation of constructor parameters during `fit`;
- no mutation of inputs when `copy=True`;
- `scale=True` and `scale=False` reproduce the documented centering and scaling semantics;
- fitted predictor and target statistics are computed from training data only.

### 13.3 Predictor-rank selection tests

- `predictor_rank="auto"` scans every and only admissible rank for the specified `n_components`;
- the automatic rank limit uses $n_{\mathrm{train,min}}$ in both the matrix-size cap and the samples-per-rank ceiling term;
- `predictor_rank="max"` uses the documented rule-derived limit directly;
- integer `predictor_rank` bypasses internal selection;
- `samples_per_predictor_rank` changes the upper bound as specified;
- automatic mode stores `predictor_rank_`, `max_predictor_rank_`, and rank-selection diagnostics;
- internal scaling is refitted independently inside each rank-selection fold;
- response-standardized MSE divides residuals by response scales from the matching training fold, uses `ddof=1`, and replaces zero scales by 1.0;
- triangular path grid contains every and only admissible pair;
- all parameter pairs use identical CV splits;
- conditional $r_\pi^*(h)$ is correct on a hand-checkable score surface;
- deterministic tie-breaking;
- fold-aware rank limits;
- failed candidates raise or are recorded through `error_score`, never silently skipped;
- `best_estimator_` is refitted with the selected pair;
- results match the preserved reference CV implementation for selected fixed-rank cases within tolerance;
- external `GridSearchCV` over `n_components` invokes conditional predictor-rank selection correctly;
- custom preprocessing is fitted inside each path-selection fold;
- a deliberately instrumented transformer detects the leakage that would occur if it were fitted before an internal automatic search;
- nested-CV examples do not resolve data-dependent parameters on outer validation data.

### 13.4 Leave-one-out tests

- `LeaveOneOut()` generates exactly $n$ splits, each with one validation observation and $n-1$ training observations;
- `PiPLSRegression(cv=LeaveOneOut())` and `PiPLSPathCV(cv=LeaveOneOut())` accept the standard splitter without a special LOO flag;
- every parameter candidate uses the same ordered set of LOO splits;
- no foldwise $R^2$ scorer is used or accepted for singleton validation folds;
- pooled response-standardized LOO-MSE equals the mean of the singleton-fold response-standardized MSE values;
- optional `oof_predictions_` preserves original sample order and corresponds to the documented selected parameterization;
- fold-local predictor and response means and scales exclude the held-out observation;
- external pipeline transformers are fitted only on the $n-1$ observations in each fold;
- the automatically generated rank grid satisfies $r_\pi\le\min(p,n-1)$ in every fold;
- the fold-safe API rule uses $n-1$ in both terms, while the unchanged manuscript-specific full-$n$ bound is passed explicitly by the reproduction script;
- the selected parameterization is refitted once on all $n$ observations when `refit=True`;
- pooled LOO $R^2$, when calculated, is not represented as a mean foldwise score;
- optional nested or held-out examples demonstrate how to obtain an unbiased estimate beyond the paper's selection-conditioned LOO performance result;
- fixed-protocol LOO results agree with Vishal's reference implementation where the scaling, rank-bound, and scoring conventions coincide.

### 13.5 Dataset tests

For each dataset:

- metadata validates against the schema;
- checksums match;
- dimensions and column counts match metadata;
- X/Y sample IDs match exactly;
- values are finite;
- column names are unique;
- split roles contain only permitted values;
- deterministic preparation reproduces byte-identical or numerically identical output;
- committed responses are not the standardized arrays from the current Corn analysis file;
- named exclusions and filters match metadata.

### 13.6 Paper-regression tests

Record the exact release commit, software environment, random seeds, split definitions, scoring rule, and expected result tolerances. At minimum, the paper scripts should reproduce:

- synthetic study tables;
- Corn and Pulp CV curves;
- Corn and Pulp selected $h$ and $r_\pi$ values;
- the paper's exact leave-one-out split count, explicit manuscript rank grid, selected parameters, response-standardized LOO-MSE performance values, and ordered out-of-fold predictions where retained;
- predictor-rank sweep figures;
- displayed $R^2$ and singular-value ratios.

Numerical tolerances should account for BLAS/SVD variation while still detecting substantive changes.

## 14. Examples and documentation

Each example should answer one specific question and run from a clean checkout.

1. `01_pls_style_estimator.py`: use `PiPLSRegression(n_components=...)`, fit, predict, score, and inspect `predictor_rank_`.
2. `02_automatic_predictor_rank.py`: show how `samples_per_predictor_rank` changes the admissible rank bound for one fixed `n_components`.
3. `03_two_parameter_path.py`: scan the complete triangular surface, inspect response-standardized CV-MSE versus `n_components`, and report the conditional best predictor rank.
4. `04_rule_fixed_predictor_rank.py`: use `predictor_rank="max"`.
5. `05_explicit_fixed_predictor_rank.py`: use an integer predictor rank for a reproducibility case.
6. `06_custom_preprocessing_path.py`: place a transformer and fixed Pi-PLS template in a pipeline, then search with `PiPLSPathCV` without leakage.
7. `07_custom_cv_splitter.py`: supply repeated, grouped, or time-aware CV.
8. `08_leave_one_out.py`: reproduce the manuscript's LOO protocol, calculate response-standardized LOO-MSE, report the selected minimum as the paper's selection-conditioned performance result, and distinguish optional pooled $R^2$ from the primary loss.
9. `09_official_train_test_split.py`: preserve the SARCOS split.

The documentation should begin with the direct comparison:

```python
pls = PLSRegression(n_components=3)
pipls = PiPLSRegression(n_components=3)
```

The next section should explain that Pi-PLS selects `predictor_rank_` automatically, bounded by `samples_per_predictor_rank`. The explicitly fixed integer interface should appear only under advanced usage.

Plotting should be function-oriented:

```python
from pipls.plotting import plot_cv_path, plot_latent_map

plot_cv_path(search)
plot_latent_map(model, feature_names=..., target_names=...)
```

Do not create one near-duplicate plotting module per dataset. Paper-specific figure composition belongs in `scripts/reproduce_paper`.

## 15. Packaging and dependency groups

Recommended `pyproject.toml` structure:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pipls"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.26",
  "scikit-learn>=1.4",
]

[project.optional-dependencies]
data = ["pandas>=2.0", "scipy>=1.11", "openpyxl>=3.1", "PyYAML>=6"]
plot = ["matplotlib>=3.8"]
dev = ["pytest", "pytest-cov", "ruff", "build", "twine"]
docs = ["sphinx", "myst-parser", "sphinx-rtd-theme"]
```

The exact minimum versions should be chosen from the oldest versions tested in CI, not copied without verification. The key separation is:

- core estimation: NumPy and scikit-learn;
- source-data conversion: pandas, SciPy, OpenPyXL, YAML;
- plotting: Matplotlib;
- development and documentation: separate extras.

Replace Vishal's `setuptools.backends.legacy:build` backend with the standard `setuptools.build_meta` backend. Keep the `src/` layout from the smaller package.

## 16. Migration map

| Destination | Preferred source | Action |
|---|---|---|
| `.llm/*` | generic LLM workflow template plus Pi-PLS specification | retain the tracked, non-installable workflow layer; replace generic notes with Pi-PLS mathematical, numerical, API, snapshot, patch, and validation contracts |
| `src/pipls/_core.py` | Vishal `pipls/core.py` | retain the algorithm and manuscript notation internally; rewrite validation, add a result structure, and test fixed $(h,r_\pi)$ behavior |
| `src/pipls/regression.py` | both estimators | use theory from Vishal and API discipline from the smaller package; expose `n_components`, `predictor_rank`, and `samples_per_predictor_rank`; implement fold-local automatic rank selection |
| `src/pipls/model_selection.py` | new, with Vishal CV as oracle | implement `predictor_rank_limit`, triangular path analysis, pipeline-aware searching, optional ordered out-of-fold predictions, explicit leave-one-out support through scikit-learn splitters, and preserved fixed-rank comparisons |
| `src/pipls/metrics.py` | new | define scalar score, explicit scaled-CV scorer, and LOO-safe pooled diagnostic helpers without foldwise $R^2$ |
| `src/pipls/datasets.py` | new generic loader | replace hard-coded source-format adapters |
| `scripts/prepare_data/*` | Vishal dataset adapters | convert each adapter into a deterministic named-column preparation script |
| `src/pipls/plotting.py` | selected generic Vishal plotting code | remove data-storage dependency and dataset clones |
| affine/core export | smaller package | retain only after the primary estimator is stable and tests confirm exact predictions |
| response-subspace variants | smaller package | omit from paper release |
| block scaling | smaller package trial code/design notes | omit implementation; preserve only pipeline-compatible architecture |
| custom CV base classes | Vishal | do not expose publicly; retain temporarily in a comparison branch or tests |

## 17. Clean public export procedure

1. Create a new empty Git repository.
2. Add the proposed skeleton, including the tracked `.llm/` layer and root `Makefile`; do not copy either embedded `.git` directory.
3. Write `.llm/project.md`, `.llm/mathematics.md`, `.llm/numerical_contracts.md`, `.llm/development.md`, and `.llm/public_api.md` before substantial porting, so that implementation work begins from an explicit repository, mathematical, numerical, and API contract.
4. Port the trusted fixed-$(h,r_\pi)$ core and write theory tests first.
5. Freeze the public naming contract: `n_components`, `predictor_rank`, and `samples_per_predictor_rank`.
6. Implement fixed integer predictor-rank fitting and make estimator checks pass.
7. Add `predictor_rank="max"` using one tested rank-bound helper.
8. Add `predictor_rank="auto"` with fold-local centering, scaling, and conditional CV selection.
9. Freeze reference outputs from Vishal's current implementation for selected deterministic cases.
10. Implement `PiPLSPathCV`, including full-pipeline cloning for custom preprocessing.
11. Freeze and implement the leave-one-out protocol: splitter usage, the $n_{\mathrm{train,min}}=n-1$ API bound, explicit manuscript rank-grid overrides, response-standardized singleton-fold scoring, selection-conditioned performance reporting, optional ordered out-of-fold predictions, and refitting.
12. Resolve the ceiling, small-sample threshold, and meaning of $n$; encode the result in one helper plus tests.
13. Convert each dataset with a named preparation script and complete provenance/licensing metadata.
14. Reproduce every manuscript result from the common-format datasets.
15. Add CI for tests, LOO protocol checks, leakage checks, data validation, and package build.
16. Build wheel and source distribution in a clean environment and run installation smoke tests.
17. Tag an immutable paper release, for example `v0.1.0-paper`, and place the tag, commit hash, and repository DOI in the manuscript.

Do not preserve archive clutter in Git history. The supplied Vishal tree contains an embedded repository, platform metadata, generated results, and large binary artifacts; the smaller archive contains editor backups and compiled LaTeX outputs. A clean import is preferable to history rewriting.


## 18. Proposed implementation phases

### Phase A: specification freeze

Deliverables:

- one-page mathematical contract for `_core.py`;
- public naming contract;
- exact admissibility rules;
- one exact ceiling-based rank-bound rule for a supplied numeric `samples_per_predictor_rank`;
- exact response-standardized CV-MSE definition;
- exact semantics of `predictor_rank="auto"`, `"max"`, and integer values;
- codified author decisions: fold-safe API bounds use $n_{\mathrm{train,min}}$, LOO performance uses the selected minimum response-standardized MSE, and paper-specific rank rules remain explicit reproduction inputs;
- paper dataset list and licensing status;
- the initial `.llm` repository map, mathematical contract, numerical contract, development rules, and public API inventory;
- tested snapshot, patch-creation, and guarded patch-application scripts;
- a root `Makefile` whose targets match CI.

The default workflow is no longer open: the ordinary estimator profiles predictor rank conditionally for the requested `n_components`. No API should be declared stable before the remaining numerical rules are settled.

### Phase B: fixed core and estimator

Deliverables:

- trusted core port;
- fixed-integer `PiPLSRegression`;
- scalar score and correct coefficient orientation;
- PLS-style `fit`, `predict`, `transform`, and `score`;
- estimator checks passing;
- equivalence tests against Vishal's core;
- initial API documentation.

### Phase C: automatic selection and preprocessing

Deliverables:

- `predictor_rank_limit`;
- `predictor_rank="max"`;
- `predictor_rank="auto"`;
- fold-local centering and scaling in every internal candidate fit;
- automatic-selection diagnostics;
- tests proving that validation observations do not influence fitted preprocessing statistics.

### Phase D: path analysis

Deliverables:

- triangular grid builder;
- `PiPLSPathCV`;
- pipeline-step detection and nested parameter setting;
- complete-pipeline cloning within each fold;
- repeated K-fold, leave-one-out, predefined split, group, and time-split examples;
- a dedicated `LeaveOneOut()` example with response-standardized singleton-fold MSE, selection-conditioned performance reporting, optional ordered out-of-fold predictions, and full-data refitting;
- explicit response-standardized MSE utility and negative scorer;
- regression comparison against Vishal's fixed-rank CV outputs.

### Phase E: datasets

Deliverables:

- schema and registry;
- converters and metadata for all datasets;
- Corn reconstruction;
- licensing decisions;
- checksum and shape tests;
- generic loader.

### Phase F: paper reproduction and release

Deliverables:

- all figures and tables generated from scripts;
- reproduction manifest;
- CI and clean-package build;
- tagged release, archived DOI, and manuscript repository reference;
- a clean `make snapshot` archive test;
- a documented root-relative patch workflow verified on at least one representative mathematical or numerical change.

## 19. Fixed decisions and remaining author choices

### Decisions fixed in revision 4

1. **General API rank bound:** use the smallest actual training-fold size throughout:

\begin{equation}
r_{\pi,\max}
=
\min\left[
p,n_{\mathrm{train,min}},
\left\lceil
\frac{n_{\mathrm{train,min}}}
{\texttt{samples\_per\_predictor\_rank}}
\right\rceil
\right].
\end{equation}

Under `LeaveOneOut()`, this uses $n-1$.

2. **Paper rank rule:** do not revise the manuscript protocol at this stage. Preserve the current one-in-five rule for $n\le50$ and one-in-ten rule otherwise, and preserve any full-$n$ rank grid required to reproduce published results. Encode it explicitly in paper scripts or manifests rather than as an estimator default.

3. **LOO role:** use LOO both for parameter selection and for the paper's reported performance results.

4. **Selection and performance objective:** use response-standardized MSE, with response scales fitted independently on each training fold and uniform response weighting.

5. **Reporting label:** the minimum LOO error selected from the evaluated path is a `selection-conditioned LOO performance estimate`; it is not described as an unbiased external-test estimate.

6. **Public interface:** ordinary use remains `PiPLSRegression(n_components=...)`; `LeaveOneOut()` is supplied through `cv`; `predictor_rank="auto"` is the normal mode; and custom preprocessing is searched through pipeline-aware `PiPLSPathCV`.

7. **LLM workflow boundary:** `.llm/` is tracked repository infrastructure, excluded from the installable package, and used as the authoritative communication layer for repository snapshots and root-relative patch exchange. It supplements but does not replace tests, documentation, review, or Git.

### Remaining choices

1. **Global selection of `n_components`:** minimum mean response-standardized CV-MSE, a one-standard-error rule, or analyst inspection of the profiled path. The current implementation plan uses the absolute minimum with deterministic low-complexity tie-breaking unless the paper specifies otherwise.

2. **Pooled LOO $R^2$:** whether to report it as a secondary diagnostic. It must never be presented as mean foldwise $R^2$ or as the primary performance criterion.

3. **Default internal CV splitter:** the exact behavior represented by `cv=5`, including whether shuffling is ever implicit. The conservative scikit-learn-style recommendation is deterministic, unshuffled five-fold splitting unless a splitter is supplied.

4. **Group and time metadata:** whether automatic estimator mode supports metadata routing in version 0.1.0 or directs such analyses to `PiPLSPathCV`.

5. **Corn processing:** exact Savitzky-Golay polynomial order, derivative order, window interpretation, boundary mode, wavelength range, and response units.

6. **FRED-MD validation:** temporal split and named response exclusion.

7. **Dataset redistribution:** which converted datasets may legally be committed.

8. **Compatibility attributes:** which PLS-style names are mathematically justified, rather than aliases added only for familiarity.

## 20. Release acceptance criteria

The paper repository is ready to publish only when all of the following are true:

- the core implementation matches the manuscript and trusted reference predictions;
- `.llm/project.md`, `.llm/mathematics.md`, `.llm/numerical_contracts.md`, `.llm/development.md`, and `.llm/public_api.md` agree with the implementation and public documentation;
- `.llm` is excluded from the wheel and is never imported by runtime package code;
- `make snapshot` produces a clean archive with repository-state metadata and no Git history, caches, compiled files, generated documentation, or forbidden result artifacts;
- the documented direct-Git workflow checks patch applicability before application and requires review before staging and committing;
- every LLM-assisted patch records which validation targets passed, failed, or were not run;
- `PiPLSRegression(n_components=k)` is the principal documented call;
- every estimator parameter has one documented meaning;
- the public API uses `n_components`, `predictor_rank`, and `samples_per_predictor_rank`;
- no public `h`, `r_pi`, or `c` aliases are present;
- `predictor_rank="auto"` selects the conditional CV optimum within the documented bound;
- `predictor_rank="max"` and integer predictor rank behave exactly as documented;
- `predictor_rank_` and `max_predictor_rank_` are exposed after fitting;
- all admissibility inequalities are enforced;
- `score()` returns scalar $R^2$;
- `coef_` and `predict()` follow the documented scikit-learn orientation;
- scikit-learn estimator checks pass on the supported version range;
- built-in centering and scaling are fitted separately on every internal training fold;
- arbitrary learned preprocessing is fitted only inside the complete pipeline's CV folds;
- the two-parameter path evaluates the exact intended triangular grid;
- the general rank-bound rule uses $n_{\mathrm{train,min}}$ in every term and is unambiguous and tested;
- paper-reproduction rank grids are passed explicitly and are never inferred from hidden estimator thresholds;
- `LeaveOneOut()` produces the documented fold-safe rank grid and fits all preprocessing on the corresponding $n-1$ training observations;
- no foldwise $R^2$ is used for singleton LOO validation folds;
- any pooled LOO $R^2$ is calculated from ordered out-of-fold predictions and labeled as a pooled diagnostic;
- the paper uses LOO for both parameter selection and reported performance, labels the selected minimum as a selection-conditioned LOO performance estimate, and does not claim it is an unbiased external-test estimate;
- response-standardized MSE is the implemented selection and paper-performance objective, with fold-local response scales, uniform response weighting, and aggregation stated in code, docs, and figure captions;
- every dataset has verified provenance, licensing, deterministic processing, X/Y alignment, and checksums;
- stored X/Y data are unstandardized under the repository definition;
- Corn has been regenerated from the original source with the manuscript's documented spectral treatment;
- all paper results are regenerated from a clean checkout;
- no block-scaling or alternative-response-subspace behavior is exposed as standard Pi-PLS;
- the source distribution and wheel install and pass smoke tests in a clean environment;
- the manuscript identifies the exact release tag, commit, and archival DOI.

## 21. External interface references

The compatibility recommendations above use the following official references:

- [scikit-learn `PLSRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html), including the `n_components` constructor convention, scaling of $X$ and $y$, scalar $R^2$ scoring, coefficient orientation, and transform behavior;
- [scikit-learn pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html), including joint parameter selection and prevention of preprocessing leakage;
- [scikit-learn `GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), including list-of-dictionaries parameter grids, scorers, refitting, and CV splitter support;
- [scikit-learn `LeaveOneOut`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html), for the standard $n$-split LOO protocol;
- [scikit-learn `cross_val_predict`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html), for ordered out-of-fold predictions from a fixed parameterization;
- [scikit-learn `r2_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html), including the requirement for at least two samples in an individual $R^2$ calculation;
- [scikit-learn nested cross-validation example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), for separating parameter selection from generalization-error assessment;
- [scikit-learn `TransformedTargetRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html), for target transformations inside composite estimators;
- [developing scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html), for constructor, validation, fitted-attribute, cloning, and estimator-check conventions;
- [Python Packaging User Guide: `pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/), for the build-system and dependency metadata structure.

## 22. Factual basis of this plan

This plan is based on direct inspection of the three supplied Pi-PLS artifacts and the supplied generic LLM-oriented repository template, execution of the smaller package's tests, execution of scikit-learn estimator checks against both current wrappers, numerical comparison of the two theory-equivalent core paths, inspection of the real-data loaders and source-file dimensions, the subsequent API-design and leave-one-out protocol discussions, and the official scikit-learn and Python packaging documentation cited above.

The revised API recommendations distinguish three categories explicitly:

- observed mathematical and implementation behavior from the supplied code and manuscript;
- public-interface decisions settled in the discussion, including descriptive parameter names and PLS-style default use;
- remaining scientific and reproducibility choices, including global `n_components` selection policy, optional pooled LOO $R^2$, exact spectral preprocessing, temporal validation details, and dataset provenance. The general rank-bound sample size, the unchanged paper-specific rule, the role of LOO in performance reporting, and the response-standardized error units are fixed decisions in revision 4.

No unresolved choice is presented as a tested fact.
