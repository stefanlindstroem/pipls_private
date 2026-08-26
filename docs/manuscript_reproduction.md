# Companion-manuscript synthetic data

This guide shows how to generate the Gaussian latent-space model used in the
[companion manuscript](citation.md#companion-paper). It covers the manuscript's
**data-generating distribution** and the information needed to recreate
one **seeded synthetic dataset**. It does not reproduce the manuscript's complete tables, figures,
resampling study, comparator implementations, or reporting pipeline.

The scientific reference is:

> Vishal Agrawal, Fritjof Nilsson, and Stefan B. Lindström, “Panoramic Partial Least Squares
> (Pi-PLS): Transparent, parsimonious, and more interpretable multivariate regression model.”
> Manuscript under revision at *Computers & Chemical Engineering*, CACE-D-26-00847.

See [authors, license, and citation](citation.md#companion-paper) for the maintained citation metadata.

## Three reproducibility levels

Keep three distinct targets separate:

1. **Data-generating distribution.** Match the dimensions, Gaussian construction, and noise
   standard deviations defined by the [`make_synthetic_data()` API](api/datasets.md#synthetic-generator).
2. **One seeded dataset.** In addition, record the exact generator arguments, random seed, package
   version, and numerical environment. The same inputs then identify one deterministic realization.
3. **Complete manuscript results.** Also reproduce the parameter grids, all random seeds, data
   splitting and cross-validation protocols, preprocessing, comparator implementations, dependency
   versions, aggregation rules, and figure or table code. Those publication-specific assets belong
   in a separate reproduction repository that pins a released `pipls` version.

The package therefore supplies the method and exact synthetic distribution without treating this
software repository as the orchestration environment for one paper.

## Generate the manuscript distribution

The package generator implements the manuscript's predictor-specific, shared, and response-specific
Gaussian latent construction. The [Datasets and generators API](api/datasets.md#synthetic-generator)
is the canonical definition of its equation, latent roles, distributional assumptions, and
non-transformation boundary.

The following call uses a representative geometry discussed in the manuscript: 40 observations,
80 predictors, 30 responses, four predictor-specific directions, four shared directions, one
response-specific direction, and noise standard deviations of 0.5 in both blocks.

```python
from pipls.datasets import make_synthetic_data

X, Y = make_synthetic_data(
    n_samples=40,
    n_features=80,
    n_targets=30,
    n_predictor_specific=4,
    n_shared=4,
    n_response_specific=1,
    noise=(0.5, 0.5),
    random_state=0,
)
```

The seed `0` is illustrative. It reproduces this package-generated realization; it should not be
interpreted as a seed used for a manuscript result unless that seed is separately recorded in the
publication-reproduction materials.

See the
[`make_synthetic_data()` API](api/datasets.md#pipls.datasets.make_synthetic_data)
for the complete validation and return contract.

## Generator boundary

The generator returns only the observed matrices `X` and `Y`. The latent score, loading, signal, and
noise matrices are implementation details rather than public result objects. The generating
dimensions are already explicit in the function arguments and are sufficient for the synthetic
validation and reproduction protocols maintained by the package.

## Record one deterministic realization

A seeded dataset is reproducible only when its complete generation context is retained. Record at
least:

- the `pipls` version or source commit;
- the NumPy version;
- every `make_synthetic_data()` argument;
- the integer `random_state`;
- any serialization or numeric-output format used downstream.

The generator uses a local NumPy `default_rng`. A fixed seed and identical arguments reproduce the
same arrays for a given package implementation, while the internal random-draw order remains an
implementation detail. Recording a seed without the dimensions, noise settings, package version,
and numerical environment is insufficient for an exact realization-level claim.

## Known dimensions in the synthetic experiments

For the manuscript synthetic geometry, the noiseless predictor signal has rank at most
$d_{\mathrm{p}}+d_{\mathrm{s}}$, and the predictable shared relation has dimension at most
$d_{\mathrm{s}}$. The manuscript's synthetic evaluations use the known generating dimensions

\begin{equation}
r_\pi=d_{\mathrm{p}}+d_{\mathrm{s}},
\qquad
h=d_{\mathrm{s}}.
\end{equation}

With the representative call above, these values are:

```python
d_p = 4
d_s = 4

predictor_rank = d_p + d_s
n_components = d_s
```

These are oracle dimensions available because the generating dimensions are known. They are not a
real-data rank-selection rule and do not alter the package's practical search workflow. Under
additive noise, the observed leading singular directions also need not separate signal and noise
exactly.

A fixed model can be evaluated at these known dimensions when reproducing that synthetic protocol.
For manuscript-aligned Π-PLS fitting, the
[response-subspace construction](theory.md#response-subspace-selection) is also fixed:
`response_subspace="cross_covariance"`. This is both the package default and the construction used
in the [peer-reviewed companion publication](citation.md#companion-paper). The alternative
`"least_squares"` policy is a software extension and must not be used when claiming reproduction of
the publication's response-subspace construction.

The manuscript's real-data workflow is different from the synthetic oracle-rank protocol: it fixes
$r_\pi$ with the EPV-inspired rule $r_\pi=\min[p,\lceil n/c\rceil]$ and then selects $h$ by
cross-validation. The separate roles of these two controls are summarized under
[Interpretation of the ranks](theory.md#interpretation-of-the-ranks). In this package that complete
model configuration can be requested explicitly:

```python
from pipls import PiPLSRegression, PiPLSSearchCV

publication_template = PiPLSRegression(
    n_components=1,
    predictor_rank=1,
    response_subspace="cross_covariance",
)

search = PiPLSSearchCV(
    estimator=publication_template,
    predictor_rank_values="epv",
    samples_per_predictor_rank=10.0,
    cv=cv,
).fit(X, Y)
```

The pair `(1, 1)` is only a valid search-template seed; `PiPLSSearchCV` replaces it with the
evaluated component count and EPV-fixed predictor rank while preserving `response_subspace`. Use the
value of $c$ declared by the reproduction protocol; the manuscript describes $c=10$ as its ordinary
EPV choice and $c=5$ as a more permissive small-sample choice. The package default rank policy is
not that manuscript heuristic: ordinary `PiPLSSearchCV()` optimizes $r_\pi$ over the complete
fold-feasible domain with exhaustive coverage.

Any claim of complete manuscript-result reproduction must additionally match the manuscript's
preprocessing, validation, comparator, repetition, aggregation, and reporting settings.

## What this guide does not reproduce

This page does not encode the manuscript's complete simulation grid or generate its figures and
tables. In particular, it does not specify or run:

- all parameter combinations and random instances;
- repeated cross-validation and external train-test protocols;
- ordinary PLS, CCA, or OLS comparison implementations;
- result aggregation, uncertainty summaries, or plotting code;
- publication-specific dependency pins and execution manifests.

Those assets should live in a downstream reproduction repository and depend on a tagged `pipls`
release. This separation allows the package to remain a general software product while still
providing the exact synthetic data model required by the paper.

## Relationship to ordinary package workflows

Nothing in this guide changes or replaces the package's practical real-data workflow.
`PiPLSSearchCV`, its defaults, conditional predictor-rank search, selection rules, validation
reports, and the maintained Pulp, Sugarcane, and Tobacco examples remain independent package
capabilities.

For a worked synthetic train/test analysis using an explicit row split, see the
[first synthetic tutorial](tutorials/synthetic.md). For the fixed Π-PLS construction, see the
[theory overview](theory.md); for software and generated-documentation controls, see
[reproducibility](reproducibility.md).
