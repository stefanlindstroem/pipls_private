# Companion-manuscript synthetic data

This guide shows how to generate the Gaussian latent-space model used in the companion manuscript.
It covers the manuscript's **data-generating distribution** and the information needed to recreate
one **seeded synthetic dataset**. It does not reproduce the manuscript's complete tables, figures,
resampling study, comparator implementations, or reporting pipeline.

The scientific reference is:

> Vishal Agrawal, Fritjof Nilsson, and Stefan B. Lindström, “Panoramic Partial Least Squares
> (Pi-PLS): Transparent, parsimonious, and more interpretable multivariate regression model.”
> Manuscript under revision at *Computers & Chemical Engineering*, CACE-D-26-00847.

See [authors, license, and citation](citation.md) for the maintained citation metadata.

## Three reproducibility levels

Keep three distinct targets separate:

1. **Data-generating distribution.** Match the equations, dimensions, independent Gaussian draws,
   and noise standard deviations. `make_pipls_latent_geometry()` provides this capability.
2. **One seeded dataset.** In addition, record the exact generator arguments, random seed, package
   version, and numerical environment. The same inputs then identify one deterministic realization.
3. **Complete manuscript results.** Also reproduce the parameter grids, all random seeds, data
   splitting and cross-validation protocols, preprocessing, comparator implementations, dependency
   versions, aggregation rules, and figure or table code. Those publication-specific assets belong
   in a separate reproduction repository that pins a released `pipls` version.

The package therefore supplies the method and exact synthetic distribution without treating this
software repository as the orchestration environment for one paper.

## Generate the manuscript distribution

The manuscript separates predictor-specific, shared, and response-specific latent variation.
Upright subscripts identify fixed roles or blocks: $\mathrm{p}$ is predictor-specific,
$\mathrm{s}$ is shared, $\mathrm{r}$ is response-specific, $\mathrm{sp}$ is shared-predictor,
and $\mathrm{sr}$ is shared-response.

\begin{equation}
\mathbf{X}
=
\boldsymbol{\Lambda}_{\mathrm{p}}\mathbf{L}_{\mathrm{p}}
+
\boldsymbol{\Lambda}_{\mathrm{s}}\mathbf{L}_{\mathrm{sp}}
+
\boldsymbol{\varepsilon}_{\mathrm{X}},
\qquad
\mathbf{Y}
=
\boldsymbol{\Lambda}_{\mathrm{s}}\mathbf{L}_{\mathrm{sr}}
+
\boldsymbol{\Lambda}_{\mathrm{r}}\mathbf{L}_{\mathrm{r}}
+
\boldsymbol{\varepsilon}_{\mathrm{Y}}.
\end{equation}

Every entry of the three latent-score matrices and four loading matrices is sampled independently
from $\mathcal{N}(0,1)$. Predictor and response noise entries are independent Gaussian draws with
standard deviations $\sigma_{\mathrm{X}}$ and $\sigma_{\mathrm{Y}}$.

The following call uses a representative geometry discussed in the manuscript: 40 observations,
80 predictors, 30 responses, four predictor-specific directions, four shared directions, one
response-specific direction, and noise standard deviations of 0.5 in both blocks.

```python
from pipls.datasets import make_pipls_latent_geometry

data = make_pipls_latent_geometry(
    n_samples=40,
    n_features=80,
    n_targets=30,
    n_predictor_specific=4,
    n_shared=4,
    n_response_specific=1,
    noise=(0.5, 0.5),
    random_state=0,
)

X = data.X
Y = data.Y
truth = data.truth
```

The seed `0` is illustrative. It reproduces this package-generated realization; it should not be
interpreted as a seed used for a manuscript result unless that seed is separately recorded in the
publication-reproduction materials.

The generator performs no latent-score centering or standardization, no loading
orthonormalization, no strength rescaling, and no observed-variable rescaling. The broader
`make_pipls_regression()` and `make_pipls_train_test()` APIs intentionally provide those additional
package capabilities and are not substitutes when the manuscript distribution itself is required.

See the
[`make_pipls_latent_geometry()` API](api/datasets.md#pipls.datasets.make_pipls_latent_geometry)
for its complete validation and return contract.

## Verify the stored latent geometry

`data.truth` exposes the manuscript matrices in manuscript orientation: latent dimensions are rows
of the loading matrices and observed variables are columns. The stored arrays permit direct
verification of both signal equations:

```python
import numpy as np

np.testing.assert_allclose(
    truth.x_signal,
    truth.predictor_specific_scores @ truth.predictor_specific_loadings
    + truth.shared_scores @ truth.shared_predictor_loadings,
)
np.testing.assert_allclose(
    truth.y_signal,
    truth.shared_scores @ truth.shared_response_loadings
    + truth.response_specific_scores @ truth.response_specific_loadings,
)
np.testing.assert_allclose(X, truth.x_signal + truth.x_noise)
np.testing.assert_allclose(Y, truth.y_signal + truth.y_noise)
```

All dataset and truth arrays are defensive read-only `float64` values. The dataset metadata records
the generator name, random seed, latent dimensions, noise standard deviations, and distribution.

## Record one deterministic realization

A seeded dataset is reproducible only when its complete generation context is retained. Record at
least:

- the `pipls` version or source commit;
- the NumPy version;
- every `make_pipls_latent_geometry()` argument;
- the integer `random_state`;
- any serialization or numeric-output format used downstream.

The generator uses a local NumPy `default_rng` and a fixed public draw order. Decision 0119 records
that order so a fixed seed has an auditable meaning. The metadata can be inspected directly:

```python
print(data.metadata["generator"])
print(data.metadata["random_state"])
print(data.metadata["latent_dimensions"])
print(data.metadata["noise_standard_deviation"])
```

Recording a seed without the dimensions, noise settings, package version, and numerical environment
is insufficient for an exact realization-level claim.

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
d_p = truth.n_predictor_specific
d_s = truth.n_shared

predictor_rank = d_p + d_s
n_components = d_s
```

These are oracle dimensions available because the synthetic truth is known. They are not a
real-data rank-selection rule and do not alter the package's practical search workflow. Under
additive noise, the observed leading singular directions also need not separate signal and noise
exactly.

A fixed model can be evaluated at these known dimensions when reproducing that synthetic protocol.
The manuscript's real-data workflow is different: it fixes $r_\pi$ with the EPV-inspired rule
$r_\pi=\min[p,\lceil n/c\rceil]$ and then selects $h$ by cross-validation. In this package that
protocol must be requested explicitly, for example:

```python
from pipls import PiPLSSearchCV

search = PiPLSSearchCV(
    predictor_rank_values="epv",
    samples_per_predictor_rank=10.0,
    cv=cv,
).fit(X, Y)
```

Use the value of $c$ declared by the reproduction protocol; the manuscript describes $c=10$ as its
ordinary EPV choice and $c=5$ as a more permissive small-sample choice. The package default is not
that manuscript heuristic: ordinary `PiPLSSearchCV()` optimizes $r_\pi$ over the complete
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

For general synthetic examples with configurable strengths, distributions, orthonormal loading
directions, and independent train/test blocks, continue to use the
[first synthetic tutorial](tutorials/synthetic.md). For the fixed Π-PLS construction, see the
[theory overview](theory.md); for software and generated-documentation controls, see
[reproducibility](reproducibility.md).
