# Dataset interface and synthetic generator

The optional dataset interface provides a structured in-memory boundary for one packaged Pulp
dataset, package-owned synthetic data, and experiments. Real-data users may pass ordinary arrays or
data frames directly to `fit(X, Y)`; no container or metadata file is required for model fitting.
Sugarcane and Tobacco remain repository datasets with documentary `metadata.yaml` files.

## Validated dataset container

```python
from pipls.datasets import PiPLSDataset

sample = PiPLSDataset(
    X=X,
    Y=Y,
    feature_names=("x_1", "x_2"),
    target_names=("y_1",),
    sample_ids=("sample_1", "sample_2"),
    provenance={
        "source": "local-example",
        "license": "BSD-3-Clause",
        "citation": "Example data",
        "version": "1",
    },
    metadata={"instrument": "example"},
)
```

The container validates and then freezes its contents:

- `X` is a finite two-dimensional real numeric array;
- `Y` is a finite one- or two-dimensional real numeric array and is stored as two-dimensional;
- rows of `X` and `Y` must align exactly;
- feature names, target names, and sample identifiers are non-empty, unique strings with matching
  lengths;
- provenance must contain non-empty `source`, `license`, `citation`, and `version` strings;
- metadata values may be immutable scalars, sequences, mappings, or non-object NumPy arrays;
- model arrays are copied, converted to `float64`, and made read-only;
- metadata arrays are copied with their dtype preserved and made read-only;
- object-dtype metadata arrays are rejected because their elements can retain mutable Python
  objects; use nested sequences or mappings instead;
- nested metadata mappings and sequences are frozen recursively.

The scikit-learn-style aliases `data` and `target` refer to `X` and `Y`. The container also exposes
`n_samples`, `n_features`, and `n_targets`.

## Deterministic latent-structure generator

```python
from pipls.datasets import make_pipls_regression

synthetic = make_pipls_regression(
    n_samples=200,
    n_features=40,
    n_targets=8,
    n_shared=3,
    n_predictor_specific=2,
    n_response_specific=1,
    shared_strength=(3.0, 2.0, 1.0),
    predictor_specific_strength=(1.5, 0.8),
    response_specific_strength=0.7,
    shared_distribution="normal",
    predictor_specific_distribution="uniform",
    response_specific_distribution="normal",
    feature_scale=1.0,
    target_scale=1.0,
    noise=(0.1, 0.2),
    random_state=0,
)
```

The generator uses a local `numpy.random.Generator`; it never changes NumPy's global random state.
A scalar `noise` applies to both blocks, while `(x_noise, y_noise)` controls them separately.
Strength and scale arguments accept either a scalar or one value per relevant latent direction or
observed variable.
Because latent score columns are centered, each generated sample block must contain more rows than
the larger of the declared predictor and response latent ranks. Invalid degenerate dimensions are
rejected rather than silently producing a lower-rank realization.

The generative model is

\begin{equation}
\mathbf{X}
=
\mathbf{T}_{\mathrm{s}}\mathbf{A}_{\mathrm{s}}^{\mathsf T}
+
\mathbf{T}_{\mathrm{x}}\mathbf{A}_{\mathrm{x}}^{\mathsf T}
+
\mathbf{E}_{\mathrm{X}},
\qquad
\mathbf{Y}
=
\mathbf{T}_{\mathrm{s}}\mathbf{B}_{\mathrm{s}}^{\mathsf T}
+
\mathbf{T}_{\mathrm{y}}\mathbf{B}_{\mathrm{y}}^{\mathsf T}
+
\mathbf{E}_{\mathrm{Y}}.
\end{equation}

Here $\mathbf{T}_{\mathrm{s}}$ is shared, $\mathbf{T}_{\mathrm{x}}$ is predictor-specific, and
$\mathbf{T}_{\mathrm{y}}$ is response-specific. The loading columns are orthonormal within
each observed block. Latent score columns are centered and scaled
to unit sample standard deviation after being drawn from the selected normal or uniform source
distribution.

`synthetic.truth` is a read-only `PiPLSRegressionTruth` containing latent scores, contributing
loading blocks, signal matrices, noise matrices, strengths, and observed-variable scales. Effects
that are structurally absent from one observed block are described by the declared latent ranks;
they are not stored as redundant zero loading arrays.

## Companion-manuscript latent geometry

Use `make_pipls_latent_geometry()` when the data-generating distribution must match the Gaussian
latent geometry in the companion manuscript:

```python
from pipls.datasets import make_pipls_latent_geometry

synthetic = make_pipls_latent_geometry(
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

The generator implements

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

Every latent-score entry and loading entry is an independent standard-normal draw. Noise entries
are independent Gaussian draws with standard deviations `noise[0]` and `noise[1]`. The function
applies no centering, score standardization, loading orthonormalization, latent-strength scaling, or
observed-variable scaling. This is the synthetic model described in the
[companion manuscript](citation.md); it is separate from the more configurable package
generator above.

`synthetic.truth` is a read-only `PiPLSLatentGeometryTruth`. Its loading matrices retain the
manuscript orientation, with latent dimensions on rows and observed variables on columns. The
stored arrays therefore verify the equations directly:

```python
truth = synthetic.truth

x_signal = (
    truth.predictor_specific_scores @ truth.predictor_specific_loadings
    + truth.shared_scores @ truth.shared_predictor_loadings
)
y_signal = (
    truth.shared_scores @ truth.shared_response_loadings
    + truth.response_specific_scores @ truth.response_specific_loadings
)
```

A fixed `random_state` reproduces the package draw sequence exactly. Reproducing a particular
manuscript table or figure additionally requires the parameter grid, random seeds, resampling
protocol, and analysis settings used for that result.

## Leakage-free train/test generation

```python
from pipls.datasets import make_pipls_train_test

train, test = make_pipls_train_test(
    n_train=150,
    n_test=50,
    n_features=40,
    n_targets=8,
    n_shared=3,
    n_predictor_specific=2,
    n_response_specific=1,
    random_state=0,
)
```

Both blocks share one generated set of loadings, strengths, and observed-variable scales. Their
latent scores and noise are independent draws. No centering, standardization, imputation, feature
selection, or other fitted preprocessing is applied across the train/test boundary. The training
block is unchanged when only `n_test` changes.

## Real-data boundary

Real-data reading remains user-owned in general. Examples and reproduction scripts show how `X` and
`Y` are formed using ordinary NumPy, pandas, or domain-specific code. Pi-PLS provides no public
registry, generic loader, downloader, preparation-only script, or required metadata sidecar.

Pulp is one explicit package-owned exception. It is bundled with the installed distribution and
available without network access:

```python
from pipls.datasets import load_pulp

data = load_pulp()
X, Y = load_pulp(return_X_y=True)
```

The default result is an immutable `PiPLSDataset`; the direct return mode supplies the same read-only
`float64` matrices. The loader applies no imputation, centering, scaling, row filtering, or learned
preprocessing. Sugarcane, Tobacco, and user datasets continue to use explicit user-owned reading.

## Pulp real-data integration

The installed Pulp resources contain 46 rows, 14 fiber-property predictors, and eight responses
adapted from the supplementary material for the following article:

> Lindström, S. B., Ferritsius, R., Carlson, J. E., Persson, J., and Nilsson, F. (2025).
> Predicting handsheet properties and enhancing refiner control using fiber analyzer data and
> latent variable modeling. *Computers & Chemical Engineering*, **199**, 109143.
> [doi:10.1016/j.compchemeng.2025.109143](https://doi.org/10.1016/j.compchemeng.2025.109143).

The article identifies the refiner controls, internal state variables, pulp descriptions, and
handsheet properties as supplementary data. The package-owned dataset selects the documented
fiber-property and response columns from that public supplementary table.

`load_pulp()` reads the installed package resources and returns labels, stable sample identifiers,
public provenance, and immutable metadata together with the two model matrices. The quick start,
ordinary-PLS comparison, complete Pulp example, and tutorial renderer all use this public loader.
The installed resources are the sole active Pulp matrix representation.

## Sugarcane spectral integration

`datasets/sugarcane/` contains 57 samples, 1,721 LabSpec absorbance predictors spanning 780
through 2500 nm, and four responses: total sugar, crude protein, acid detergent fiber, and in vitro
organic matter digestibility. The original public dataset is:

> Chaix, G., Bendoula, R., and Zgouz, A. (2021). Data set of Visible-Near Infrared handled and
> micro-spectrometers -- comparison of their accuracy for predicting some sugarcane properties.
> *Mendeley Data*, Version 1.
> [doi:10.17632/mjttsjfj2s.1](https://doi.org/10.17632/mjttsjfj2s.1).

The accompanying data paper is:

> Zgouz, A., Héran, D., Barthès, B., Bastianelli, D., Bonnal, L., Baeten, V., Lurol, S.,
> Bonin, M., Roger, J.-M., Bendoula, R., and Chaix, G. (2020). Dataset of visible-near infrared
> handheld and micro-spectrometers -- comparison of the prediction accuracy of sugarcane
> properties. *Data in Brief*, **31**, 106013.
> [doi:10.1016/j.dib.2020.106013](https://doi.org/10.1016/j.dib.2020.106013).

The Mendeley collection is licensed CC BY 4.0. The repository adaptation matches the public
LabSpec and response tables by `Sample`, removes three rows whose total-sugar response is missing,
and applies no imputation or spectral preprocessing. `examples/06_sugarcane_real_data.py` reads
`X.csv` and `Y.csv` directly, evaluates the default path-evaluating `PiPLSSearchCV()`, plots
`component_path_` in memory, and fits a separate fixed model after a visible user component choice.
It obtains the conditional predictor-rank profile at the selected component count, obtains
selection-conditioned OOF predictions through `search.validation_report()`, and writes six final
PDF figures directly from immutable public results. The compact spectral-axis
description in `metadata.yaml` avoids
repeating 1,721 equivalent per-wavelength descriptions while still defining every predictor column
exactly.

## Tobacco spectral integration

`datasets/tobacco/` contains 347 samples, 1,557 raw FT-NIR absorbance predictors spanning
approximately 10,001 through 4,000 `cm^-1`, and 13 chemical-component responses. The original
public dataset is:

> Chen, H., Guo, J., Wang, H., and Zhao, L. (2025). A Near-Infrared Spectroscopy Dataset for
> Chemical Composition Prediction and Origin Identification of Tobacco Leaves. *Mendeley Data*,
> Version 1. [doi:10.17632/9z7dgdtggk.1](https://doi.org/10.17632/9z7dgdtggk.1).

The related data paper is:

> Chen, H., Guo, J., Li, B., Wan, R., Wang, C., Su, M., Wang, X., Liu, R., Wang, S., Liu, K.,
> Chen, L., Yang, S., Xie, F., Nie, C., Zhao, L., Wang, H., and Liu, Z. (2026). A dataset for
> geographical origin identification of tobacco leaves from multiple countries using near-infrared
> spectroscopy and chemometric analysis. *Data in Brief*, **64**, 112418.
> [doi:10.1016/j.dib.2025.112418](https://doi.org/10.1016/j.dib.2025.112418).

The Mendeley collection is licensed CC BY 4.0. The repository adaptation matches the public spectra
and chemistry tables one-to-one by sample ID, orders rows by that identifier, and excludes only
source metadata columns from the model matrices. All samples and chemical responses are retained.
No imputation, smoothing, derivatives, scatter correction, centering, scaling, or other spectral
preprocessing is applied. `examples/07_tobacco_real_data.py` reads `X.csv` and `Y.csv` directly,
evaluates a Pi-PLS component path with adaptive predictor-rank scanning and full predictor SVD,
plots that path in memory, applies the one-standard-error rule, and derives the conditional
predictor-rank profile at the returned component count before fitting the fixed Pi-PLS model. It
preserves the decreasing wavenumber coordinate, obtains selection-conditioned OOF predictions
through the search validation report, calculates raw observation diagnostics, and writes six
final PDFs.
Prediction diagnostics and coefficients are
paginated in source response order. The separate ordinary-PLS comparison remains in example 04.
