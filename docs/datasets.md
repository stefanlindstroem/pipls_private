# Dataset interface and synthetic generator

The optional dataset interface provides a structured in-memory boundary for packaged
[Pulp](#pulp-real-data-integration), [Sugarcane](#sugarcane-spectral-integration), and
[Tobacco](#tobacco-spectral-integration) datasets and structured experiments. Synthetic data are
generated separately as ordinary arrays. Real-data users may
pass ordinary arrays or
data frames directly to `fit(X, Y)`; no container or metadata file is required for model fitting.
Pulp, Sugarcane, and Tobacco have installed named loaders backed by canonical package resources.
Every maintained reference-data example uses those resources. Exact container, generator, and
loader signatures are collected in the [dataset API reference](api/datasets.md).

## Validated dataset container

```python
from pipls.datasets import PiPLSDataset

sample = PiPLSDataset(
    X=X,
    Y=Y,
    feature_names=("x_1", "x_2"),
    target_names=("y_1",),
    metadata={"instrument": "example"},
)
```

The container validates the model-facing structure and copies its inputs:

- `X` is a finite two-dimensional real numeric array;
- `Y` is a finite one- or two-dimensional real numeric array and is stored as two-dimensional;
- rows of `X` and `Y` must align exactly;
- feature names and target names are non-empty, unique strings with matching lengths;
- model arrays are copied, converted to `float64`, and made read-only;
- the top-level metadata mapping is shallow-copied;
- metadata values are retained as supplied rather than recursively copied, normalized, or frozen.

The canonical matrix attributes are `X` and `Y`. The container also exposes `n_samples`,
`n_features`, and `n_targets` as dataset-level dimensions.

## Deterministic synthetic generator

`make_synthetic_data()` provides the package-owned reproducible synthetic data used by tests,
validation, examples, and the [companion manuscript](citation.md#companion-paper):

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
observed-variable scaling. This is the package's single synthetic-data model and is also the
synthetic model described in the [companion manuscript](citation.md#companion-paper).

`make_synthetic_data()` returns only the generated `X` and `Y` arrays. The latent score and loading
matrices are implementation details rather than public result objects. A fixed `random_state` makes
the returned arrays deterministic for a given package implementation; the internal random-draw
order is not part of the public API. Reproducing a particular manuscript table or figure additionally
requires the parameter grid, random seeds, resampling protocol, and analysis settings used for that
result.

## Train/test splitting

Generate one synthetic `X, Y` pair and split its rows using ordinary array operations:

```python
from pipls.datasets import make_synthetic_data

X, Y = make_synthetic_data(
    n_samples=200,
    n_features=40,
    n_targets=8,
    n_shared=3,
    n_predictor_specific=2,
    n_response_specific=1,
    random_state=0,
)

X_train, X_test = X[:150], X[150:]
Y_train, Y_test = Y[:150], Y[150:]
```

The rows share one generated loading geometry while their latent scores and noise are independent
draws. Splitting is an analysis operation rather than part of the synthetic-data model. No fitted
preprocessing is learned across the boundary by this row split.

## Real-data boundary

Real-data reading remains user-owned in general. Examples and reproduction scripts show how `X` and
`Y` are formed using ordinary NumPy, pandas, or domain-specific code. Π-PLS provides no public
registry, generic loader, downloader, preparation-only script, or required metadata sidecar.

[Pulp](#pulp-real-data-integration), [Sugarcane](#sugarcane-spectral-integration), and
[Tobacco](#tobacco-spectral-integration) are explicit package-owned reference datasets. They are
bundled with the installed distribution and available without network access:

```python
from pipls.datasets import load_pulp, load_sugarcane, load_tobacco

pulp = load_pulp()
sugarcane = load_sugarcane()
tobacco = load_tobacco()
X, Y = load_tobacco(return_X_y=True)
```

The default result is a `PiPLSDataset` with read-only model arrays; the direct return mode supplies
the same read-only `float64` matrices. The loaders apply no imputation, centering, scaling, row filtering, or learned
preprocessing.

## Using the raw files outside Python

The named loaders read ordinary language-neutral resources. The same files can be consumed directly
from R, C++, MATLAB, Julia, or another environment. For reproducible non-Python use, prefer a tagged
source release, source distribution, or wheel rather than a moving development branch or an
installation path tied to one environment.

In a source checkout or unpacked source distribution, the canonical files are:

```text
src/pipls/_data/pulp/X.csv
src/pipls/_data/pulp/Y.csv
src/pipls/_data/pulp/metadata.json
src/pipls/_data/pulp/README.md
src/pipls/_data/pulp/LICENSE.txt

src/pipls/_data/sugarcane/X.csv
src/pipls/_data/sugarcane/Y.csv
src/pipls/_data/sugarcane/metadata.json
src/pipls/_data/sugarcane/README.md
src/pipls/_data/sugarcane/LICENSE.txt

src/pipls/_data/tobacco/X.csv
src/pipls/_data/tobacco/Y.csv
src/pipls/_data/tobacco/metadata.json
src/pipls/_data/tobacco/README.md
src/pipls/_data/tobacco/LICENSE.txt
```

A wheel is a ZIP archive. Inside a wheel, the corresponding directories are
`pipls/_data/pulp/`, `pipls/_data/sugarcane/`, and `pipls/_data/tobacco/`. In an installed
environment they normally appear below the environment-specific
`<site-packages>/pipls/_data/<dataset>/` directory. Each `X.csv` and `Y.csv` pair is exactly the
matrix pair returned by its Python loader; loading applies no additional preprocessing.

## Pulp real-data integration {#pulp-real-data-integration}

The installed Pulp resources contain 46 rows, 14 fiber-property predictors, and eight responses
adapted from the supplementary material for the following article:

> Lindström, S. B., Ferritsius, R., Carlson, J. E., Persson, J., and Nilsson, F. (2025).
> Predicting handsheet properties and enhancing refiner control using fiber analyzer data and
> latent variable modeling. *Computers & Chemical Engineering*, **199**, 109143.
> [doi:10.1016/j.compchemeng.2025.109143](https://doi.org/10.1016/j.compchemeng.2025.109143).

The article identifies the refiner controls, internal state variables, pulp descriptions, and
handsheet properties as supplementary data. The package-owned dataset selects the documented
fiber-property and response columns from that public supplementary table.

`load_pulp()` reads the installed package resources and returns labels and metadata
together with the two model matrices. Public provenance is stored once under
`dataset.metadata["provenance"]`. The quick start,
ordinary-PLS comparison, complete Pulp example, and tutorial renderer all use this public loader.
The installed resources are the sole active Pulp matrix representation.

## Sugarcane spectral integration {#sugarcane-spectral-integration}

The installed Sugarcane resources contain 57 samples, 1,721 LabSpec absorbance predictors spanning
780 through 2500 nm, and four responses: total sugar, crude protein, acid detergent fiber, and in
vitro
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

The Mendeley collection is licensed CC BY 4.0. The package adaptation matches the public LabSpec
and response tables by `Sample`, removes three rows whose total-sugar response is missing, and
applies no imputation or spectral preprocessing. `load_sugarcane()` returns the labeled package
dataset with read-only model arrays or those arrays directly. `examples/05_sugarcane_real_data.py` obtains the matrices,
wavelength labels, and response names from that result. The example deliberately fixes predictor
rank with the [EPV policy](path_selection.md#epv-policy) using
`samples_per_predictor_rank=5.0`, providing additional regularization of the spectral predictor
subspace before the component count is chosen separately. It then diagnoses the accepted selection
with selection-conditioned OOF predictions, refits that same selection, and writes five
wavelength-aware PDF figures from immutable public results. The interpretation of the two complexity
controls is discussed under [$r_\pi$ and $h$](theory.md#interpretation-of-the-ranks). The compact spectral-axis description in
`metadata.json` avoids repeating 1,721 equivalent per-wavelength descriptions while still defining
every predictor column exactly.

## Tobacco spectral integration {#tobacco-spectral-integration}

The installed Tobacco resources contain 347 samples, 1,557 raw FT-NIR absorbance predictors
spanning approximately 10,001 through 4,000 `cm^-1`, and 13 chemical-component responses. The
original public dataset is:

> Chen, H., Guo, J., Wang, H., and Zhao, L. (2025). A Near-Infrared Spectroscopy Dataset for
> Chemical Composition Prediction and Origin Identification of Tobacco Leaves. *Mendeley Data*,
> Version 1. [doi:10.17632/9z7dgdtggk.1](https://doi.org/10.17632/9z7dgdtggk.1).

The related data paper is:

> Chen, H., Guo, J., Li, B., Wan, R., Wang, C., Su, M., Wang, X., Liu, R., Wang, S., Liu, K.,
> Chen, L., Yang, S., Xie, F., Nie, C., Zhao, L., Wang, H., and Liu, Z. (2026). A dataset for
> geographical origin identification of tobacco leaves from multiple countries using near-infrared
> spectroscopy and chemometric analysis. *Data in Brief*, **64**, 112418.
> [doi:10.1016/j.dib.2025.112418](https://doi.org/10.1016/j.dib.2025.112418).

The Mendeley collection is licensed CC BY 4.0. The package adaptation matches the public spectra
and chemistry tables one-to-one by sample ID, orders rows by that identifier, and excludes only
source metadata columns from the model matrices. All samples and chemical responses are retained.
No imputation, smoothing, derivatives, scatter correction, centering, scaling, or other spectral
preprocessing is applied. `load_tobacco()` returns the labeled package dataset with read-only model
arrays or those arrays directly. `examples/06_tobacco_real_data.py` obtains the matrices, decreasing wavenumber
labels, and source-order response names from that result. Unlike Sugarcane's fixed EPV rank, Tobacco
optimizes predictor rank and uses a 10% relative tolerance to retain a smaller spectral subspace
when its CV performance remains close to the conditional optimum. A separate 10% relative tolerance
then selects component count. The workflow carries that selection through conditional rank evidence,
selection-conditioned OOF review, and final refitting, and writes six final PDFs. See
[search-owned selection rules](path_selection.md#search-owned-selection-rules) for the two-stage
selection semantics. Prediction diagnostics and coefficients are paginated in source response order.
The separate ordinary-PLS comparison remains in example 03.
