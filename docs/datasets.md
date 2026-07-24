# Dataset interface and synthetic generator

The optional dataset interface provides a structured in-memory boundary for package-owned
synthetic data and experiments. Real-data users may pass ordinary arrays or data frames directly
to `fit(X, Y)`; no container or metadata file is required for model fitting. Repository-included
real datasets nevertheless use a consistent documentary `metadata.yaml`.

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

\[
X = T_s A_s^\mathsf{T} + T_x A_x^\mathsf{T} + E_x,
\qquad
Y = T_s B_s^\mathsf{T} + T_y B_y^\mathsf{T} + E_y,
\]

where `T_s` is shared, `T_x` is predictor-specific, and `T_y` is response-specific. The loading
columns are orthonormal within each observed block. Latent score columns are centered and scaled
to unit sample standard deviation after being drawn from the selected normal or uniform source
distribution.

`synthetic.truth` is a read-only `PiPLSSyntheticTruth` containing latent scores, contributing
loading blocks, signal matrices, noise matrices, strengths, and observed-variable scales. Effects
that are structurally absent from one observed block are described by the declared latent ranks;
they are not stored as redundant zero loading arrays.

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

Real-data reading remains user-owned. Examples and reproduction scripts must show how `X` and `Y`
are read and formed directly using ordinary NumPy, pandas, or domain-specific code. The shipped CSV
assets require no `data` installation extra: they are ordinary repository files, not entries in a
runtime registry. The project tracks public provenance, licenses, and analysis-facing
transformations for its own datasets, but no public registry, generic loader, preparation-only
script, or required metadata sidecar is part of the runtime API.


## Pulp real-data integration

`datasets/pulp/` contains 46 rows, 14 fiber-property predictors, and eight responses adapted
from the supplementary material for the following article:

> Lindström, S. B., Ferritsius, R., Carlson, J. E., Persson, J., and Nilsson, F. (2025).
> Predicting handsheet properties and enhancing refiner control using fiber analyzer data and
> latent variable modeling. *Computers & Chemical Engineering*, **199**, 109143.
> [doi:10.1016/j.compchemeng.2025.109143](https://doi.org/10.1016/j.compchemeng.2025.109143).

The article identifies the refiner controls, internal state variables, pulp descriptions, and
handsheet properties as supplementary data. The repository selects the documented fiber-property
and response columns from that public supplementary table.

`examples/05_pulp_real_data.py` reads `X.csv` and `Y.csv` directly with pandas and then relies on
ordinary estimator validation when fitting `PiPLSPathCV` with its adaptive defaults. It does not
repeat repository-table validation, set a predictor-rank ceiling, parse `metadata.yaml`, or call a
package loader.

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
`X.csv` and `Y.csv` directly, evaluates the default selection-only `PiPLSPathCV()`, plots
`component_path_` in memory, and fits a separate fixed model after a visible user component choice.
It calculates selection-conditioned OOF predictions with scikit-learn and writes five final PDF figures directly
from immutable inspection results. The compact spectral-axis description in `metadata.yaml` avoids
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
plots that path in memory, and then fits a separately chosen fixed Pi-PLS model. It preserves the
decreasing wavenumber coordinate, calculates selection-conditioned OOF predictions and raw
observation diagnostics, and writes five final PDFs. Prediction diagnostics and coefficients are
paginated in source response order. The separate ordinary-PLS comparison remains in example 04.
