# Datasets and generators

`pipls.datasets` provides three packaged real-data loaders, immutable dataset and truth records, and
a deterministic synthetic generator. Ordinary arrays and data frames passed directly to `fit(X, y)`
remain the normal interface for user data.

## Packaged datasets

The package includes Pulp, Sugarcane, and Tobacco as installed resources. Loading performs no
network access or preprocessing. The [Reference datasets](../datasets.md) page owns provenance,
licensing, adaptation, and matrix-dimension details.

::: pipls.datasets.load_pulp
    options:
      members: false

::: pipls.datasets.load_sugarcane
    options:
      members: false

::: pipls.datasets.load_tobacco
    options:
      members: false

## Synthetic generator

The generator distinguishes three latent roles. Its data model is

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

The figure illustrates this equation: predictor-specific directions contribute only to
$\mathbf{X}$, shared directions contribute to both $\mathbf{X}$ and $\mathbf{Y}$, and
response-specific directions contribute only to $\mathbf{Y}$. Independent noise is added to the two
observed blocks.

![Latent roles in the synthetic generator: predictor-specific variation contributes only to X, shared variation contributes to both X and Y, and response-specific variation contributes only to Y.](../assets/figures/latent_geometry_generator.svg){ style="width: 100%; height: auto;" }

`n_predictor_specific`, `n_shared`, and `n_response_specific` set the three latent dimensions.
`make_synthetic_data()` uses independent standard-normal score and loading entries plus independent
Gaussian predictor and response noise. The
[companion-manuscript synthetic-data](../manuscript_reproduction.md) page owns the
publication-reproduction boundary; the
[synthetic tutorial](../tutorials/synthetic.md) shows a worked train/test analysis using an explicit
row split.

::: pipls.datasets.make_synthetic_data
    options:
      members: false

## Containers and truth records

::: pipls.datasets.PiPLSDataset
    options:
      members:
        - n_samples
        - n_features
        - n_targets

`SyntheticDataTruth` records the latent scores, loadings, signal, and noise used to generate a
synthetic dataset. It is normally inspected through `dataset.truth`.

::: pipls.datasets.SyntheticDataTruth
    options:
      show_signature: false
      members:
        - n_shared
        - n_predictor_specific
        - n_response_specific
