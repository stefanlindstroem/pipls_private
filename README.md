# PiPLS (Π-PLS)

[![Tests](https://github.com/stefanlindstroem/pipls_private/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/stefanlindstroem/pipls_private/actions/workflows/tests.yml)
[![Documentation](https://github.com/stefanlindstroem/pipls_private/actions/workflows/documentation.yml/badge.svg?branch=master)](https://github.com/stefanlindstroem/pipls_private/actions/workflows/documentation.yml)
[![Python ≥3.10](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?logo=python&logoColor=white)](docs/compatibility.md)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-4C1.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-article-007396.svg)](https://doi.org/10.1016/j.compchemeng.2026.109913)

**Panoramic partial least squares for compact and interpretable multivariate regression in Python.**

## Overview

Panoramic partial least squares (Π-PLS) is a multivariate latent-variable regression method for
problems with correlated, potentially high-dimensional predictors and multiple responses. The
`pipls` package provides a scikit-learn-style Python implementation for model fitting, selection,
prediction, and inspection.

The method first retains a broad, rank-controlled predictor subspace and then represents the
predictive relationship through a smaller set of paired latent modes. This separation allows Π-PLS
to preserve a sufficiently broad view of predictor variation without requiring an equally large
final latent model.

In the synthetic and real-world problems examined in the accompanying study, Π-PLS achieved
competitive predictive accuracy with a parsimonious latent representation. Each mode connects one
orthonormal predictor direction to one orthonormal response direction through a nonnegative
coupling strength. Prediction and interpretation are therefore expressed through the same compact
fitted structure.

![PiPLS fitted geometry: predictor variables combine into predictor directions, each predictor direction is paired one-to-one with a response direction through a scalar dilation, and the response directions combine into predicted responses.](docs/assets/figures/pipls_model_overview.svg)

## Why PiPLS?

- **Panoramic predictor representation.** The retained predictor rank $r_\pi$ is controlled
  separately from the final mode count $h$. The model can therefore begin with a broader predictor
  representation and compress it only when forming the predictive latent structure.
- **Compact predictive model.** The final relationship is expressed through a small number of
  paired modes. In the accompanying study, this produced competitive predictive accuracy and often
  a more parsimonious model than standard PLS, although the outcome remains data dependent.
- **One-to-one latent-mode interpretation.** Every mode pairs one predictor direction with one
  response direction through a single nonnegative dilation, making the fitted relationship
  inspectable mode by mode.
- **Prediction and interpretation in one structure.** The factorization
  $\widehat{\mathbf{Y}}=\mathbf{X}\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$ is both the
  prediction model and the basis for examining predictor scores, response directions, mode
  strengths, and regression coefficients.

## How it works

1. **Retain a predictor panorama.** A singular value decomposition gives a rank-controlled
   predictor basis $\mathbf{\Pi}$ and retained scores $\mathbf{Z}=\mathbf{X}\mathbf{\Pi}$ of
   dimension $r_\pi$.
2. **Identify a response-linked latent relation.** The default construction selects an
   $h$-dimensional response subspace from the cross-covariance between $\mathbf{Z}$ and
   $\mathbf{Y}$, then estimates the reduced regression map by least squares.
3. **Form paired modes.** Diagonalizing that map produces orthonormal predictor directions
   $\mathbf{P}$, orthonormal response directions $\mathbf{Q}$, and the nonnegative diagonal
   coupling matrix $\mathbf{D}$.

The two rank controls satisfy $h\leq r_\pi$: $r_\pi$ governs the breadth of the retained predictor
representation, while $h$ governs the size of the final paired model. The
[theory overview](docs/theory.md) gives the complete derivation.

## Documentation

The [rendered documentation](https://stefanlindstroem.github.io/pipls_private/) takes users from
installation to model selection, validation, prediction, and interpretation. New users should begin
with the installation guide and quick-start tutorial; the remaining tutorials provide complete
workflows, while the theory and API sections document the mathematical construction and public
interfaces.

- [Installation](docs/installation.md) — create an isolated environment and install the package.
- [Quick start](docs/tutorials/quick_start.md) — move from an included dataset to a fitted model and
  prediction diagnostics.
- [Tutorials and examples](docs/examples.md) — follow complete executable workflows for selection,
  validation, and interpretation.
- [Theory](docs/theory.md) — study the mathematical construction and paired latent modes.
- [API reference](docs/api/index.md) — inspect the public estimator, search, dataset, and result
  interfaces.

Contributor setup and repository validation are documented separately in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Citation and license

If Π-PLS contributes to your work, please cite the accompanying article:

> Agrawal, V., Nilsson, F., and Lindström, S. B. (2026). Panoramic Partial Least Squares
> (Π-PLS): A transparent and parsimonious multivariate regression model with paired latent
> directions. *Computers & Chemical Engineering*, 109913.
> https://doi.org/10.1016/j.compchemeng.2026.109913

PiPLS is developed by Vishal Agrawal, Fritjof Nilsson, and Stefan B. Lindström. The code and
repository-authored documentation are distributed under the [BSD 3-Clause License](LICENSE);
included reference datasets retain their own licenses and attribution notices.

See the [citation and licensing guide](docs/citation.md) for the recommended software citation and
[`CITATION.cff`](CITATION.cff) for machine-readable metadata.
