# Inspection API

`pipls.inspection` contains pure numerical analysis helpers. It has no pandas or Matplotlib
requirement, returns immutable defensive copies, and performs no file writing. Result records are
obtained from these helpers; the reference emphasizes their fields rather than manual construction.
Direct construction and pickle reconstruction nevertheless validate the same finite-value, shape,
relationship, and read-only invariants. Finite inputs either produce finite float64 inspection
quantities or raise a clear `ValueError` when a requested derived value is not representable.

The Pi-PLS factorization helper is method-specific. Its $\mathbf{P}$ and $\mathbf{Q}$ arrays
are orthonormal predictor and response directions, its entries $d_k=D_{kk}$ are dilations, and
$(P_{:k},d_k,Q_{:k})$ defines paired latent mode $k$. The latent-structure, biplot, observation,
and prediction helpers use PLS-family quantities and can therefore be applied to a compatible fitted
Pi-PLS or ordinary PLS estimator. For interpretation, scientific limitations, and relationships
between these quantities, begin with [Model inspection concepts](../model_inspection.md).

## Result objects and labels

::: pipls.inspection.PredictionKind
    options:
      members: false

::: pipls.inspection.BiplotCoordinates
    options:
      show_signature: false
      members:
        - n_samples
        - n_features

::: pipls.inspection.LatentStructure
    options:
      show_signature: false
      members:
        - n_samples
        - n_features
        - n_targets
        - n_components

::: pipls.inspection.ObservationDiagnostics
    options:
      show_signature: false
      members:
        - n_samples

::: pipls.inspection.PiPLSDisplayFactors
    options:
      show_signature: false
      members:
        - n_features
        - n_targets
        - n_components

::: pipls.inspection.PredictionDiagnostics
    options:
      show_signature: false
      members:
        - n_samples
        - n_targets

## PLS-family analysis

::: pipls.inspection.latent_structure
    options:
      members: false

::: pipls.inspection.biplot_coordinates
    options:
      members: false

::: pipls.inspection.observation_diagnostics
    options:
      members: false

::: pipls.inspection.prediction_diagnostics
    options:
      members: false

## Pi-PLS factorization display

::: pipls.inspection.pipls_display_factors
    options:
      members: false
