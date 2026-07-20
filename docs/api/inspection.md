# Inspection

`pipls.inspection` contains pure numerical analysis helpers. It has no pandas or Matplotlib
requirement, returns immutable defensive copies, and performs no file writing.

The Pi-PLS factorization helper is method-specific. The latent-structure, biplot, observation, and
prediction helpers use PLS-family quantities and can therefore be applied to a compatible fitted
Pi-PLS or ordinary PLS estimator.

## Result objects and labels

::: pipls.inspection.PredictionKind
    options:
      members: false

::: pipls.inspection.BiplotCoordinates
    options:
      members:
        - n_samples
        - n_features

::: pipls.inspection.LatentStructure
    options:
      members:
        - n_samples
        - n_features
        - n_targets
        - n_components

::: pipls.inspection.ObservationDiagnostics
    options:
      members:
        - n_samples

::: pipls.inspection.PiPLSDisplayFactors
    options:
      members:
        - n_features
        - n_targets
        - n_components

::: pipls.inspection.PredictionDiagnostics
    options:
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
