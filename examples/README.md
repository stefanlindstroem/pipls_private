# Examples

Pi-PLS predicts through paired predictor and response latent variables. Applied workflows normally
scan the number of latent components by cross-validation and inspect CV-MSE against component count.
The resulting table or curve is called the component path; a common choice is an elbow or plateau
where further components add little improvement. For the mathematical construction, see
`docs/theory.md`.

The examples are arranged by user task rather than by implementation complexity. Each numbered
script is self-contained: it explains its data, purpose, and printed or written results without
assuming familiarity with a publication. Start with the literal-matrix fit, then move to synthetic
data, explicit comparison, or the complete real-data workflows. The complete workflows are
intentionally more extensive than ordinary estimator use.

## Start here

- `01_minimal_fit_and_plot.py`: literal NumPy matrices, one fixed `PiPLSRegression` fit, predictions,
  and one caller-composed panel of the $P$, $D$, $Q$, and $QD$ factor plots. It performs no
  cross-validation or parameter selection.

Run it with:

```bash
python -m pip install -e ".[examples]"
PYTHONPATH=src MPLBACKEND=Agg python examples/01_minimal_fit_and_plot.py
```

The script writes `examples/results/minimal_fit_and_plot.pdf`. Its predictor and response names are
ordinary Python lists, demonstrating that plotting labels may come from any explicit metadata
source rather than from pandas or CSV headers.

## Synthetic data and explicit comparison

- `08_synthetic_data.py`: generates an independent train/test problem with known shared,
  predictor-specific, and response-specific latent directions; fits one Pi-PLS model; and prints
  labeled data, model, and held-out evaluation summaries.
- `09_pls_path_comparison.py`: the explicit Pulp, Sugarcane, and Tobacco Pi-PLS-versus-PLS
  component-path CV-MSE comparisons. Ordinary PLS appears here as a reference model.

Grouped, leave-one-out, and temporal validation require application-specific sampling semantics.
They are documented in `docs/cross_validation.md` rather than combined into a context-free numbered
example.

## Complete Pi-PLS reference workflows

- `10_pulp_real_data.py`: the canonical tutorial analysis, using one terminal-Pi-PLS pipeline
  for component-path evaluation, one selected fixed pipeline, selection-conditioned OOF
  predictions, a balanced score-loading biplot, and the current transitional CSV/report layer.
- `11_sugarcane_real_data.py`: the direct reference workflow. It reads the component path and
  inspection results in memory, calculates OOF predictions with scikit-learn, and writes five
  wavelength-aware final PDF figures without generated analytical CSV files.
- `12_tobacco_real_data.py`: adaptive Pi-PLS predictor-rank scanning with explicit full predictor
  SVD, one selected Pi-PLS interpretation model, selection-conditioned Pi-PLS OOF predictions,
  decreasing-wavenumber spectral plots, deterministic response pagination, and raw observation
  diagnostics through the current transitional CSV/report layer.

These are application analyses rather than introductory snippets. Sugarcane exposes its complete
scientific sequence directly in the numbered script: path evaluation and plotting, fixed fitting,
OOF prediction, immutable inspection results, and explicit Matplotlib composition. Its five figures
are `component_path.pdf`, `pipls_factors.pdf`, `prediction_diagnostics.pdf`,
`latent_structure.pdf`, and `coefficients.pdf`. Pulp and Tobacco retain shared report support until
their scheduled Phase F4 migrations. `make examples` runs every numbered example in filename order,
including the slower real-data workflows. It remains separate from `make check`.

## Example support modules

The remaining transitional workflows import implementation support from `examples/_support/`:

- `pls_component_path.py`: ordinary-PLS path evaluation;
- `plot_component_path.py`: component-path CSV-to-PDF rendering for example 09, Pulp, and Tobacco;
- `fixed_model_oof.py`: cloning and aligned OOF prediction for fixed models that have not migrated
  to direct scikit-learn orchestration;
- `post_analysis_artifacts.py`: canonical table construction, CSV round trips, pagination, and
  multipage report composition for Pulp and Tobacco;
- `pulp_workflow.py`: the shared Pulp pipeline, path, fixed-fit, OOF, and inspection workflow used
  by example 10 and the tutorial assets.

Sugarcane imports none of these helpers. Reusable numerical inspection belongs in
`pipls.inspection`, and optional one-axis rendering belongs in `pipls.plotting`.

## Real-data workflow contract

The Pulp, Sugarcane, and Tobacco examples show label acquisition as a separate I/O step:

```python
X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")
response_names = Y.columns.tolist()
```

Users whose arrays do not carry column headers can supply equivalent lists from a schema, laboratory
information system, or other domain metadata. The package plotting API does not read files or
invent scientific variable names. The committed datasets already have tested headers and ordering,
so the numbered examples use them directly instead of repeating repository-integrity checks.

Example 09 writes the separate Pi-PLS and ordinary PLS path tables used by the three overlaid
comparison figures. Sugarcane demonstrates the target direct workflow:

1. `PiPLSPathCV(refit=False)` returns `component_path_`, which is plotted directly with Matplotlib.
2. `path.for_n_components(CHOSEN_N_COMPONENTS)` supplies the fixed component count and predictor
   rank.
3. `cross_val_predict()` with five non-shuffled folds produces
   `selection-conditioned OOF predictions`.
4. `pipls_display_factors()`, `latent_structure()`, and `prediction_diagnostics()` return the
   in-memory objects used by the public one-axis plotters.
5. The script creates and saves the five final figures itself.

Pulp is the canonical tutorial workflow. Its support module evaluates a one-step scikit-learn
pipeline ending in `PiPLSRegression`, transfers the selected pair with nested pipeline parameters,
and supplies the fitted estimator and inspection results to example 10. Tobacco retains direct
fixed-estimator orchestration plus response pagination. Their current CSV round trips remain
transitional and are not the reference pattern for new examples.

Full-data factor, score, loading, and coefficient figures are interpretive. Prediction and residual
figures retain explicit provenance. Sugarcane reads its strictly increasing wavelength coordinate
from the `X.csv` headers. Tobacco preserves its decreasing wavenumber coordinate and partitions all
thirteen responses in source order. Generated files under `examples/results/` are ignored by Git.

## Output directories

`examples/results/`, the PLS-path-comparison directory, and the Pulp, Sugarcane, and Tobacco
post-analysis subdirectories are tracked with placeholder files. The examples therefore write directly to known destinations and do not
contain directory-creation code. Generated CSV and PDF files remain ignored, and `make clean`
removes them while preserving the tracked directory structure.
