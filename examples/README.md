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

- `10_pulp_real_data.py`: direct pandas reading, one Pi-PLS component path, one selected Pi-PLS
  interpretation model, selection-conditioned Pi-PLS OOF predictions, seven canonical
  post-analysis CSV files, a balanced score-loading biplot, and a multipage report.
- `11_sugarcane_real_data.py`: direct pandas reading, one Pi-PLS component path, one selected
  Pi-PLS interpretation model, selection-conditioned Pi-PLS OOF predictions, seven canonical
  post-analysis CSV files, and a wavelength-aware report.
- `12_tobacco_real_data.py`: adaptive Pi-PLS predictor-rank scanning with explicit full predictor
  SVD, one selected Pi-PLS interpretation model, selection-conditioned Pi-PLS OOF predictions,
  decreasing-wavenumber spectral plots, deterministic response pagination, eight canonical
  post-analysis CSV files, and raw observation diagnostics.

These are application analyses rather than introductory snippets, but the numbered scripts keep
only their scientific stages visible. Their PDF reports compose the separate $P$, $D$, $Q$, and
$QD$ charts in a $2\times2$ factor panel and the three prediction diagnostics in a $1\times3$
panel. A later page groups shared latent-model views: Pulp uses scores, biplot, X loadings, and Y
loadings in a $2\times2$ panel; Sugarcane uses scores and the two loading views in a $1\times3$
panel; Tobacco replaces the biplot with observation diagnostics in a $2\times2$ panel.
Coefficient curves retain full-width response pages. Reusable validation, CSV reconstruction, and
report composition remain in `_support`. `make examples` runs every numbered example in filename
order, including the slower real-data workflows. It remains separate from `make check`.

## Example support modules

The complete workflows import implementation support from `examples/_support/`:

- `pls_component_path.py`: ordinary-PLS path evaluation;
- `plot_component_path.py`: component-path CSV-to-PDF rendering;
- `fixed_model_oof.py`: cloning and aligned OOF prediction for already fixed models;
- `post_analysis_artifacts.py`: canonical table construction, CSV round trips, pagination, and
  multipage report composition.

The underscore-prefixed directory marks these files as support for the complete examples, not as
the shortest route to fitting Pi-PLS. They remain example-owned because they contain pandas I/O,
fixed-model OOF orchestration, physical-axis handling, and report composition. Reusable numerical
inspection belongs in `pipls.inspection`, and optional rendering belongs in `pipls.plotting`.

## Real-data workflow contract

The Pulp, Sugarcane, and Tobacco examples show label acquisition as a separate I/O step:

```python
X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")
predictor_names = X.columns.tolist()
response_names = Y.columns.tolist()
```

Users whose arrays do not carry column headers can supply equivalent lists from a schema, laboratory
information system, or other domain metadata. The package plotting API does not read files or
invent scientific variable names. The committed datasets already have tested headers and ordering,
so the numbered examples use them directly instead of repeating repository-integrity checks.

Example 09 writes the separate Pi-PLS and ordinary PLS path tables used by the three overlaid
comparison figures. Examples 10–12 instead share this normal Pi-PLS workflow:

1. `PiPLSPathCV(refit=False)` produces one Pi-PLS row per admissible component count.
   `component_path.csv` is written before `component_path.pdf` is rendered from it.
2. A visible component-count choice selects one fixed full-data Pi-PLS model for interpretation.
3. The selected Pi-PLS parameters are cloned inside five non-shuffled folds to produce
   `selection-conditioned OOF predictions`. Seven common long-form CSV files are written and
   reread before report generation.

Each dataset keeps `component_path.pdf` and `post_analysis.pdf` as separate files in the same
analysis directory.

Full-data decomposition, score, loading, coefficient, biplot, and observation-diagnostic figures
are interpretive. Prediction and residual figures retain explicit provenance. The display-standardized
columns in `predictions.csv` use the complete observed-response matrix and do not reproduce the
fold-local scaling used by the component-path loss.

Pulp reconstructs its biplot from `x_scores.csv` and `x_loadings.csv`. Sugarcane reads its
strictly increasing wavelength coordinate from `X.csv`. Tobacco preserves its decreasing
wavenumber coordinate, partitions all thirteen responses in source order, and adds
`observation_diagnostics.csv`. Generated files under `examples/results/` are ignored by Git.

## Output directories

`examples/results/`, the PLS-path-comparison directory, and the Pulp, Sugarcane, and Tobacco
post-analysis subdirectories are tracked with placeholder files. The examples therefore write directly to known destinations and do not
contain directory-creation code. Generated CSV and PDF files remain ignored, and `make clean`
removes them while preserving the tracked directory structure.
