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
- `02_synthetic_path_selection.py`: the short tutorial workflow. It generates independent synthetic
  train/test data, evaluates the component path and conditional predictor-rank profile, fits one
  selected fixed model, and writes three final PDF figures.

Run it with:

```bash
python -m pip install -e ".[examples]"
PYTHONPATH=src MPLBACKEND=Agg python examples/01_minimal_fit_and_plot.py
```

The script writes `examples/results/minimal_fit_and_plot.pdf`. Its predictor and response names are
ordinary Python lists, demonstrating that plotting labels may come from any explicit metadata
source rather than from pandas or CSV headers.

## Explicit comparison

- `09_pls_path_comparison.py`: the explicit Pulp, Sugarcane, and Tobacco Pi-PLS-versus-PLS
  component-path CV-MSE comparisons. Ordinary PLS appears here as a reference model.

Grouped, leave-one-out, and temporal validation require application-specific sampling semantics.
They are documented in `docs/cross_validation.md` rather than combined into a context-free numbered
example.

## Complete Pi-PLS reference workflows

- `10_pulp_real_data.py`: the direct canonical tutorial analysis. It evaluates the path, inspects
  the conditional predictor-rank profile at three components, fits the selected fixed model,
  calculates selection-conditioned OOF predictions, and writes six final PDF figures directly from
  in-memory results.
- `11_sugarcane_real_data.py`: the direct reference workflow. It reads the component path and
  inspection results in memory, calculates OOF predictions with scikit-learn, and writes five
  wavelength-aware final PDF figures without generated analytical CSV files.
- `12_tobacco_real_data.py`: adaptive Pi-PLS predictor-rank scanning with explicit full predictor
  SVD, one selected Pi-PLS interpretation model, selection-conditioned Pi-PLS OOF predictions,
  decreasing-wavenumber spectral plots, deterministic response pagination, and raw observation
  diagnostics through direct in-memory results and caller-owned multipage PDFs.

These are application analyses rather than introductory snippets. Pulp and Sugarcane expose their
complete scientific sequences directly in the numbered scripts: path evaluation and plotting,
fixed fitting, OOF prediction, immutable inspection results, and explicit Matplotlib composition.
Pulp writes `component_path.pdf`, `predictor_rank_profile.pdf`, `pipls_factors.pdf`,
`latent_structure.pdf`, `coefficients.pdf`, and `prediction_diagnostics.pdf`. Sugarcane writes five
corresponding figures without a predictor-rank-profile page. Tobacco also writes five final PDFs;
`prediction_diagnostics.pdf` and `coefficients.pdf` each contain three source-order response pages.
`make examples` runs every numbered example in filename order, including the slower
real-data workflows. It remains separate from `make check`.

## Example support module

The comparison workflow imports one implementation helper from `examples/_support/`:

- `pls_component_path.py`: immutable ordinary-PLS path evaluation for example 09.

Example 09 owns the Matplotlib comparison figures directly. Pulp, Sugarcane, and Tobacco import no
comparison helper. Reusable numerical inspection belongs in `pipls.inspection`; every maintained
figure is rendered directly from immutable arrays with ordinary Matplotlib.

## Real-data workflow contract

The Pulp, Sugarcane, and Tobacco examples show label acquisition as a separate I/O step:

```python
X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")
response_names = Y.columns.tolist()
```

Users whose arrays do not carry column headers can supply equivalent lists from a schema, laboratory
information system, or other domain metadata. The package inspection API does not read files or invent scientific variable names. The committed datasets already have tested headers and ordering,
so the numbered examples use them directly instead of repeating repository-integrity checks.

Example 09 keeps the Pi-PLS and ordinary PLS paths in memory and creates the three overlaid
comparison figures directly. Sugarcane demonstrates the complete-analysis workflow:

1. `PiPLSPathCV(refit=False)` returns `component_path_`, which is plotted directly with Matplotlib.
2. `path.for_n_components(CHOSEN_N_COMPONENTS)` supplies the fixed component count and predictor
   rank.
3. `cross_val_predict()` with five non-shuffled folds produces
   `selection-conditioned OOF predictions`.
4. `pipls_display_factors()`, `latent_structure()`, and `prediction_diagnostics()` return
   immutable in-memory results.
5. The script plots latent structure, prediction diagnostics, and $P$, $D$, $Q$, and $QD$ factors
   directly with Matplotlib and saves the five final figures itself.

Pulp is the canonical tutorial workflow. Example 10 performs the same direct analysis shown in
the tutorial: it uses `component_path_`, retrieves the immutable conditional rank profile with
`predictor_rank_profile()`, fits one explicit `PiPLSRegression`, calculates OOF predictions with
`cross_val_predict()`, and renders immutable inspection arrays directly. Tobacco follows the same direct pattern
and owns its full-SVD configuration, response pagination, and multipage PDF output visibly.

Full-data factor, score, loading, and coefficient figures are interpretive. Prediction and residual
figures retain explicit provenance. Numbered examples never serialize analytical results for later
plotting; CSV use is limited to the committed `X.csv` and `Y.csv` inputs. Sugarcane reads its
strictly increasing wavelength coordinate
from the `X.csv` headers. Tobacco preserves its decreasing wavenumber coordinate and partitions all
thirteen responses in source order. Generated files under `examples/results/` are ignored by Git.

## Output directories

`examples/results/`, the synthetic-tutorial and PLS-path-comparison directories, and the Pulp,
Sugarcane, and Tobacco post-analysis subdirectories are tracked with placeholder files. The examples therefore write directly to known destinations and do not
contain directory-creation code. Generated PDF files remain ignored,
and `make clean`
removes them while preserving the tracked directory structure.
