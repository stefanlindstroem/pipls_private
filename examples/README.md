# Examples

Pi-PLS predicts through paired latent modes, each containing one predictor direction, one response
direction, and one dilation. Applied workflows normally scan `n_components`, the number of paired
latent modes, by cross-validation and inspect CV-MSE against that count.
The resulting table or curve is called the component path. A clear elbow or plateau can motivate a
component count; when no clear elbow is present, the conventional one-standard-error rule provides a
reproducible parsimony heuristic. Example 07 demonstrates that rule. See the
[component-path discussion](../docs/path_analysis.md#one-standard-error-component-heuristic) and the
[served example catalogue](../docs/examples.md#tobacco-one-standard-error-selection). For the
mathematical construction, see `docs/theory.md`.

The examples are arranged by user task rather than by implementation complexity. Each numbered
script is self-contained: it explains its data, purpose, and printed or written results without
assuming familiarity with a publication. Start with the package-owned Pulp quick start, then move
to synthetic data, explicit comparison, or the complete real-data workflows. The complete
workflows are intentionally more extensive than ordinary estimator use.

## Start here

- `01_pulp_quick_start.py`: the shortest installed-data workflow. It loads Pulp through
  `load_pulp()`, evaluates the default component path, applies the one-standard-error rule through
  `refit()`, and combines all standardized observed and fitted responses in one plot. The plotted
  values describe full-data calibration fit, not OOF validation.
- `02_synthetic_path_selection.py`: the short tutorial workflow. It generates independent synthetic
  train/test data, evaluates the component path and conditional predictor-rank profile, fits one
  selected fixed model, and writes three final PDF figures.
- `03_leave_one_out_validation.py`: a focused small-sample calibration workflow. It evaluates a
  compact explicit path with `LeaveOneOut`, requests ordered OOF predictions through
  `search.validation_report()`, and distinguishes
  pooled OOF $R^2$ from undefined mean foldwise $R^2$.

Run it with:

```bash
python -m pip install -e ".[examples]"
PYTHONPATH=src MPLBACKEND=Agg python examples/01_pulp_quick_start.py
```

The script writes `examples/results/pulp_quick_start.pdf`. The package loader provides the
matrices and documentary labels, while the plot uses only the standardized numerical diagnostics.

Example 01 intentionally uses the default `cv=5` to keep the first complete search and refit in one
expression. Maintained analytical examples that use ordinary five-fold regression CV construct
`KFold(n_splits=5, shuffle=True, random_state=0)` explicitly. The fixed seed makes those analyses
reproducible while preventing row order from defining the folds. Example 03 uses `LeaveOneOut`,
which exhaustively holds out each observation and therefore has no shuffle option.

## Explicit comparison

- `04_pls_path_comparison.py`: the explicit Pulp, Sugarcane, and Tobacco Pi-PLS-versus-PLS
  component-path CV-MSE comparisons. Ordinary PLS appears here as a reference model.

Grouped and temporal validation require application-specific sampling semantics and remain in
`docs/path_analysis.md`. Example 03 gives leave-one-out validation one concrete small-calibration
use case rather than combining unrelated split protocols in one context-free script.

## Complete Pi-PLS reference workflows

- `05_pulp_real_data.py`: the direct canonical tutorial analysis. It evaluates the path, inspects
  the conditional predictor-rank profile at three components, fits the selected fixed model,
  requests selection-conditioned OOF predictions through `search.validation_report()`, orients
  the displayed factors so the tensile-index response is positive, and writes six final PDF
  figures directly from in-memory results.
- `06_sugarcane_real_data.py`: the direct reference workflow. It reads the component path and
  inspection results in memory, requests OOF predictions through the search validation report,
  and writes five wavelength-aware final PDF figures without generated analytical CSV files.
- `07_tobacco_real_data.py`: adaptive Pi-PLS predictor-rank scanning with explicit full predictor
  SVD and explicit `search.select(rule=...)` calls. Its component-path figure shows
  the minimum-CV-MSE row, the horizontal 1-SE threshold, and the recommended row;
  `search.refit(..., rule="one_standard_error")` fits that row without manual parameter
  transfer. The workflow
  then requests selection-conditioned OOF predictions through the same named rule, produces
  decreasing-wavenumber spectral plots,
  deterministic response pagination, and raw observation diagnostics through direct in-memory
  results and caller-owned multipage PDFs. See the
  [1-SE rule](../docs/path_analysis.md#one-standard-error-component-heuristic) and the
  [focused Tobacco explanation](../docs/examples.md#tobacco-one-standard-error-selection).

These are application analyses rather than introductory snippets. Pulp and Sugarcane expose their
complete scientific sequences directly in the numbered scripts: path evaluation and plotting,
explicit component-count choice, fixed fitting, OOF prediction, immutable inspection results, and
explicit Matplotlib composition. Tobacco replaces the manual component-count choice with the
explicit 1-SE recommendation described above while retaining the same separate fixed-fit boundary.
Pulp writes `component_path.pdf`, `predictor_rank_profile.pdf`, `pipls_factors.pdf`,
`latent_structure.pdf`, `coefficients.pdf`, and `prediction_diagnostics.pdf`. Sugarcane writes five
corresponding figures without a predictor-rank-profile page. Tobacco also writes five final PDFs;
`prediction_diagnostics.pdf` and `coefficients.pdf` each contain three source-order response pages.
`make examples` runs every numbered example in filename order, including the slower
real-data workflows. It remains separate from `make check`.

## Example support module

The comparison workflow imports one implementation helper from `examples/_support/`:

- `pls_component_path.py`: immutable ordinary-PLS path evaluation for example 04.

Example 04 owns the Matplotlib comparison figures directly. Pulp, Sugarcane, and Tobacco import no
comparison helper. Reusable numerical inspection belongs in `pipls.inspection`; every maintained
figure is rendered directly from immutable arrays with ordinary Matplotlib. The Pulp biplot
uses optional `adjustText` only to reposition its Matplotlib text labels.

## Real-data workflow contract

Pulp is package-owned and all maintained Pulp consumers use its named loader:

```python
data = load_pulp()
X, Y = data.data, data.target
response_names = data.target_names
```

Sugarcane and Tobacco continue to show label acquisition through explicit repository CSV reading.
Users whose arrays do not carry column headers can supply equivalent lists from a schema, laboratory
information system, or other domain metadata. The package inspection API does not invent scientific
variable names.

Example 04 keeps the Pi-PLS and ordinary PLS paths in memory and creates the three overlaid
comparison figures directly. Sugarcane demonstrates the complete-analysis workflow:

1. the default path-evaluating `PiPLSSearchCV()` returns `component_path_`, which is plotted directly
   with Matplotlib.
2. `search.refit(..., n_components=CHOSEN_N_COMPONENTS)` resolves the stored predictor rank
   and fits the chosen path row on all observations.
3. `search.validation_report(..., n_components=CHOSEN_N_COMPONENTS)` reuses the exact
   seeded shuffled folds stored by the search and returns `selection-conditioned OOF predictions`.
4. `pipls_display_factors()`, `latent_structure()`, and `prediction_diagnostics()` return
   immutable in-memory results; the Pulp factor call anchors component signs to positive `TI`
   entries.
5. The script plots latent structure, prediction diagnostics, and $P$, $D$, $Q$, and $QD$ factors
   directly with Matplotlib and saves the five final figures itself.

Pulp is the canonical tutorial workflow. Example 05 performs the same direct analysis shown in
the tutorial: it uses `component_path_`, retrieves the immutable conditional rank profile with
`predictor_rank_profile()`, fits the chosen row through `search.refit()`, obtains OOF
predictions through `search.validation_report()`, and renders immutable inspection arrays
directly. Tobacco
follows the same direct result-to-Matplotlib pattern, but applies the named
`"one_standard_error"` refit rule. It uses `search.select()` with the
`"minimum_cv_mse"` and `"one_standard_error"` rules to construct the explanatory
component-path figure and owns its full-SVD configuration, response pagination, and multipage PDF
output visibly.

Full-data factor, score, loading, and coefficient figures are interpretive. Prediction and residual
figures retain explicit provenance. Numbered examples never serialize analytical results for later
plotting; CSV use is limited to the committed `X.csv` and `Y.csv` inputs. Sugarcane reads its
strictly increasing wavelength coordinate
from the `X.csv` headers. Tobacco preserves its decreasing wavenumber coordinate and partitions all
thirteen responses in source order. Generated files under `examples/results/` are ignored by Git.

## Output directories

`examples/results/`, the synthetic-tutorial and PLS-path-comparison directories, and the Pulp,
Sugarcane, and Tobacco post-analysis subdirectories are tracked with placeholder files and shipped
in the source distribution. The examples therefore write directly to known destinations and do not
contain directory-creation code. Generated PDF files remain ignored, and `make clean` removes them
while preserving the tracked directory structure.
