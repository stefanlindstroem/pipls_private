# Examples

Pi-PLS predicts through paired latent modes, each containing one predictor direction, one response
direction, and one dilation. Applied workflows normally scan `n_components`, the number of paired
latent modes, by cross-validation and inspect CV-MSE against that count.
The resulting table or curve is called the component path. A clear elbow or plateau can motivate a
component count; when no clear elbow is present, explicit relative tolerances provide a transparent
parsimony policy. Example 07 demonstrates separate 10% predictor-rank and component-count
tolerances. See the [component-path discussion](../docs/path_analysis.md#search-owned-selection-rules)
and the [served example catalogue](../docs/examples.md#tobacco-two-relative-tolerance-decisions).
For the mathematical construction, see `docs/theory.md`.

The examples are arranged by user task rather than by implementation complexity. Each numbered
script is self-contained: it explains its data, purpose, and printed or written results without
assuming familiarity with a publication. Start with the package-owned Pulp quick start, then move
to synthetic data, explicit comparison, or the complete real-data workflows. The complete
workflows are intentionally more extensive than ordinary estimator use.

## Start here

- `01_pulp_quick_start.py`: the shortest installed-data workflow. It loads Pulp through
  `load_pulp()`, evaluates the default component path, applies the minimum-CV-MSE rule through
  `refit()`, and combines all standardized observed and fitted responses in one plot. The plotted
  values describe full-data calibration fit, not OOF validation.
- `02_synthetic_path_selection.py`: the short manual-selection workflow. It generates independent
  synthetic train/test data, completes search and refitting for the declared component count, then
  inspects `model.selection_`, the component path, and the conditional predictor-rank profile before
  writing three final PDF figures.
- `03_leave_one_out_validation.py`: a focused small-sample calibration workflow. It evaluates a
  compact explicit path with `LeaveOneOut`, obtains one fitting-free selection through
  `search.select(rule="best_score")`, evaluates it through `search.oof_report()`, and distinguishes
  pooled OOF $R^2$ from undefined mean foldwise $R^2$.

Run it with:

```bash
python -m pip install -e ".[examples]"
PYTHONPATH=src MPLBACKEND=Agg python examples/01_pulp_quick_start.py
```

The script writes `examples/results/pulp_quick_start.pdf`. The package loader provides the
matrices and documentary labels, while the plot uses only the standardized numerical diagnostics.

Example 01 intentionally uses the default `cv=5` to keep the first complete search and refit in one
expression. Maintained analytical examples that use one ordinary five-fold regression partition
construct
`KFold(n_splits=5, shuffle=True, random_state=0)` explicitly. Example 05 instead uses
`RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)` for the complete Pulp analysis. The fixed
seed makes these analyses reproducible while preventing row order from defining the folds. Example
03 uses `LeaveOneOut`, which exhaustively holds out each observation and therefore has no shuffle
option.

## Explicit comparison

- `04_pls_path_comparison.py`: the explicit Pulp, Sugarcane, and Tobacco Pi-PLS-versus-PLS
  component-path CV-MSE comparisons. Ordinary PLS appears here as a reference model.

Grouped and temporal validation require application-specific sampling semantics and remain in
`docs/path_analysis.md`. Example 03 gives leave-one-out validation one concrete small-calibration
use case rather than combining unrelated split protocols in one context-free script.

## Complete Pi-PLS reference workflows

- `05_pulp_real_data.py`: the direct canonical tutorial analysis. It completes search and manual
  refitting at three components, retrieves the fitted model's `selection_`, component path, and
  conditional predictor-rank profile, requests selection-conditioned OOF predictions through
  `search.oof_report(..., selection=selection)`, orients the displayed factors so the tensile-index
  response is positive, and writes six final PDF figures directly from in-memory results.
- `06_sugarcane_real_data.py`: the direct reference workflow. It completes modeling before
  retrieving `model.selection_`, the component path, the conditional predictor-rank profile, and an
  OOF report for that selection. It writes six wavelength-aware final PDF figures without generated
  analytical CSV files.
- `07_tobacco_real_data.py`: adaptive Pi-PLS predictor-rank scanning with explicit full predictor
  SVD and two separately named 10% relative tolerances. The search constructor applies the
  predictor-rank tolerance independently at each component count; `refit()` then applies the
  component-count tolerance to the conditioned path. The workflow obtains exact and retained rank
  evidence, the component-path reference minimum, and a selection-driven OOF report before
  rendering both thresholds, decreasing-wavenumber spectral displays, deterministic response
  pagination, and raw observation diagnostics through caller-owned PDFs. See the
  [selection rules](../docs/path_analysis.md#search-owned-selection-rules) and the
  [focused Tobacco explanation](../docs/examples.md#tobacco-two-relative-tolerance-decisions).

These are application analyses rather than introductory snippets. Pulp, Sugarcane, and Tobacco
complete search and full-data refitting before retrieving the fitted selection, retained path
evidence, conditional rank profile, optional OOF diagnostics, immutable fitted-model inspection
results, and caller-owned Matplotlib composition. Tobacco replaces the manual component-count
choice with the two explicit 10% relative-tolerance decisions described above.
Pulp, Sugarcane, and Tobacco each write `component_path.pdf`,
`predictor_rank_profile.pdf`, `pipls_factors.pdf`, `latent_structure.pdf`, `coefficients.pdf`, and
`prediction_diagnostics.pdf`. For Tobacco,
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

Pulp, Sugarcane, and Tobacco are package-owned, and every maintained consumer uses the
corresponding named loader:

```python
data = load_sugarcane()
X, Y = data.X, data.Y
wavelengths = np.asarray(data.feature_names, dtype=np.float64)
response_names = list(data.target_names)
```

The Tobacco workflow uses the same pattern for its decreasing wavenumber axis. Users whose arrays do
not carry labels can supply equivalent coordinates and names from a schema, laboratory information
system, or other domain metadata. The package inspection API does not invent scientific variable
names.

Example 04 keeps the Pi-PLS and ordinary PLS paths in memory and creates the three overlaid
comparison figures directly. Sugarcane demonstrates the complete manual-analysis workflow:

1. `PiPLSSearchCV(cv=CV).fit(X, Y)` evaluates the path.
2. `search.refit(..., n_components=CHOSEN_N_COMPONENTS)` fits the declared component count and its
   conditionally selected predictor rank on all observations.
3. `selection = model.selection_` retrieves the exact row used by the fitted model.
4. `component_path_` and `predictor_rank_profile(selection.n_components)` provide the retained
   selection evidence.
5. `search.oof_report(X, Y, selection=selection)` reuses the exact seeded shuffled folds and returns
   selection-conditioned OOF predictions for that fitted specification.
6. `pipls_display_factors()`, `latent_structure()`, and `prediction_diagnostics()` return immutable
   in-memory results.
7. The script renders the completed path, rank-profile, latent-structure, prediction-diagnostic, and
   factor results with Matplotlib and saves the six final figures itself.

Pulp is the canonical tutorial workflow. Example 05 follows the same ordering: search and refitting
complete modeling, `model.selection_` identifies the fitted row, the search supplies path and rank
profile evidence, `oof_report()` averages ten predictions per observation across the 50 stored
splits, and rendering occurs only after the numerical analysis is complete. Tobacco follows the same
direct
result-to-Matplotlib pattern but applies the named `"minimum_cv_mse"` refit rule with a 10%
relative tolerance. Its `model.selection_` supplies the selected row, exact reference minimum,
resolved tolerance, and threshold used by the component-path figure; the selected count then supplies
`predictor_rank_profile()`. It owns its
full-SVD configuration, response pagination, and multipage PDF output visibly.

Full-data factor, score, loading, and coefficient figures are interpretive. Prediction and residual
figures retain explicit provenance. Numbered examples never serialize analytical results for later
plotting and do not read repository CSV files. Sugarcane derives its strictly increasing wavelength
coordinate from `data.feature_names`. Tobacco derives its decreasing wavenumber coordinate and all
thirteen source-order response names from the same immutable dataset result. Generated files under
`examples/results/` are ignored by Git.

## Output directories

`examples/results/`, the synthetic-tutorial and PLS-path-comparison directories, and the Pulp,
Sugarcane, and Tobacco post-analysis subdirectories are tracked with placeholder files and shipped
in the source distribution. The examples therefore write directly to known destinations and do not
contain directory-creation code. Generated PDF files remain ignored, and `make clean` removes them
while preserving the tracked directory structure.
