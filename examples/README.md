# Examples

For ordinary use, Π-PLS follows the familiar PLS component-selection workflow: scan
`n_components` by cross-validation and inspect prediction error against that count. The resulting
table or curve is the component path. Π-PLS resolves its retained predictor rank conditionally for
each component count, so most workflows can treat `n_components` as the main complexity parameter.
A clear elbow or plateau can motivate a component count; when no clear elbow is present, explicit
relative tolerances provide a transparent parsimony policy. Example 06 demonstrates separate 10%
predictor-rank and component-count tolerances. See the
[component-path discussion](../docs/path_analysis.md#search-owned-selection-rules) and the
[served example catalogue](../docs/examples.md#tobacco-two-relative-tolerance-decisions). For the
mathematical construction, see `docs/theory.md`.

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
  synthetic train/test data, evaluates and first inspects the unselected component path, chooses a
  component count and creates one selection, inspects the selected path and conditional
  predictor-rank profile, refits that selection, and writes four final PDF figures including
  independent-test prediction diagnostics.

Run the quick start with:

```bash
python -m pip install -e ".[examples]"
PYTHONPATH=src MPLBACKEND=Agg python examples/01_pulp_quick_start.py
```

The script writes `examples/results/pulp_quick_start.pdf`. The package loader provides the
matrices and documentary labels, while the plot uses only the standardized numerical diagnostics.

Example 01 intentionally uses the default `cv=5` to keep the first complete search and refit in one
expression. Maintained analytical examples that use one ordinary five-fold regression partition
construct `KFold(n_splits=5, shuffle=True, random_state=0)` explicitly. Example 04 instead uses
`RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)` for the complete Pulp analysis. The fixed
seed makes these analyses reproducible while preventing row order from defining the folds.

## Explicit comparison

- `03_pls_path_comparison.py`: the explicit Pulp, Sugarcane, and Tobacco PLS-family component-path
  comparison. For each dataset it materializes one shuffled five-fold protocol and reuses those
  exact splits for separate Π-PLS searches with `response_subspace="cross_covariance"` and
  `response_subspace="least_squares"`, plus ordinary PLS. Pulp uses the exhaustive predictor-rank
  default; the higher-dimensional Sugarcane and Tobacco searches request adaptive coverage
  explicitly. Cross-covariance is the peer-reviewed Π-PLS default, whereas least squares is an
  RRR-inspired software extension outside the peer-reviewed publication. The workflow fits no
  final model.

Grouped and temporal validation require application-specific sampling semantics and remain in
`docs/path_analysis.md`.

## Complete Pi-PLS reference workflows

- `04_pulp_real_data.py`: the direct canonical tutorial analysis. It evaluates the repeated-CV path,
  first inspects the unselected component path, creates the declared three-component selection,
  inspects its selected path, rank evidence, and selection-conditioned OOF predictions, refits the
  exact same selection, orients the displayed factors so the tensile-index response is positive,
  and writes ten final PDF figures directly from in-memory results.
- `05_sugarcane_real_data.py`: the direct reference workflow. It requests adaptive predictor-rank
  coverage for the high-dimensional spectral search, creates one manual selection, inspects its
  conditional predictor-rank evidence, computes an OOF report, refits that selection, and writes six
  wavelength-aware final PDF figures.
- `06_tobacco_real_data.py`: adaptive Π-PLS predictor-rank scanning with explicit full predictor
  SVD and two separately named 10% relative tolerances. The search constructor applies the
  predictor-rank tolerance independently at each component count; `search.select()` then applies the
  component-count tolerance to the conditioned path. The workflow obtains exact and retained rank
  evidence, the component-path reference minimum, and a selection-conditioned OOF report before it
  refits the same selection and renders both thresholds, decreasing-wavenumber spectral displays,
  deterministic response pagination, and raw observation diagnostics through caller-owned PDFs.
  See the [selection rules](../docs/path_analysis.md#search-owned-selection-rules) and the
  [focused Tobacco explanation](../docs/examples.md#tobacco-two-relative-tolerance-decisions).

These are application analyses rather than introductory snippets. Pulp, Sugarcane, and Tobacco
inspect the component path, create one immutable selection, inspect its selected path and optional
OOF evidence, refit the same row on all observations, calculate immutable fitted-model inspection
results, and only then compose figures. Tobacco replaces the manual component-count choice with the
two explicit 10% relative-tolerance decisions described above. Sugarcane and Tobacco each write
`component_path.pdf`, `predictor_rank_profile.pdf`, `pipls_factors.pdf`,
`latent_structure.pdf`, `coefficients.pdf`, and `prediction_diagnostics.pdf`. Pulp writes those
six figures plus `selected_component_path.pdf`, `final_fit_observed_vs_predicted.pdf`,
`final_fit_r2.pdf`, and `final_fit_residual_distribution.pdf`. For Tobacco,
`prediction_diagnostics.pdf` and `coefficients.pdf` each contain three source-order response pages.
`make examples` runs every numbered example in filename order, including the slower real-data
workflows and the three-way PLS-family comparison. It remains separate from `make check`.

## Example support module

The Π-PLS-versus-PLS comparison workflow imports one implementation helper from
`examples/_support/`:

- `pls_component_path.py`: immutable ordinary-PLS path evaluation for example 03.

Example 03 owns the Matplotlib comparison figures directly. Pulp, Sugarcane, and Tobacco import no
comparison helper. Reusable numerical inspection belongs in `pipls.inspection`; every maintained
figure is rendered directly from immutable arrays with ordinary Matplotlib. The Pulp biplot uses
optional `adjustText` only to reposition its Matplotlib text labels.

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

Example 03 keeps both Π-PLS response-policy paths and the ordinary-PLS path in memory and creates
the three overlaid comparison figures directly. Sugarcane demonstrates the complete manual-analysis workflow:

1. `PiPLSSearchCV(search_method="adaptive", cv=CV).fit(X, Y)` evaluates the path with
   adaptive predictor-rank coverage to reduce candidate work for this high-dimensional dataset.
2. `component_path_` provides the unconditional evidence used to choose a component count.
3. `selection = search.select(n_components=CHOSEN_N_COMPONENTS)` records that choice as one complete
   immutable row without fitting.
4. `predictor_rank_profile(selection.n_components)`, `oof_report(...)`, and
   `prediction_diagnostics()` provide selection-conditioned evidence for reviewing that row.
5. `search.refit(X, Y, selection=selection)` fits the accepted component-count and predictor-rank
   pair on all observations and records it as `model.selection_`.
6. `pipls_display_factors()` and `latent_structure()` return immutable fitted-model results.
7. The script renders the completed path, rank-profile, latent-structure, prediction-diagnostic, and
   factor results with Matplotlib and saves the six final figures itself.

Pulp is the canonical tutorial workflow. Example 04 follows the same ordering, with
`oof_report()` averaging ten predictions per observation across the 50 stored splits before the
same selection is refitted. Tobacco applies `search.select(rule="minimum_cv_mse",
relative_tolerance=0.10)` to the conditioned path. That selection supplies the exact reference
minimum, resolved tolerance, threshold, and component count used by both the rank-profile and
component-path figures; the same object is then passed to OOF reporting and final refitting.
Tobacco owns its full-SVD configuration, response pagination, and multipage PDF output visibly.

Full-data factor, score, loading, and coefficient figures are interpretive. Prediction and residual
figures retain explicit provenance. Numbered examples render their figures directly from in-memory
analysis results. Sugarcane derives its strictly increasing wavelength
coordinate from `data.feature_names`. Tobacco derives its decreasing wavenumber coordinate and all
thirteen source-order response names from the same immutable dataset result. Generated files under
`examples/results/` are ignored by Git.

## Output directories

`examples/results/`, the synthetic-tutorial and PLS-path-comparison directories, and the Pulp,
Sugarcane, and Tobacco post-analysis subdirectories are tracked with placeholder files and shipped
in the source distribution. The examples therefore write directly to known destinations and do not
contain directory-creation code. Generated PDF files remain ignored, and `make clean` removes them
while preserving the tracked directory structure.
