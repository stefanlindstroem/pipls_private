# Examples

The numbered examples are executable workflows organized by programming task. Start with the
package-owned [Pulp quick start](tutorials/quick_start.md), continue with the
[synthetic tutorial](tutorials/synthetic.md) for component-path inspection and independent-test
prediction, then use the
[Pulp tutorial](tutorials/pulp.md) for a complete real-data analysis. This page is a catalogue of
the maintained scripts.

## Choose an example

| Script | Programming task | Main output |
|---|---|---|
| `01_pulp_quick_start.py` | Search, refit, and plot standardized fitted values for package-owned Pulp data | Selected model summary and `pulp_quick_start.pdf` |
| `02_synthetic_path_selection.py` | Fit a declared component count, inspect the retained selection evidence, and evaluate independent test predictions | Three PDF figures and printed external-test $R^2$ |
| `03_leave_one_out_validation.py` | Select one path row without fitting a final model, then evaluate it with leave-one-out OOF reporting | Printed selected rank pair and immutable OOF summary |
| `04_pls_path_comparison.py` | Compare matched Pi-PLS and ordinary PLS component paths | One comparison PDF for each reference dataset |
| `05_pulp_real_data.py` | Fit a manual Pulp model, then inspect its selection, rank profile, OOF behavior, and latent structure | Six PDF figures |
| `06_sugarcane_real_data.py` | Run the complete wavelength-aware Sugarcane workflow | Six PDF figures |
| `07_tobacco_real_data.py` | Apply the one-standard-error rule in a complete Tobacco spectral workflow | Six PDFs, including a conditional rank profile and multipage reports |

The [path-selection reference](api/path.md) documents the search and post-fit refit operation used
by example 01. The [synthetic tutorial](tutorials/synthetic.md) extracts the maintained example 02
workflow directly. The comparison in example 04 is optional and is not part of routine Pi-PLS
fitting.

## Run one example

Install the example dependencies and execute a script from the repository root:

```bash
python -m pip install ".[examples]"
python examples/01_pulp_quick_start.py
```

Generated files are written below `examples/results/`.

## Run all examples

```bash
make examples
```

The complete real-data analyses are intentionally outside `make check` because they are application
workflows and may take substantially longer than the package test suite.


## Cross-validation partitions

Example 01 intentionally uses the scikit-learn-compatible default `cv=5` to keep the opening
workflow to one search/refit expression. Examples 02 and 04–07 use explicit five-fold shuffled
regression splits with `KFold(n_splits=5, shuffle=True, random_state=0)`. The complete Pulp,
Sugarcane, and Tobacco workflows use `oof_report(..., selection=model.selection_)` to reuse the
exact partition materialized by the path search. Example 03 deliberately fits no final model: it
uses `search.select(rule="best_score")` and passes that selection to `oof_report()`. Its
`LeaveOneOut` splitter is exhaustive, so shuffling is not defined. Grouped,
temporal, or otherwise structured data require an application-specific splitter instead.

## Leave-one-out validation

`examples/03_leave_one_out_validation.py` represents a small calibration study with twelve costly
observations. It evaluates a compact explicit Pi-PLS path with `LeaveOneOut`, requests one ordered
OOF prediction per observation, and reports the immutable validation summary. The displayed pooled
OOF $R^2$ is calculated across all held-out predictions; it is not mean foldwise $R^2$, which is
undefined for singleton validation folds. See the
[leave-one-out interpretation](path_analysis.md#leave-one-out-interpretation) for the associated
scoring and selection qualifications.

## Complete real-data analyses

The [dataset documentation](datasets.md) gives the original source, DOI, license, and repository
adaptation for each real-data integration. The three complete analyses evaluate one Pi-PLS
component path and fit one selected fixed model:

- `examples/05_pulp_real_data.py`: the direct tutorial workflow for named scalar predictors and
  responses. Search and refitting complete modeling; `model.selection_`, `component_path_`,
  `predictor_rank_profile()`, `oof_report()`, and immutable inspection results are then calculated
  before the six figures are rendered;
- `examples/06_sugarcane_real_data.py`: the direct reference workflow with the same ordering,
  wavelength-aware inspection, and six final PDF figures;
- `examples/07_tobacco_real_data.py`: a complete spectral workflow that uses the
  [one-standard-error rule](path_analysis.md#one-standard-error-component-heuristic) to recommend
  the final component count, then inspects the conditional predictor-rank profile at that returned
  count before fixed-model inspection. See the focused explanation below.

### Tobacco: one-standard-error selection

The Tobacco component path has no clear elbow that would by itself motivate one component count.
Example 07 therefore demonstrates the conventional
[one-standard-error rule](path_analysis.md#one-standard-error-component-heuristic) as a reproducible
parsimony heuristic. The search first refits the 1-SE-selected full-data model. Analysis then reads
`model.selection_`, whose `reference_minimum` and `one_standard_error_threshold` provide the
minimum-row and threshold annotations without repeated selection calls. The example obtains the
conditional rank profile through
`search.predictor_rank_profile(selection.n_components)` and evaluates the same selection through
`search.oof_report(X, Y, selection=selection)` before rendering fitted-model diagnostics.

The [search-owned selection rules](path_analysis.md#search-owned-selection-rules) define the
selection object, and the [component-path API reference](api/path.md) gives the exact method surface.

## Output artifacts and rendering ownership

Pulp, Sugarcane, and Tobacco each write six final PDF figures, including
`predictor_rank_profile.pdf`. Tobacco retains three-page prediction-diagnostic and coefficient
PDFs. No numbered
example writes a generated CSV file: Pulp, Sugarcane, and Tobacco are supplied by their named
`pipls.datasets` loaders. Every figure is constructed directly from
`component_path_`, conditional predictor-rank profiles, explicit OOF reports for retained
selections, and immutable inspection results. The Pulp factor view anchors every component to a positive tensile-
index (`TI`) response entry; Sugarcane and Tobacco retain the default predictor-based orientation.
The example layer owns Matplotlib chart construction, physical coordinates, subplot layouts,
legends, figure-level titles, PDF output, and closing. Pi-PLS factor panels use the same direct
array-to-Matplotlib boundary.

The Pulp tutorial extracts its checked snippets directly from `examples/05_pulp_real_data.py`.
The sole module under `examples/_support/` evaluates the nontrivial fold-local ordinary-PLS path for
example 04. It is not required for ordinary estimator use.
