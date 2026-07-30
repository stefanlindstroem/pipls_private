# Examples

The numbered examples are executable workflows organized by programming task. Start with the
[synthetic tutorial](tutorials/synthetic.md) for the short selection-and-prediction sequence, then
continue with the [Pulp tutorial](tutorials/pulp.md) for a complete real-data analysis. This page is
a catalogue of the maintained scripts.

## Choose an example

| Script | Programming task | Main output |
|---|---|---|
| `01_minimal_fit_and_plot.py` | Fit and inspect one known Pi-PLS rank pair from literal NumPy arrays | Printed predictions and `minimal_fit_and_plot.pdf` |
| `02_synthetic_path_selection.py` | Inspect a component path, select one fixed model, and evaluate independent test predictions | Three PDF figures and printed external-test $R^2$ |
| `03_leave_one_out_validation.py` | Validate a small calibration study with leave-one-out splits and ordered OOF predictions | Printed selected rank pair and immutable validation summary |
| `04_pls_path_comparison.py` | Compare matched Pi-PLS and ordinary PLS component paths | One comparison PDF for each reference dataset |
| `05_pulp_real_data.py` | Run the complete Pulp selection, rank-profile, inspection, and OOF workflow | Six PDF figures |
| `06_sugarcane_real_data.py` | Run the complete wavelength-aware Sugarcane workflow | Five PDF figures |
| `07_tobacco_real_data.py` | Apply the one-standard-error rule in a complete Tobacco spectral workflow | Five PDFs, including multipage diagnostics and coefficients |

The [fixed-regression reference](api/regression.md) documents the estimator used by example 01.
The [synthetic tutorial](tutorials/synthetic.md) extracts the maintained example 02 workflow
directly. The comparison in example 04 is optional and is not part of routine Pi-PLS fitting.

## Run one example

Install the example dependencies and execute a script from the repository root:

```bash
python -m pip install ".[examples]"
python examples/02_synthetic_path_selection.py
```

Generated files are written below `examples/results/`.

## Run all examples

```bash
make examples
```

The complete real-data analyses are intentionally outside `make check` because they are application
workflows and may take substantially longer than the package test suite.

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
  responses, including the component path, an immutable conditional predictor-rank profile from
  `predictor_rank_profile()`, score-loading
  biplot, fixed-model inspection, and OOF diagnostics;
- `examples/06_sugarcane_real_data.py`: the direct reference workflow, with a visible in-memory
  component path, scikit-learn OOF prediction, wavelength-aware inspection, and five final PDF
  figures;
- `examples/07_tobacco_real_data.py`: a complete spectral workflow that uses the
  [one-standard-error rule](path_analysis.md#one-standard-error-component-heuristic) to recommend
  the final component count before fixed-model inspection. See the focused explanation below.

### Tobacco: one-standard-error selection

The Tobacco component path has no clear elbow that would by itself motivate one component count.
Example 07 therefore demonstrates the conventional
[one-standard-error rule](path_analysis.md#one-standard-error-component-heuristic) as a reproducible
parsimony heuristic. Its component-path figure marks the minimum-mean-CV-MSE row, draws the
horizontal 1-SE threshold, and marks the smallest evaluated component count whose mean CV-MSE does
not exceed that threshold. The returned row also supplies the conditionally selected predictor rank
used by the final fixed model.

The [result-object recommendation methods](path_analysis.md#result-object-recommendations) describe
how the stored row is obtained, and the [component-path API reference](api/path.md) gives the exact
method surface. The example calls the result method explicitly instead of using the path-level
`selection_rule="one_standard_error"` orchestration.

## Output artifacts and rendering ownership

Pulp writes six final PDF figures, including `predictor_rank_profile.pdf`; Sugarcane writes five;
Tobacco writes five, with three-page prediction-diagnostic and coefficient PDFs. No numbered
example writes a generated CSV file: committed `X.csv` and `Y.csv` tables are inputs, while every
figure is constructed directly from `component_path_`, scikit-learn OOF predictions, and
immutable inspection results. The Pulp factor view anchors every component to a positive tensile-
index (`TI`) response entry; Sugarcane and Tobacco retain the default predictor-based orientation.
The example layer owns Matplotlib chart construction, physical coordinates, subplot layouts,
legends, figure-level titles, PDF output, and closing. Pi-PLS factor panels use the same direct
array-to-Matplotlib boundary.

The Pulp tutorial extracts its checked snippets directly from `examples/05_pulp_real_data.py`.
The sole module under `examples/_support/` evaluates the nontrivial fold-local ordinary-PLS path for
example 04. It is not required for ordinary estimator use.
