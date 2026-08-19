# Examples

The numbered examples are executable workflows organized by modeling purpose. Start with the
package-owned [Pulp quick start](tutorials/quick_start.md), continue with the
[synthetic tutorial](tutorials/synthetic.md) for component-path inspection and independent-test
prediction, then use the
[Pulp tutorial](tutorials/pulp.md) for a complete real-data analysis. This page is a catalogue of
the maintained scripts.

## Choose an example

| Script | Workflow focus | Main output |
|---|---|---|
| `01_pulp_quick_start.py` | Short automatic workflow on Pulp: select by a CV rule, refit, and inspect fitted predictions | Selected model summary and `pulp_quick_start.pdf` |
| `02_synthetic_path_selection.py` | Learn manual component selection on synthetic train/test data, including path and optional predictor-rank inspection | Four PDF figures and printed external-test $R^2$ |
| `03_pls_path_comparison.py` | Compare matched Π-PLS and ordinary PLS CV component paths without choosing a final model | One comparison PDF for each reference dataset |
| `04_pulp_real_data.py` | Full manual-selection workflow on Pulp: choose $h$ explicitly, review selection-conditioned OOF evidence, refit, and interpret the model | Ten PDF figures |
| `05_sugarcane_real_data.py` | Manual-selection spectral workflow on Sugarcane with wavelength-aware model inspection | Six PDF figures |
| `06_tobacco_real_data.py` | Full automated-selection spectral workflow on Tobacco using separate relative-tolerance rules for $r_\pi$ and $h$ | Six PDFs, including threshold-annotated rank and component profiles |
| `07_response_subspace_comparison.py` | Compare the publication-default cross-covariance and optional least-squares response-subspace policies on identical Pulp CV splits | One overlaid component-path PDF and printed selected CV-MSE summaries |

The [path-selection reference](api/path.md) documents both the compact automatic route used by
example 01 and the explicit selection handoff used by the analytical examples. The
[synthetic tutorial](tutorials/synthetic.md) extracts the maintained example 02 workflow directly.
The comparison in example 03 is optional and is not part of routine Π-PLS fitting.
Example 07 is also an optional programming-user comparison. It runs two separate Π-PLS searches
with the same materialized five-fold partitions because `response_subspace` is fixed estimator
configuration rather than a third search dimension. The `"cross_covariance"` search is the
peer-reviewed package default; `"least_squares"` is the RRR-inspired software extension and is
not part of the peer-reviewed companion publication. The resulting CV-MSE values are
model-development evidence, not independent post-selection validation.

Examples 01, 02, and the Pulp branch of example 03 use the package's exhaustive predictor-rank
default. The high-dimensional Sugarcane and Tobacco branches of example 03 and complete examples
05 and 06 request `search_method="adaptive"` explicitly to control candidate cost over the new
full hard-feasible rank domain. That choice changes candidate coverage, not the admissible rank
endpoints; `search_is_exhaustive_` records whether the adaptive run happened to cover all pairs.

## Compare response-subspace policies {#compare-response-subspace-policies}

`examples/07_response_subspace_comparison.py` demonstrates the programming-user comparison
without changing the search surface. It materializes one shuffled five-fold protocol, fits one
`PiPLSSearchCV` with `response_subspace="cross_covariance"` and one with
`response_subspace="least_squares"`, applies the same minimum-CV-MSE selection rule to each
conditioned component path, and overlays the two paths. No final model is fitted because the
purpose is to compare model-development evidence under matched validation partitions.

The example does not claim that either response policy is generally superior. The
least-squares route can be efficient for some problems, but the appropriate choice is
data-dependent. For the mathematical distinction, see
[Response-subspace selection](theory.md#response-subspace-selection).

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
workflow to one automatic search/refit expression. Examples 02, 03, 05, 06, and 07 use one
explicit shuffled five-fold partition. Example 07 materializes that partition once and reuses the
same splits for both response-subspace policies. The complete Pulp workflow instead uses
`RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)`: 50 materialized splits and ten OOF
predictions per observation. This deliberate stability analysis costs approximately ten times one
five-fold partition; the quick start remains lighter. The complete Pulp, Sugarcane, and Tobacco
workflows create one selection before OOF reporting and pass that same object to final refitting.
Examples 03 and 07 are comparison workflows and fit no final model. Grouped, temporal, or
otherwise structured data
require an application-specific splitter. The
[computational-performance
guide](computational_performance.md#develop-with-a-smaller-validation-protocol)
explains how to use a lighter seeded protocol during development and restore the final declared
validation effort for reported results.

## Complete real-data analyses

The [dataset documentation](datasets.md) gives the original source, DOI, license, and repository
adaptation for each real-data integration. The three complete analyses evaluate one Π-PLS
component path, create one immutable selection, inspect its selected evidence and optional OOF
report, and then fit that exact row on all observations:

- `examples/04_pulp_real_data.py`: the direct tutorial workflow for named scalar predictors and
  responses. It first presents the unselected component path, creates the declared manual
  selection, reviews the selected path, conditional predictor-rank profile, and
  selection-conditioned OOF predictions, refits the same selection, calculates immutable
  fitted-model results, and then renders ten figures;
- `examples/05_sugarcane_real_data.py`: the direct reference workflow with the same ordering,
  wavelength-aware inspection, and six final PDF figures;
- `examples/06_tobacco_real_data.py`: a complete spectral workflow with two explicit parsimony
  decisions. For every component count, the search retains the smallest evaluated predictor rank
  within 10% of the exact conditional optimum. The workflow then selects the smallest
  component-count row within 10% of the minimum on that conditioned path, inspects that selection,
  and refits it. See the focused explanation below.

### Tobacco: two relative-tolerance decisions

The Tobacco workflow demonstrates the hierarchical selection contract with two separately named
10% tolerances:

```python
PREDICTOR_RANK_RELATIVE_TOLERANCE = 0.10
COMPONENT_RELATIVE_TOLERANCE = 0.10

search = PiPLSSearchCV(
    predictor_rank_relative_tolerance=PREDICTOR_RANK_RELATIVE_TOLERANCE,
    search_method="adaptive",
    cv=CV,
).fit(X, Y)

selection = search.select(
    rule="minimum_cv_mse",
    relative_tolerance=COMPONENT_RELATIVE_TOLERANCE,
)
rank_profile = search.predictor_rank_profile(selection.n_components)
report = search.oof_report(X, Y, selection=selection)
model = search.refit(X, Y, selection=selection)
```

The first tolerance acts independently within each evaluated component count and determines the
predictor rank stored in `component_path_`. The second acts only on that conditioned path and
determines the final component count. They do not form one global 10% rule; under the default
CV-MSE scorer, two limiting 10% allowances can compound to $1.1^2=1.21$ relative to the global
candidate minimum.

The predictor-rank profile renders its exact conditional optimum, the scorer-derived 10% CV-MSE
threshold, and the smaller retained rank. The component path separately renders its exact
conditioned-path minimum, 10% threshold, and retained component count. The console report names both
reference and retained choices. The immutable `selection` carries the component evidence and
identifies the rank profile, OOF report, and final full-data fit. After fitting succeeds,
`model.selection_` records that exact object as model provenance. Absolute caps remain available for
both stages but are not used in this example.

The [search-owned selection rules](path_analysis.md#search-owned-selection-rules) define the
selection object, and the [component-path API reference](api/path.md) gives the exact method surface.

## Output artifacts and rendering ownership

Sugarcane and Tobacco each write six final PDF figures, including
`predictor_rank_profile.pdf`. Pulp writes those six figures plus
`selected_component_path.pdf` and the three final-fit diagnostics
`final_fit_observed_vs_predicted.pdf`, `final_fit_r2.pdf`, and
`final_fit_residual_distribution.pdf`. Tobacco retains three-page prediction-diagnostic and
coefficient PDFs. Pulp, Sugarcane, and Tobacco are supplied by their named `pipls.datasets`
loaders. Every figure is constructed directly from
`component_path_`, conditional predictor-rank profiles, explicit OOF reports for retained
selections, and immutable inspection results. The Pulp factor view anchors every component to a
positive tensile-index (`TI`) response entry; Sugarcane and Tobacco retain the default
predictor-based orientation. The example layer owns Matplotlib chart construction, physical
coordinates, subplot layouts, legends, figure-level titles, PDF output, and closing. Π-PLS factor
panels use the same direct array-to-Matplotlib boundary.

The Pulp tutorial extracts its checked snippets directly from `examples/04_pulp_real_data.py`.
The sole module under `examples/_support/` evaluates the nontrivial fold-local ordinary-PLS path for
example 03. It is not required for ordinary estimator use.
