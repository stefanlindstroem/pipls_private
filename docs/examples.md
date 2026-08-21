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
| `01_pulp_quick_start.py` | Short automatic workflow on [Pulp](datasets.md#pulp-real-data-integration): select by a CV rule, refit, and inspect fitted predictions | Selected model summary and `pulp_quick_start.pdf` |
| `02_synthetic_path_selection.py` | Learn manual component selection on synthetic train/test data, including path and optional predictor-rank inspection | Four PDF figures and printed external-test $R^2$ |
| `03_pls_path_comparison.py` | Compare both Π-PLS response-subspace policies with ordinary PLS on matched CV splits for three reference datasets plus one deterministic near-saturated synthetic stress case, without choosing a final model | Four three-way comparison PDFs |
| `04_pulp_real_data.py` | Full manual-selection workflow on [Pulp](datasets.md#pulp-real-data-integration): choose $h$ explicitly, review selection-conditioned OOF evidence, refit, and interpret the model | Ten PDF figures |
| `05_sugarcane_real_data.py` | Manual-selection spectral workflow on [Sugarcane](datasets.md#sugarcane-spectral-integration) with wavelength-aware model inspection | Six PDF figures |
| `06_tobacco_real_data.py` | Full automated-selection spectral workflow on [Tobacco](datasets.md#tobacco-spectral-integration) using separate relative-tolerance rules for $r_\pi$ and $h$ | Six PDFs, including threshold-annotated rank and component profiles |

The [path-selection reference](api/path.md) documents both the compact automatic route used by
example 01 and the explicit selection handoff used by the analytical examples. The
[synthetic tutorial](tutorials/synthetic.md) extracts the maintained example 02 workflow directly.
The comparison in example 03 is optional and is not part of routine Π-PLS fitting. For each
comparison case it materializes one shuffled five-fold protocol and reuses those exact splits for
separate `"cross_covariance"` and `"least_squares"` Π-PLS searches plus the ordinary-PLS path.
`response_subspace` remains fixed estimator configuration rather than a third search dimension.
The cross-covariance policy is the package default from the
[peer-reviewed companion publication](citation.md#companion-paper); least squares is the
RRR-inspired software extension and is not part of that publication. The resulting
CV-MSE paths are model-development evidence, not independent post-selection validation.

Examples 01, 02, and the [Pulp](datasets.md#pulp-real-data-integration) branch of example 03 use
the package's exhaustive predictor-rank default; the synthetic stress branch requests exhaustive
coverage explicitly. The high-dimensional [Sugarcane](datasets.md#sugarcane-spectral-integration)
and [Tobacco](datasets.md#tobacco-spectral-integration) branches of example 03 and complete examples
05 and 06 request `search_method="adaptive"` explicitly to control candidate cost over the new
full hard-feasible rank domain. That choice changes candidate coverage, not the admissible rank
endpoints; `search_is_exhaustive_` records whether the adaptive run happened to cover all pairs.

## Compare response-subspace policies {#compare-response-subspace-policies}

`examples/03_pls_path_comparison.py` is the maintained programming-user comparison. For
[Pulp](datasets.md#pulp-real-data-integration),
[Sugarcane](datasets.md#sugarcane-spectral-integration),
[Tobacco](datasets.md#tobacco-spectral-integration), and one deterministic synthetic stress case it
materializes one shuffled
five-fold protocol, fits one `PiPLSSearchCV` with `response_subspace="cross_covariance"` and one
with `response_subspace="least_squares"`, and evaluates ordinary PLS on those same folds. The
three conditioned component paths are overlaid in one case-specific figure. No final model is
fitted because the purpose is to compare model-development evidence under matched validation
partitions.

The synthetic case has 25 observations, 40 predictors, 10 responses, 5 shared latent directions,
15 predictor-specific directions, no response-specific directions, and Gaussian predictor and
response noise with standard deviation 0.3. Five-fold training sets contain 20 observations, so
centered predictor rank is at most 19 while the declared systematic predictor signal has 20 latent
directions. In this fixed `random_state=0` realization, both Π-PLS response policies attain their
minimum mean CV-MSE at 5 components: 0.9036 for cross-covariance and 0.9022 for least squares.
Ordinary PLS reaches 0.9373 at 8 components. The two Π-PLS curves remain close and alternate in
which one is lower, so this one stress realization does not establish a general performance
ordering.

The example does not claim that either response policy or ordinary PLS is generally superior. The
appropriate model is data-dependent. For the mathematical distinction between the two Π-PLS
response policies, see [Response-subspace selection](theory.md#response-subspace-selection).

## Run one example

Install the example dependencies and execute a script from the repository root:

```bash
python -m pip install ".[examples]"
python examples/01_pulp_quick_start.py
```

Generated files are written below `examples/results/`. Run only the fixed stress case with
`python examples/03_pls_path_comparison.py --dataset synthetic_stress`.

## Run all examples

```bash
make examples
```

The complete real-data analyses are intentionally outside `make check` because they are application
workflows and may take substantially longer than the package test suite.

## Cross-validation partitions

Example 01 intentionally uses the scikit-learn-compatible default `cv=5` to keep the opening
workflow to one automatic search/refit expression. Examples 02, 03, 05, and 06 use one explicit
shuffled five-fold partition. Example 03 materializes that partition once per comparison case and
reuses the same splits for both response-subspace policies and ordinary PLS. The complete Pulp workflow
instead uses
`RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)`: 50 materialized splits and ten OOF
predictions per observation. This deliberate stability analysis costs approximately ten times one
five-fold partition; the quick start remains lighter. The complete Pulp, Sugarcane, and Tobacco
workflows create one selection before OOF reporting and pass that same object to final refitting.
Example 03 is a comparison workflow and fits no final model. Grouped, temporal, or
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
coefficient PDFs. [Pulp](datasets.md#pulp-real-data-integration),
[Sugarcane](datasets.md#sugarcane-spectral-integration), and
[Tobacco](datasets.md#tobacco-spectral-integration) are supplied by their named `pipls.datasets`
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
