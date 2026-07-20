# Decision 0047: separate the PLS comparison from Pi-PLS real-data workflows

## Status

Accepted and implemented.

This decision supersedes the example-ownership portions of Decisions 0036, 0038, 0042, 0045, and
0046. Their numerical CV-MSE definitions, canonical CSV-first plotting contract, Pi-PLS-specific
factorization inspection, shared PLS-family analysis API, and concise-example principles remain
accepted.

## Context

Examples 10, 11, and 12 previously served two different purposes at once:

1. compare Pi-PLS with ordinary PLS through overlaid component-path CV-MSE curves; and
2. demonstrate a normal Pi-PLS analysis from path selection through fitted-model inspection.

The comparison is useful, but it should not appear as an automatic step in ordinary Pi-PLS use.
A user following a dataset analysis should first see how to evaluate a Pi-PLS path, choose one
Pi-PLS configuration, fit that model, and inspect it. The external PLS reference belongs in one
explicit comparison example.

The Pi-PLS path is also part of each dataset analysis. Its CSV and PDF should therefore live beside
the post-analysis report rather than in the shared results root.

## Decision

### Dedicated comparison example

`examples/09_pls_path_comparison.py` owns all real-data Pi-PLS-versus-PLS path comparisons. It runs
the Pulp, Sugarcane, and Tobacco datasets, writes separate canonical Pi-PLS and PLS path CSV files,
and generates one overlaid comparison PDF per dataset under:

```text
examples/results/pls_path_comparison/
```

The comparison uses the existing common ordered five-fold partitions, fold-local scaling,
response-standardized CV-MSE, fold SD, and matching component counts. The ordinary PLS path remains
implemented by `examples/_support/pls_component_path.py`.

### Dataset analysis examples

Examples 10, 11, and 12 do not evaluate or plot ordinary PLS. Each example:

1. evaluates one Pi-PLS component path;
2. writes `component_path.csv` and `component_path.pdf` in its dataset analysis directory;
3. chooses one component count and reads the corresponding predictor rank from that Pi-PLS path;
4. fits one fixed `PiPLSRegression`;
5. writes the existing post-analysis tables and `post_analysis.pdf` in the same directory.

The dataset directories are:

```text
examples/results/pulp_post_analysis/
examples/results/sugarcane_post_analysis/
examples/results/tobacco_post_analysis/
```

`component_path.pdf` and `post_analysis.pdf` remain separate because they answer different
questions: model-selection diagnostics versus fitted-model interpretation and prediction
diagnostics.

### Plot helper boundary

`examples/_support/plot_component_path.py` exposes two explicit functions:

- `plot_pipls_component_path()` for one Pi-PLS path;
- `plot_component_path_comparison()` for an overlaid Pi-PLS-versus-PLS comparison.

Both render only from canonical CSV files. The Pi-PLS path retains predictor-rank annotations.

## Consequences

- Examples 10–12 present a normal Pi-PLS workflow without implying that ordinary PLS comparison is
  required in routine use.
- Users still have one clear, reproducible example for all Pi-PLS-versus-PLS CV-MSE comparisons.
- Ordinary PLS remains confined to the comparison helper and declared comparator benchmarks.
- Each dataset analysis directory contains its own selection artifact and post-analysis artifact.
- `make examples` still runs every numbered example, including the explicit comparison example and
  the three Pi-PLS analyses.
- No estimator, path-search, scoring, fold, selected-model, post-analysis, or public package API
  behavior changes.
