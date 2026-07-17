# Predictor-rank selection

`PiPLSRegression` supports adaptive, exhaustive, rule-fixed, and explicit predictor ranks for a
fixed `n_components`.

## Shared CV bound

For both CV modes, the estimator materializes one split set. If the smallest training fold has
$n_{\mathrm{train,min}}$ samples and the input has $p$ predictors, then

\begin{equation}
r_{\pi,\max}
=
\min\left[
p,
n_{\mathrm{train,min}},
\left\lceil
n_{\mathrm{train,min}} / \texttt{samples\_per\_predictor\_rank}
\right\rceil
\right].
\end{equation}

Every evaluated candidate uses the same splits and fold-local centering and scaling. Scores use
standard scikit-learn orientation, where larger is better. Equal mean scores within
`rtol=1e-12` and `atol=1e-15` are resolved in favor of the smaller predictor rank. The selected
rank is refitted once on all data supplied to `fit()`.

The default scorer is negative response-standardized MSE. Fold response scales are training-fold
sample standard deviations with `ddof=1`; zero scales and singleton-fold scales become 1.

## Statistical-support warning

The default is $c=5$. Values below 5 remain legal for exploratory work but emit
`pipls.StatisticalSupportWarning` in `"max"`, `"optimal"`, and `"auto"` modes. Fewer than five
training samples per retained predictor-rank direction provide insufficient statistical support
for the resulting rank bound to be trusted without external validation. Explicit integer ranks
bypass the $c$ rule and do not emit this warning.

## Adaptive `"auto"` mode

```python
model = PiPLSRegression(n_components=2)
```

This uses the defaults `predictor_rank="auto"`, `samples_per_predictor_rank=5`, `cv=5`,
and response-standardized MSE scoring. It begins with seven deterministic approximately
logarithmic integer ranks,
including both endpoints. It caches every result, finds the best evaluated rank using the normal
tie rule, and refines between its evaluated neighbors. When that interval contains at most 10
integer ranks, the remaining ranks in the interval are evaluated exhaustively.

The method is deterministic but approximate: a discrete CV curve need not be unimodal, so
`"auto"` is not guaranteed to equal `"optimal"`. It does become exhaustive for small admissible
sets.

## Exhaustive `"optimal"` mode

```python
model = PiPLSRegression(
    n_components=2,
    predictor_rank="optimal",
    cv=5,
)
```

This evaluates every admissible integer rank and therefore returns the CV optimum for the fixed
split set, scorer, and tie rule.

## Search diagnostics

Both CV modes expose `predictor_rank_values_`, `predictor_rank_cv_results_`, `best_score_`,
`best_response_standardized_mse_`, `n_splits_`, and `cv_n_train_min_`. Search-specific diagnostics
are:

- `predictor_rank_evaluation_order_`;
- `predictor_rank_search_history_`;
- `predictor_rank_search_method_`;
- `predictor_rank_search_interval_`;
- `n_predictor_rank_candidates_`;
- `n_predictor_rank_evaluated_`;
- `n_predictor_rank_skipped_`;
- `predictor_rank_search_exhaustive_`.

## Rule-fixed `"max"` mode

`predictor_rank="max"` uses the same rule directly without rank CV. Here the data supplied to
`fit()` determine the training-size term.

## Explicit integer mode

A positive integer fixes the predictor rank and bypasses the rule-derived bound, while remaining
subject to dimensional and numerical-rank validation in the fixed core.


## Groups, OOF predictions, and validation reports

`PiPLSRegression.fit(X, y, groups=groups)` passes groups to its internal splitter. With
`return_oof_predictions=True`, the selected fixed rank is refitted on the same materialized
training folds and produces row-ordered `oof_predictions_` plus per-row counts. Automatic and
optimal rank reports are labeled selection-conditioned; explicit integer and rule-fixed ranks are
labeled fixed-parameter. Singleton validation folds reject ordinary R2 scoring.
