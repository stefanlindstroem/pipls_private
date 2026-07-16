# Predictor-rank selection

`PiPLSRegression` supports automatic, rule-fixed, and explicit predictor ranks for a fixed
`n_components`.

## Automatic mode

Automatic mode is the default:

```python
model = PiPLSRegression(
    n_components=2,
    predictor_rank="auto",
    samples_per_predictor_rank=10,
    cv=5,
    scoring="neg_response_standardized_mean_squared_error",
    n_jobs=None,
)
model.fit(X, Y)
```

The estimator materializes the CV splits once. If the smallest training fold contains
$n_{\mathrm{train,min}}$ samples and the input has $p$ predictor columns, the upper rank is

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

Every integer rank in

\begin{equation}
\{\texttt{n\_components},\ldots,r_{\pi,\max}\}
\end{equation}

is evaluated on the same splits. Each candidate estimator fits centering and optional scaling
only on the corresponding training fold. Scores use standard scikit-learn orientation: larger is
better. Equal mean scores within `rtol=1e-12` and `atol=1e-15` are resolved in favor of the
smaller predictor rank. The selected rank is then refitted once on all data supplied to `fit()`.

The default scorer is negative response-standardized MSE. For each validation fold, response
scales are sample standard deviations estimated from that fold's training responses with
`ddof=1`; zero scales and singleton-fold scales are replaced by 1.

Automatic-mode diagnostics include `predictor_rank_values_`, `predictor_rank_cv_results_`,
`best_score_`, `best_response_standardized_mse_`, `n_splits_`, and `cv_n_train_min_`.

## Rule-fixed mode

```python
model = PiPLSRegression(
    n_components=2,
    predictor_rank="max",
    samples_per_predictor_rank=10,
)
```

This uses the same rule directly without scanning ranks. Here the data supplied to `fit()` are
the training data used in the bound.

## Explicit mode

```python
model = PiPLSRegression(n_components=2, predictor_rank=4)
```

An explicit integer fixes the predictor rank and bypasses the rule-derived bound. It remains
subject to dimensional and numerical-rank validation in the fixed core.


## Accepted search-policy roadmap

The exhaustive scan described above is the current Phase C2b implementation. The accepted target
interface separates exact search coverage from adaptive computational shortcuts:

- `predictor_rank="optimal"` will evaluate every admissible integer rank and therefore return the
  CV optimum for the fixed split set and scoring rule;
- `predictor_rank="auto"` will use deterministic logarithmic coarse-to-fine exploration, cache all
  evaluated candidates, refine around the best observed region, and switch to exhaustive search
  once the remaining integer interval is small;
- `predictor_rank="max"` and explicit integer ranks retain their present meanings.

Adaptive search is approximate because a discrete CV-loss curve need not be unimodal. It must use
the same folds, fold-local preprocessing, scorer, and low-rank tie rule as exhaustive search, and
it must report all evaluated ranks and the final refinement interval. Randomized SVD is a separate
future numerical policy rather than an implicit consequence of choosing `"auto"`.
