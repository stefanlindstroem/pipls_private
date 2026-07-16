# Predictor-rank modes

`PiPLSRegression` currently supports two predictor-rank modes for a fixed `n_components`.

An explicit integer fixes the predictor rank directly:

```python
model = PiPLSRegression(n_components=2, predictor_rank=4)
```

The rule-fixed mode derives the upper rank and uses it directly:

```python
model = PiPLSRegression(
    n_components=2,
    predictor_rank="max",
    samples_per_predictor_rank=10,
)
```

For data passed to `fit()` with $n_{\mathrm{train}}$ rows and $p$ predictor columns, the current
rule-fixed mode uses

\begin{equation}
r_{\pi,\max}
=
\min\left[
p,
n_{\mathrm{train}},
\left\lceil
n_{\mathrm{train}} / \texttt{samples\_per\_predictor\_rank}
\right\rceil
\right].
\end{equation}

After fitting, `max_predictor_rank_` stores this bound and `predictor_rank_` stores the rank
actually used. Explicit integer ranks are not clamped to the rule-derived bound.

`predictor_rank="auto"` is not implemented yet. Its later implementation will materialize CV
splits, pass the smallest training-fold size to the same bound helper, evaluate all admissible
ranks for the fixed `n_components`, and refit on the complete input data.
