# Decision: 0004-response-standardized-mse

Status: implemented as public scorer callables and as automatic-selection diagnostics.

For validation fold $k$, let $s_{k,j}$ be the sample standard deviation (`ddof=1`) of response
column $j$ estimated from that fold's training observations. A zero scale, or every scale from a
singleton training fold, is replaced by 1.0. For validation index set $\mathcal{V}_k$, define

\begin{equation}
\operatorname{MSE}_{k,\mathrm{response\text{-}std}}
=
\frac{1}{|\mathcal{V}_k|q}
\sum_{i\in\mathcal{V}_k}
\sum_{j=1}^{q}
\left[
\frac{y_{ij}-\widehat{y}_{ij}}{s_{k,j}}
\right]^2.
\end{equation}

The implementation computes this quantity in original response units after prediction. It never
concatenates responses transformed under different fold-specific scalers. Responses receive
uniform weight. Candidate losses are the unweighted mean of split losses.

`response_standardized_mse` returns the positive loss and `neg_response_standardized_mse` returns
its negative for scikit-learn scorer maximization. Both follow the `(estimator, X, y)` scorer
signature and use the estimator's training-derived `response_scale_for_scoring_`.
