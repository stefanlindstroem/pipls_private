# Decision: 0004-response-standardized-mse

Status: private metric and fold-scale primitives implemented; public scorer and automatic
selection integration remain pending.

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
uniform weight. Phase C2b will average these fold losses across the materialized splits for each
candidate predictor rank.
