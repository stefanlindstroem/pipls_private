# Mathematical contract

This file is the concise normative mathematical contract for implementation. Source code and
tests must be kept consistent with it. Read `.llm/theory.md` for the fuller derivation,
interpretation, limiting cases, rank roles, and theory-to-implementation consequences.

## Core notation

Mathematical statements use uppercase $X$ and $Y$ for predictor and response matrices. Python
estimator signatures retain scikit-learn's `X` and `y` convention, including when `y` is a
two-dimensional multivariate response matrix. The programming name does not change the
mathematical object.

| Quantity | Shape | Meaning |
|---|---:|---|
| $\mathbf{X}_{\mathrm{cs}}$ | $(n,p)$ | centered, optionally scaled predictors |
| $\mathbf{Y}_{\mathrm{cs}}$ | $(n,q)$ | centered, optionally scaled responses |
| $\mathbf{\Pi}$ | $(p,r_\pi)$ | predictor basis from the leading right singular vectors of $\mathbf{X}_{\mathrm{cs}}$ |
| $\mathbf{Z}$ | $(n,r_\pi)$ | truncated predictors, $\mathbf{Z}=\mathbf{X}_{\mathrm{cs}}\mathbf{\Pi}$ |
| $\mathbf{C}$ | $(q,h)$ | leading right singular vectors of $\mathbf{Z}^{\mathsf{T}}\mathbf{Y}_{\mathrm{cs}}$ |
| $\mathbf{W}$ | $(r_\pi,h)$ | least-squares map from $\mathbf{Z}$ to $\mathbf{Y}_{\mathrm{cs}}\mathbf{C}$ |
| $\mathbf{P}$ | $(p,h)$ | predictor rotations |
| $\mathbf{D}$ | $(h,h)$ | nonnegative diagonal dilation matrix |
| $\mathbf{Q}$ | $(q,h)$ | response rotations |

The intended fixed-parameter construction is:

1. compute $\mathbf{\Pi}$;
2. set $\mathbf{Z}=\mathbf{X}_{\mathrm{cs}}\mathbf{\Pi}$;
3. compute $\mathbf{C}$ from $\mathbf{Z}^{\mathsf{T}}\mathbf{Y}_{\mathrm{cs}}$;
4. solve $\mathbf{Z}\mathbf{W}\approx\mathbf{Y}_{\mathrm{cs}}\mathbf{C}$ by least squares;
5. factor $\mathbf{W}=\mathbf{M}\mathbf{D}\mathbf{N}^{\mathsf{T}}$;
6. set $\mathbf{P}=\mathbf{\Pi}\mathbf{M}$ and $\mathbf{Q}=\mathbf{C}\mathbf{N}$.

Admissibility requires $1\le h\le\min(r_\pi,q)$ and $r_\pi\le\min(n,p)$. The
private core additionally requires $r_\pi$ not to exceed the numerical rank of the supplied
predictor matrix. For the thin SVD singular values $s_i$, numerical rank is determined by
$s_i > \tau_X$, where $\tau_X = \max(n,p)\,\epsilon_{64}\,s_1$.

The centered/scaled regression map is

\begin{equation}
\mathbf{B}_{\mathrm{cs}}=\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf{T}}
=\mathbf{\Pi}\mathbf{W}\mathbf{C}^{\mathsf{T}},
\end{equation}

so fitted values may be computed equivalently as $\mathbf{X}_{\mathrm{cs}}\mathbf{B}_{\mathrm{cs}}$
or $\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf{T}}$.

Repeated or nearly repeated singular values identify invariant subspaces, not intrinsically
numbered basis vectors. Tests must compare projection matrices, principal angles, singular
values, regression maps, or predictions rather than raw basis columns.

For path selection, let $r_{\mathrm{num,min}}$ be the minimum predictor rank verified after
fold-local pipeline preprocessing and terminal-estimator centering/scaling. The default path ceiling
is

\begin{equation}
r_{\pi,\max}=\min\left[p_{\min},n_{\mathrm{train,min}}-1,
r_{\mathrm{num,min}},\left\lceil\frac{n}{c}\right\rceil\right].
\end{equation}
