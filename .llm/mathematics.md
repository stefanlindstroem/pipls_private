# Mathematical contract

This file is the concise normative mathematical contract for implementation. Source code and
tests must be kept consistent with it. Read `.llm/theory.md` for the fuller derivation,
interpretation, limiting cases, rank roles, and theory-to-implementation consequences.

## Core notation

Mathematical statements use uppercase $X$ and $Y$ for predictor and response matrices. Python
estimator signatures retain scikit-learn's `X` and `y` convention, including when `y` is a
two-dimensional multivariate response matrix. The programming name does not change the
mathematical object.

Complete matrices are bold. Use `\mathbf` for Latin matrix symbols and `\boldsymbol` for Greek
matrix symbols that must render in bold. Descriptive, role, block, method, and extremum subscripts
are upright with `\mathrm`, for example $d_{\mathrm{p}}$,
$\boldsymbol{\Lambda}_{\mathrm{s}}$, $\mathbf{L}_{\mathrm{sp}}$,
$\mathbf{U}_{\mathrm{X}}$, and $h_{\mathrm{max}}$. Mathematical indices and dimensions remain
italic, for example $d_k$, $D_{kk}$, $P_{:k}$, $s_i$, $r_\pi$, and $\mathbf{I}_p$.

| Quantity | Shape | Meaning |
|---|---:|---|
| $\mathbf{X}_{\mathrm{cs}}$ | $(n,p)$ | centered, optionally scaled predictors |
| $\mathbf{Y}_{\mathrm{cs}}$ | $(n,q)$ | centered, optionally scaled responses |
| $\mathbf{\Pi}$ | $(p,r_\pi)$ | retained predictor basis from the leading right singular vectors of $\mathbf{X}_{\mathrm{cs}}$ |
| $\mathbf{Z}$ | $(n,r_\pi)$ | retained predictor scores, $\mathbf{Z}=\mathbf{X}_{\mathrm{cs}}\mathbf{\Pi}$ |
| $\mathbf{C}$ | $(q,h)$ | response basis maximizing retained squared cross-covariance |
| $\mathbf{W}$ | $(r_\pi,h)$ | least-squares map from $\mathbf{Z}$ to $\mathbf{Y}_{\mathrm{cs}}\mathbf{C}$ |
| $\mathbf{M}$ | $(r_\pi,h)$ | left singular vectors of $\mathbf{W}$ |
| $\mathbf{D}$ | $(h,h)$ | nonnegative diagonal dilation matrix |
| $\mathbf{N}$ | $(h,h)$ | right singular vectors of $\mathbf{W}$ |
| $\mathbf{P}$ | $(p,h)$ | orthonormal predictor directions, $\mathbf{P}=\mathbf{\Pi}\mathbf{M}$ |
| $\mathbf{Q}$ | $(q,h)$ | orthonormal response directions, $\mathbf{Q}=\mathbf{C}\mathbf{N}$ |

Admissibility requires

\begin{equation}
1\le h\le\min(r_\pi,q),
\qquad
r_\pi\le\min(n,p).
\end{equation}

## Terminology and orientation

$\mathbf{\Pi}$ is the retained predictor basis and
$\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}$ is the retained-subspace projector. The columns of
$\mathbf{P}$ and $\mathbf{Q}$ are orthonormal predictor and response directions. The projectors
onto their final spans are $\mathbf{P}\mathbf{P}^{\mathsf T}$ and
$\mathbf{Q}\mathbf{Q}^{\mathsf T}$; neither $\mathbf{P}$ nor $\mathbf{Q}$ is itself a
projection matrix.

For each $k$, $d_k=D_{kk}$ is the dilation of paired latent mode
$(P_{:k},d_k,Q_{:k})$. The matrices $\mathbf{X}_{\mathrm{cs}}\mathbf{P}$ and
$\mathbf{Y}_{\mathrm{cs}}\mathbf{Q}$ contain predictor and response scores. Public
`n_components` counts the $h$ paired latent modes, while public `predictor_rank` denotes the retained
predictor-subspace dimension $r_\pi$.

The package's weighted response directions have response-by-mode orientation
$\mathbf{Q}\mathbf{D}$, and

\begin{equation}
\mathbf{D}\mathbf{Q}^{\mathsf T}
=
(\mathbf{Q}\mathbf{D})^{\mathsf T}.
\end{equation}

The public decomposition fields are `predictor_directions` and `response_directions`. The standard
PLS-style fitted attributes remain `x_rotations_` and `y_rotations_`. Predictor and response
directions are not the estimator's separate least-squares reconstruction loadings.

The private core additionally requires $r_\pi$ not to exceed the numerical rank of the supplied
predictor matrix. For thin-SVD singular values $s_i$, numerical rank is determined by
$s_i>\tau_{\mathrm{X}}$, where

\begin{equation}
\tau_{\mathrm{X}}=\max(n,p)\,\epsilon_{64}\,s_1.
\end{equation}

## Fixed-parameter construction

For the thin SVD

\begin{equation}
\mathbf{X}_{\mathrm{cs}}
=\mathbf{U}_{\mathrm{X}}\mathbf{S}_{\mathrm{X}}\mathbf{V}_{\mathrm{X}}^{\mathsf T},
\end{equation}

retain the first $r_\pi$ right singular vectors in $\mathbf{\Pi}$. The retained-subspace projector
and predictor decomposition are

\begin{equation}
\mathbf{\Pi}^{\mathsf T}\mathbf{\Pi}=\mathbf{I}_{r_\pi},
\end{equation}

\begin{equation}
\mathbf{X}_{\mathrm{cs}}
=
\mathbf{X}_{\mathrm{cs}}\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}
+
\mathbf{X}_{\mathrm{cs}}
(\mathbf{I}_p-\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}).
\end{equation}

Set

\begin{equation}
\mathbf{Z}=\mathbf{X}_{\mathrm{cs}}\mathbf{\Pi}.
\end{equation}

The response basis solves

\begin{equation}
\max_{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h}
\left\|
\mathbf{Z}^{\mathsf T}\mathbf{Y}_{\mathrm{cs}}\mathbf{C}
\right\|_{\mathrm{F}}^2.
\end{equation}

One optimum is formed by the leading $h$ right singular vectors of
$\mathbf{Z}^{\mathsf T}\mathbf{Y}_{\mathrm{cs}}$. Thus

\begin{equation}
\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h.
\end{equation}

Solve

\begin{equation}
\mathbf{Z}\mathbf{W}
\approx
\mathbf{Y}_{\mathrm{cs}}\mathbf{C}
\end{equation}

by minimum-norm least squares, then factor

\begin{equation}
\mathbf{W}=\mathbf{M}\mathbf{D}\mathbf{N}^{\mathsf T},
\qquad
\mathbf{D}=\operatorname{diag}(d_1,\ldots,d_h),
\qquad
d_1\ge\cdots\ge d_h\ge0.
\end{equation}

Define

\begin{equation}
\mathbf{P}=\mathbf{\Pi}\mathbf{M},
\qquad
\mathbf{Q}=\mathbf{C}\mathbf{N}.
\end{equation}

The resulting invariants are

\begin{equation}
\mathbf{P}^{\mathsf T}\mathbf{P}=\mathbf{I}_h,
\qquad
\mathbf{Q}^{\mathsf T}\mathbf{Q}=\mathbf{I}_h,
\end{equation}

and the diagonal latent relation is

\begin{equation}
\mathbf{Y}_{\mathrm{cs}}\mathbf{Q}
=
\mathbf{X}_{\mathrm{cs}}\mathbf{P}\mathbf{D}
+
\mathbf{E}_\pi.
\end{equation}

The centered/scaled regression map is

\begin{equation}
\mathbf{B}_{\mathrm{cs}}
=
\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}
=
\mathbf{\Pi}\mathbf{W}\mathbf{C}^{\mathsf T}.
\end{equation}

Fitted values must agree under both factorizations:

\begin{equation}
\mathbf{X}_{\mathrm{cs}}\mathbf{B}_{\mathrm{cs}}
=
\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf T}.
\end{equation}

After $\mathbf{\Pi}$ is fixed, the nominal fitted dimension is

\begin{equation}
(r_\pi+q-h)h.
\end{equation}

This count is based on $\mathbf{M}$, $\mathbf{D}$, and $\mathbf{Q}$; the entries of
$\mathbf{P}=\mathbf{\Pi}\mathbf{M}$ are not independently free.

Repeated or nearly repeated singular values identify invariant subspaces, not intrinsically
numbered basis vectors. Tests must compare projection matrices, principal angles, singular values,
regression maps, or predictions rather than raw basis columns.

## Path-selection boundary

For path selection, let $r_{\mathrm{num,min}}$ be the minimum predictor rank verified after
fold-local pipeline preprocessing and terminal-estimator centering/scaling. The default path ceiling
is

\begin{equation}
r_{\pi,\mathrm{max}}
=
\min\left[
 p_{\mathrm{min}},
 n_{\mathrm{train,min}}-1,
 r_{\mathrm{num,min}},
 \left\lceil\frac{n}{c}\right\rceil
\right].
\end{equation}

This path policy is a package contract around the fixed model. It is not part of the mathematical
definition above and is not changed by companion-manuscript theory alignment.
