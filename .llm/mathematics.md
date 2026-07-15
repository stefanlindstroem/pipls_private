# Mathematical contract

This file records the intended method before implementation. Source code and tests must
be kept consistent with it.

## Core notation

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

Admissibility requires $1\le h\le\min(r_\pi,q)$ and $r_\pi\le\min(n,p)$, with the
centered matrix rank handled explicitly.

Repeated or nearly repeated singular values identify invariant subspaces, not intrinsically
numbered basis vectors. Tests must compare projection matrices, principal angles, singular
values, regression maps, or predictions rather than raw basis columns.
