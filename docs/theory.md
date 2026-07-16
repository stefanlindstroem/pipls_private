# Pi-PLS theory overview

Pi-PLS models multivariate linear regression through two rank controls and a diagonally coupled
latent representation.

For centered or centered-and-scaled predictors $\mathbf{X}\in\mathbb{R}^{n	imes p}$ and
responses $\mathbf{Y}\in\mathbb{R}^{n	imes q}$, the method:

1. retains a rank-$r_\pi$ predictor subspace from the leading right singular vectors of
   $\mathbf{X}$;
2. selects an $h$-dimensional response subspace from the leading right singular vectors of
   $\mathbf{Z}^{\mathsf T}\mathbf{Y}$, where $\mathbf{Z}=\mathbf{X}\mathbf{\Pi}$;
3. solves the reduced least-squares problem
   $\mathbf{Z}\mathbf{W}pprox\mathbf{Y}\mathbf{C}$;
4. diagonalizes $\mathbf{W}$ to obtain orthonormal predictor and response modes linked one to one
   by nonnegative dilation values.

The resulting centered/scaled regression map is

$$
\mathbf{B}_{\mathrm{cs}}
=
\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}
=
\mathbf{\Pi}\mathbf{W}\mathbf{C}^{\mathsf T}.
$$

Here $r_\pi$ controls the retained predictor-side signal space, while $h$ controls the number of
paired predictor-response modes. The structural conditions are
$1\le h\le\min(r_\pi,q)$ and $r_\pi\le\min(n,p)$, with an additional numerical-rank check in the
implementation.

The complete persistent theory reference for development is
[`../.llm/theory.md`](../.llm/theory.md). It includes the derivation, interpretation, OLS and
PLS-SVD relationships, identifiability rules, model-size argument, predictor-rank rationale,
cross-validation consequences, and implementation review checklist.

The concise normative equations used for code review remain in
[`../.llm/mathematics.md`](../.llm/mathematics.md), and finite-precision behavior is specified in
[`../.llm/numerical_contracts.md`](../.llm/numerical_contracts.md).
