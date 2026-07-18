# Pi-PLS theory reference

## Purpose and authority

This document is the persistent conceptual reference for LLM-assisted development of Pi-PLS.
It summarizes the mathematical construction, interpretation, limiting cases, rank roles, and
implementation consequences of the method so that the project owner does not need to attach the
manuscript for routine implementation work.

The scientific source for this summary is the project owner's Pi-PLS manuscript. This file is a
repository-maintained explanation, not a replacement for the manuscript. The authority hierarchy
for implementation work is:

1. explicit scientific decisions from the project owner;
2. accepted decision records under `docs/decisions/`;
3. the normative equations and invariants in `.llm/mathematics.md`;
4. the numerical rules in `.llm/numerical_contracts.md`;
5. this conceptual reference;
6. source code and tests as evidence of currently implemented behavior.

If a future manuscript revision conflicts with this document, do not silently reconcile the two.
Record the discrepancy, obtain an explicit decision, and update all affected contracts together.

## Problem setting

Let

- $\mathbf{X}\in\mathbb{R}^{n\times p}$ be the predictor matrix;
- $\mathbf{Y}\in\mathbb{R}^{n\times q}$ be the response matrix;
- $n$ be the number of observations;
- $p$ be the number of predictor variables;
- $q$ be the number of response variables.

The theoretical core is formulated for column-centered matrices. The public estimator may also
scale columns, but centering and scaling are preprocessing operations outside the fixed-parameter
core. In centered or centered-and-scaled coordinates, multivariate linear regression seeks
$\mathbf{B}\in\mathbb{R}^{p\times q}$ such that

$$
\mathbf{Y}=\mathbf{X}\mathbf{B}+\mathbf{E}.
$$

Unconstrained multivariate OLS uses the minimum-norm coefficient matrix
$\mathbf{B}_{\mathrm{OLS}}=\mathbf{X}^{+}\mathbf{Y}$. Pi-PLS instead separates two structural
choices:

1. which predictor subspace is retained, controlled by the predictor rank $r_\pi$;
2. how many coupled predictor-response modes are retained, controlled by the latent dimension
   $h$.

This separation is central. The roles of $r_\pi$ and $h$ are not interchangeable.

## Conceptual position among related methods

Pi-PLS combines ideas familiar from several multivariate methods:

- Like truncated-SVD regression, it first restricts estimation to a rank-controlled predictor
  subspace.
- Like PLS and PLS-SVD, it uses predictor-response cross-covariance to identify response-relevant
  latent directions.
- Like reduced-rank regression, it represents the coefficient matrix through a low-rank
  factorization.
- Unlike iterative deflation-based PLS algorithms, the fixed-parameter Pi-PLS construction is a
  sequence of SVD and least-squares operations with an explicit closed form.
- Unlike PLS-SVD, Pi-PLS first preserves a chosen predictor-side signal subspace and only then
  identifies response directions from cross-covariance within that subspace.

The defining interpretive feature is a diagonal one-to-one coupling between orthogonal predictor
and response modes.

## Notation and dimensions

| Quantity | Shape | Meaning |
|---|---:|---|
| $\mathbf{X}$ | $(n,p)$ | centered or centered-and-scaled predictors supplied to the core |
| $\mathbf{Y}$ | $(n,q)$ | centered or centered-and-scaled responses supplied to the core |
| $r_\pi$ | scalar | retained predictor-subspace dimension |
| $h$ | scalar | number of coupled latent modes |
| $\mathbf{\Pi}$ | $(p,r_\pi)$ | orthonormal retained predictor basis |
| $\mathbf{Z}=\mathbf{X}\mathbf{\Pi}$ | $(n,r_\pi)$ | predictor scores in the retained subspace |
| $\mathbf{C}$ | $(q,h)$ | orthonormal response basis selected by cross-covariance |
| $\mathbf{W}$ | $(r_\pi,h)$ | least-squares map from $\mathbf{Z}$ to $\mathbf{Y}\mathbf{C}$ |
| $\mathbf{M}$ | $(r_\pi,h)$ | left singular vectors of $\mathbf{W}$ |
| $\mathbf{D}$ | $(h,h)$ | nonnegative diagonal dilation matrix |
| $\mathbf{N}$ | $(h,h)$ | right singular vectors of $\mathbf{W}$ |
| $\mathbf{P}=\mathbf{\Pi}\mathbf{M}$ | $(p,h)$ | orthonormal predictor latent basis |
| $\mathbf{Q}=\mathbf{C}\mathbf{N}$ | $(q,h)$ | orthonormal response latent basis |
| $\mathbf{B}_{\mathrm{cs}}$ | $(p,q)$ | regression map in core coordinates |

The dimensional admissibility conditions are

$$
1\le h\le \min(r_\pi,q),
\qquad
h\le r_\pi\le \min(n,p).
$$

The implementation additionally requires $r_\pi$ not to exceed the numerical rank of the supplied
predictor matrix.

## Step 1: retain a predictor signal subspace

Let the singular value decomposition of the core predictor matrix be

$$
\mathbf{X}=\mathbf{U}_X\mathbf{S}_X\mathbf{V}_X^{\mathsf T}.
$$

Retain the leading $r_\pi$ right singular vectors:

$$
\mathbf{\Pi}=\mathbf{V}_{X(:,1:r_\pi)},
\qquad
\mathbf{\Pi}^{\mathsf T}\mathbf{\Pi}=\mathbf{I}_{r_\pi}.
$$

The corresponding orthogonal projector is
$\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}$, and the predictor matrix decomposes as

$$
\mathbf{X}
=
\mathbf{X}\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}
+
\mathbf{X}(\mathbf{I}_p-\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}).
$$

Define the retained predictor scores

$$
\mathbf{Z}=\mathbf{X}\mathbf{\Pi}.
$$

The first term in the decomposition is the rank-$r_\pi$ predictor approximation
$\mathbf{Z}\mathbf{\Pi}^{\mathsf T}$. The second term is orthogonal to the retained predictor
subspace and is treated as truncation residual. Thus, $r_\pi$ determines which predictor variation
remains available to all later stages. Any predictive direction removed here cannot be recovered
by increasing $h$.

## Step 2: select the response subspace by cross-covariance

Within the retained predictor space, form the predictor-response cross-product matrix

$$
\mathbf{\Sigma}_{ZY}=\mathbf{Z}^{\mathsf T}\mathbf{Y}
\in\mathbb{R}^{r_\pi\times q}.
$$

Pi-PLS seeks an orthonormal response basis
$\mathbf{C}\in\mathbb{R}^{q\times h}$ that maximizes retained squared cross-covariance:

$$
\max_{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h}
\left\|\mathbf{\Sigma}_{ZY}\mathbf{C}\right\|_F^2.
$$

If

$$
\mathbf{\Sigma}_{ZY}=\mathbf{U}\mathbf{S}\mathbf{V}^{\mathsf T},
$$

then one optimum is

$$
\mathbf{C}=\mathbf{V}_{(:,1:h)}.
$$

Therefore, $\mathbf{C}$ spans the $h$ response directions most strongly coupled, in the
cross-covariance sense, to the retained predictor representation. The basis vectors themselves
are not unique when the relevant singular values are repeated, but the maximizing subspace is the
meaningful object.

## Step 3: regress in the reduced latent coordinates

Project the response matrix onto the selected response subspace and solve

$$
\mathbf{Y}\mathbf{C}=\mathbf{Z}\mathbf{W}+\mathbf{E}'',
$$

by least squares. The minimum-norm solution is

$$
\mathbf{W}=\mathbf{Z}^{+}\mathbf{Y}\mathbf{C}.
$$

This step estimates how retained predictor scores map to the selected response scores. It avoids
forming an explicit inverse; the implementation uses a least-squares solver.

The regression map before diagonalization is

$$
\mathbf{B}_{\mathrm{cs}}
=
\mathbf{\Pi}\mathbf{W}\mathbf{C}^{\mathsf T}.
$$

The fitted core-coordinate responses are therefore

$$
\widehat{\mathbf{Y}}
=
\mathbf{X}\mathbf{B}_{\mathrm{cs}}
=
\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf T}.
$$

These two prediction expressions must remain numerically equivalent.

## Step 4: diagonalize the latent coupling

Take the economical SVD of the latent regression matrix:

$$
\mathbf{W}=\mathbf{M}\mathbf{D}\mathbf{N}^{\mathsf T},
$$

where

$$
\mathbf{M}^{\mathsf T}\mathbf{M}=\mathbf{I}_h,
\qquad
\mathbf{N}^{\mathsf T}\mathbf{N}=\mathbf{I}_h,
$$

and

$$
\mathbf{D}=\operatorname{diag}(D_1,\ldots,D_h),
\qquad
D_1\ge D_2\ge\cdots\ge D_h\ge0.
$$

Define

$$
\mathbf{P}=\mathbf{\Pi}\mathbf{M},
\qquad
\mathbf{Q}=\mathbf{C}\mathbf{N}.
$$

Because both factors are products of matrices with orthonormal columns,

$$
\mathbf{P}^{\mathsf T}\mathbf{P}=\mathbf{I}_h,
\qquad
\mathbf{Q}^{\mathsf T}\mathbf{Q}=\mathbf{I}_h.
$$

The projected regression relation becomes

$$
\mathbf{Y}\mathbf{Q}
=
\mathbf{X}\mathbf{P}\mathbf{D}
+
\mathbf{E}_{\pi},
$$

with $\mathbf{E}_{\pi}=\mathbf{E}''\mathbf{N}$. Orthogonal rotation preserves the residual Frobenius norm and rotates the residual covariance
without changing its eigenvalues.

This is the panoramic one-to-one representation. Column $j$ of $\mathbf{X}\mathbf{P}$ is coupled
only to column $j$ of $\mathbf{Y}\mathbf{Q}$, with coupling strength $D_j$. There are no
cross-coupling terms between distinct latent modes in these coordinates.

## Regression-map identities and prediction

The centered/scaled regression map has two equivalent factorizations:

$$
\boxed{
\mathbf{B}_{\mathrm{cs}}
=
\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}
=
\mathbf{\Pi}\mathbf{W}\mathbf{C}^{\mathsf T}
}
$$

and hence

$$
\widehat{\mathbf{Y}}
=
\mathbf{X}\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}.
$$

For a new core-coordinate predictor matrix $\mathbf{X}_{\mathrm{new}}$,

$$
\widehat{\mathbf{Y}}_{\mathrm{new}}
=
\mathbf{X}_{\mathrm{new}}\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}.
$$

In the public estimator, let $\boldsymbol\mu_X$, $\boldsymbol\mu_Y$ be training means and let
$\mathbf{s}_X$, $\mathbf{s}_Y$ be the safe training scales. Then

$$
\mathbf{X}_{\mathrm{cs}}
=
(\mathbf{X}-\boldsymbol\mu_X)\operatorname{diag}(\mathbf{s}_X)^{-1},
$$

and predictions are transformed back to response units. Equivalently, the original-unit
coefficient matrix is

$$
\mathbf{B}
=
\operatorname{diag}(\mathbf{s}_X)^{-1}
\mathbf{B}_{\mathrm{cs}}
\operatorname{diag}(\mathbf{s}_Y),
$$

with intercept

$$
\mathbf{b}_0
=
\boldsymbol\mu_Y-\boldsymbol\mu_X\mathbf{B}.
$$

When `scale=False`, the scale vectors are ones and the same expressions reduce to centering only.

## Interpretation of the two ranks

### Predictor rank $r_\pi$

$r_\pi$ controls how much predictor-side variation is retained before response information is
used. It is a regularization and signal-preservation parameter.

- If $r_\pi$ is too small, predictive directions may be irreversibly removed.
- Once all predictive directions are retained, additional predictor directions may have little
  effect because the response-subspace step can assign them negligible cross-covariance weight.
- This creates an important asymmetry: under-specification can destroy signal, whereas moderate
  over-specification may mainly increase computation or variance.

The manuscript reports an empirical sharp-decrease-then-plateau pattern in CV error as $r_\pi$
increases. This pattern motivates practical upper bounds and adaptive search, but it is not a
proof that every dataset has a unimodal or perfectly flat CV curve.

### Latent dimension $h$

$h$ controls the rank and interpretive complexity of the fitted predictor-response coupling.
It is the number of paired orthogonal modes retained after the predictor subspace has been fixed.
Increasing $h$ cannot recover predictor directions excluded by $r_\pi$.

The admissible relation $h\le r_\pi$ is structural: the latent regression matrix
$\mathbf{W}\in\mathbb{R}^{r_\pi\times h}$ cannot support more independent coupled modes than the
retained predictor dimension. Also $h\le q$ because the response latent basis has $q$ rows.

## Model size

The factorization uses

- $ph$ entries in $\mathbf{P}$;
- $h$ diagonal entries in $\mathbf{D}$;
- $qh$ entries in $\mathbf{Q}$.

After accounting for the orthonormality constraints on $\mathbf{P}$ and $\mathbf{Q}$, the
manuscript gives the effective model dimension

$$
(p+q-h)h.
$$

This is substantially smaller than the $pq$ coefficients of unconstrained multivariate OLS when
$h\ll\min(p,q)$.

## Limiting and comparative cases

### OLS limit

If the predictor truncation preserves the estimable predictor row space and $h$ retains every
estimable response-side direction, then

$$
\widehat{\mathbf{Y}}
=
\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf T}
$$

coincides with the multivariate OLS fitted response. In that rank-preserving case, Pi-PLS is an
orthogonal latent-coordinate representation of the same fitted map rather than a different fit.

### Relation to reduced-rank regression

Both methods produce low-rank coefficient structures. Pi-PLS differs by explicitly selecting a
predictor signal subspace first, selecting response directions through
$\mathbf{Z}^{\mathsf T}\mathbf{Y}$, and then rotating the latent regression into a diagonal
one-to-one coupling.

### Relation to PLS and PLS-SVD

PLS and Pi-PLS both exploit predictor-response covariance. Classical PLS is usually constructed
through iterative component extraction and deflation. Pi-PLS instead uses a closed sequence of
SVD and least-squares operations.

PLS-SVD derives both sides directly from a cross-covariance operator. Pi-PLS first fixes the
retained predictor subspace from the singular structure of $\mathbf{X}$, then derives response
directions from cross-covariance with that retained subspace. Pi-PLS can therefore preserve
predictor variation not aligned with the leading raw cross-covariance modes.

## Identifiability and invariance

Individual singular vectors are not generally identifiable mathematical objects.

- Every singular-vector pair is sign-indeterminate.
- Repeated singular values permit arbitrary orthogonal rotations within the corresponding
  invariant subspace.
- Nearly repeated singular values can make individual columns numerically unstable even when the
  subspace and regression map are stable.
- A zero dilation value makes its associated paired directions irrelevant to prediction and
  generally non-identifiable.

Consequently, scientific and regression tests should compare:

- projection matrices such as $\mathbf{P}\mathbf{P}^{\mathsf T}$;
- principal angles between subspaces;
- ordered singular or dilation values;
- the regression map $\mathbf{B}_{\mathrm{cs}}$;
- fitted values and predictions;
- reconstruction identities.

Tests should not require raw singular-vector columns to match a fixed sign or basis orientation.

## Numerical-rank and degeneracy policy

The mathematical bound $r_\pi\le\min(n,p)$ is insufficient for finite-precision computation. The
private core estimates predictor numerical rank from the thin-SVD singular values $s_i$ using

$$
\tau_X=\max(n,p)\,\epsilon_{64}\,s_1,
$$

and counts singular values satisfying $s_i>\tau_X$. A requested $r_\pi$ above this numerical rank
is invalid and raises an error rather than being silently clamped.

Constant columns, rank-deficient matrices, $p\gg n$, singleton CV training folds, and repeated
singular values are expected boundary cases. Their exact handling is specified in
`.llm/numerical_contracts.md`.

## Predictor-rank selection: theory versus API policy

The manuscript discusses $r_\pi$ as the dimension that must be large enough to retain the full
predictor-response coupling. It reports that under-specification causes irreversible loss, while
moderate over-specification often lies on a broad predictive plateau. It also discusses
sample-size heuristics and exhaustive CV as practical selection approaches.

The repository API uses the full-sample-supported, fold-feasible upper bound

$$
r_{\pi,\max}
=
\min\left(
 p,
 n_{\mathrm{train,min}}-1,
 \left\lceil
 \frac{n}
 {\texttt{samples\_per\_predictor\_rank}}
 \right\rceil
\right).
$$

This bound is an API and regularization policy, not a theorem of Pi-PLS. The total supplied sample
count $n$ defines statistical support for the final model, which is refitted on all supplied rows.
For internal CV, $n_{\mathrm{train,min}}$ is the smallest materialized training-fold size. Because
each candidate centers its training predictors, $n_{\mathrm{train,min}}-1$ is a hard feasibility
cap. Using the public sample count in the support term does not fit any quantity from $X$ or $Y$
outside the training folds.

The fixed estimator accepts one explicit integer $r_\pi$. Rule-derived ceilings and adaptive or
exhaustive rank search belong to `PiPLSPathCV`, which evaluates fixed-estimator clones on the
admissible triangular surface.

Adaptive search reports every evaluated rank and does not guarantee the exhaustive optimum for
an arbitrary non-unimodal CV curve. Search approximation and linear-algebra approximation are
separate. The implemented `svd_solver` policy permits randomized approximation only for the first
predictor-matrix SVD; the response-subspace and coupling SVDs remain exact. Automatic solver
selection is conservative and depends on matrix dimensions and retained-rank fraction, while
explicit `"full"` remains the reference path.

## Cross-validation consequences

Any CV-based selection of $r_\pi$ or $h$ must obey the following theoretical separation:

1. fit centering and scaling only on the training fold;
2. derive predictor and response subspaces only from the training fold;
3. predict the validation fold using that fold-trained model;
4. reuse the same materialized split set for every candidate being compared;
5. derive the samples-per-rank support term from total supplied $n$, then cap candidates by every
   centered training fold's feasible dimensions;
6. refit the selected fixed-parameter model once on all data supplied to `fit()`.

Response-standardized MSE is used so response columns with different physical scales contribute
comparably. The scale for each response must be estimated from the corresponding training fold,
not from the validation fold or the complete dataset.

## Core algorithm summary

For centered or centered-and-scaled $\mathbf{X}$ and $\mathbf{Y}$:

1. compute a thin SVD of $\mathbf{X}$;
2. set $\mathbf{\Pi}$ to the leading $r_\pi$ right singular vectors;
3. form $\mathbf{Z}=\mathbf{X}\mathbf{\Pi}$;
4. compute a thin SVD of $\mathbf{Z}^{\mathsf T}\mathbf{Y}$;
5. set $\mathbf{C}$ to its leading $h$ right singular vectors;
6. solve $\mathbf{Z}\mathbf{W}\approx\mathbf{Y}\mathbf{C}$ by least squares;
7. compute the economical SVD $\mathbf{W}=\mathbf{M}\mathbf{D}\mathbf{N}^{\mathsf T}$;
8. set $\mathbf{P}=\mathbf{\Pi}\mathbf{M}$ and $\mathbf{Q}=\mathbf{C}\mathbf{N}$;
9. return the factorization and regression map
   $\mathbf{B}_{\mathrm{cs}}=\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$.

## Implementation review checklist

A change to the core theory or its implementation should preserve or explicitly revise all of the
following:

- $1\le h\le\min(r_\pi,q)$;
- $r_\pi$ does not exceed predictor numerical rank;
- $\mathbf{\Pi}^{\mathsf T}\mathbf{\Pi}=\mathbf{I}$;
- $\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}$;
- $\mathbf{P}^{\mathsf T}\mathbf{P}=\mathbf{I}$;
- $\mathbf{Q}^{\mathsf T}\mathbf{Q}=\mathbf{I}$;
- $D_1\ge\cdots\ge D_h\ge0$;
- $\mathbf{B}_{\mathrm{cs}}=\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$;
- $\mathbf{B}_{\mathrm{cs}}=\mathbf{\Pi}\mathbf{W}\mathbf{C}^{\mathsf T}$;
- predictions from the two factorizations agree;
- singular-vector signs and repeated-subspace bases are not treated as identifiers;
- preprocessing used in selection is fit within each training fold;
- rank-search approximation is not conflated with SVD approximation.

## Scope boundaries

This file explains the linear Pi-PLS method currently targeted by the repository. It does not
specify nonlinear kernels, sparsity penalties, missing-data algorithms, probabilistic inference,
uncertainty intervals, or causal interpretation. Such extensions require separate scientific
decisions and must not be inferred from the present factorization.
