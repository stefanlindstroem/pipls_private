# Π-PLS theory overview

Π-PLS is a multivariate linear-regression method with two explicit rank controls. It first retains
a rank-controlled predictor subspace, then constructs a response subspace under a configured
selection criterion, and finally diagonalizes the reduced regression map into paired
predictor-response modes. The package default uses the cross-covariance construction from the
[peer-reviewed companion publication](citation.md#companion-paper); programming users may instead
request a least-squares-driven response-subspace construction.

## Scientific source and package scope

The scientific source for the construction summarized on this page is the
[companion manuscript](citation.md#companion-paper):

> Vishal Agrawal, Fritjof Nilsson, and Stefan B. Lindström, “Panoramic Partial Least Squares
> (Pi-PLS): Transparent, parsimonious, and more interpretable multivariate regression model.”
> Manuscript under revision at *Computers & Chemical Engineering*, CACE-D-26-00847.

The manuscript states the fixed mathematical core for centered predictor and response matrices,
including the cross-covariance response-subspace construction. The package implements that
peer-reviewed construction as its default and adds ordinary software facilities around it, including
optional scaling, numerical-rank checks, configurable predictor SVD solvers, cross-validated search,
immutable result records, and prediction diagnostics. The package also provides one
least-squares/RRR-inspired response-subspace construction for programming users. That alternative is
a software extension and is not part of the
[peer-reviewed companion publication](citation.md#companion-paper).

The publication/software boundary matters when interpreting results: manuscript-aligned analyses use
the cross-covariance response-subspace construction, while analyses using the least-squares option
should identify that model choice explicitly.

The Pulp source paper documents the provenance and scientific context of the
[Pulp dataset](datasets.md#pulp-real-data-integration); it is not the theoretical reference for
Π-PLS. Pulp provenance and analysis are covered in the
[reference-dataset guide](datasets.md) and [Pulp tutorial](tutorials/pulp.md). Citation metadata for
the software and companion manuscript is maintained under the
[companion-paper citation](citation.md#companion-paper).

## Problem setting and two rank controls {#problem-setting-and-two-rank-controls}

Let $\mathbf{X}\in\mathbb{R}^{n\times p}$ contain column-centered predictors and let
$\mathbf{Y}\in\mathbb{R}^{n\times q}$ contain column-centered responses. Multivariate linear
regression seeks
$\mathbf{B}\in\mathbb{R}^{p\times q}$ such that

\begin{equation}
\mathbf{Y}=\mathbf{X}\mathbf{B}+\mathbf{E}.
\end{equation}

Π-PLS separates two structural choices:

- the predictor rank $r_\pi$, which controls the dimension of the retained predictor subspace;
- the component count $h$, which controls the number of paired predictor-response modes.

The admissible dimensions satisfy

\begin{equation}
1\leq h\leq\min(r_\pi,q),
\qquad
r_\pi\leq\min(n,p).
\end{equation}

For a centered estimator fit, the effective sample-space bound becomes
$r_\pi\leq\min(n-1,p)$. The implementation also requires $r_\pi$ not to exceed the verified numerical
rank of the preprocessed predictor matrix.

## Canonical terminology {#canonical-terminology}

The package uses the following terms for the fixed Π-PLS construction:

| Object or public name | Canonical meaning |
|---|---|
| $\mathbf{\Pi}$ | retained predictor basis |
| $\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}$ | retained-subspace projector |
| $\mathbf{P}$ | orthonormal predictor directions |
| $\mathbf{Q}$ | orthonormal response directions |
| $D_k=D_{kk}$ | dilation of paired latent mode $k$ |
| $\mathbf{X}\mathbf{P}$ | predictor scores |
| $\mathbf{Y}\mathbf{Q}$ | response scores |
| $(P_{:k},D_k,Q_{:k})$ | paired latent mode $k$ |
| `predictor_rank` | retained predictor-subspace dimension $r_\pi$ |
| `n_components` | number of paired latent modes $h$ |

The word **direction** is the primary mathematical term for columns of $\mathbf{P}$ and
$\mathbf{Q}$. `PiPLSDecomposition` exposes these arrays as `predictor_directions` and
`response_directions`. The estimator retains the standard PLS-style names `x_rotations_` and
`y_rotations_`; none of these names makes $\mathbf{P}$ or $\mathbf{Q}$ a projection matrix. The
projectors onto the final direction spans are $\mathbf{P}\mathbf{P}^{\mathsf T}$ and
$\mathbf{Q}\mathbf{Q}^{\mathsf T}$.

The Π-PLS directions are also distinct from `x_loadings_` and `y_loadings_`, which are
least-squares reconstruction loadings for the centered or centered-and-scaled training blocks.
For response-side factor displays, the package stores response-by-mode weighted directions
$\mathbf{Q}\mathbf{D}$, where column $k$ is $D_kQ_{:k}$. The manuscript's mode-by-response
orientation is the transpose:

\begin{equation}
\mathbf{D}\mathbf{Q}^{\mathsf T}=(\mathbf{Q}\mathbf{D})^{\mathsf T}.
\end{equation}

The API word “component” is retained because it is familiar in regression software. In Π-PLS,
`n_components` counts paired latent modes, not predictor-SVD directions and not synthetic latent
components. Likewise, `predictor_rank` is a retained observed-subspace dimension; “predictor signal
rank” is reserved for synthetic settings where the noiseless generating rank is known.

## 1. Rank-controlled predictor projection {#rank-controlled-predictor-projection}

Take a singular value decomposition

\begin{equation}
\mathbf{X}=\mathbf{U}_{\mathrm{X}}\mathbf{S}_{\mathrm{X}}\mathbf{V}_{\mathrm{X}}^{\mathsf T}
\end{equation}

and retain the leading $r_\pi$ right singular vectors:

\begin{equation}
\mathbf{\Pi}=\mathbf{V}_{\mathrm{X}}(:,1:r_\pi)\in\mathbb{R}^{p\times r_\pi},
\qquad
\mathbf{\Pi}^{\mathsf T}\mathbf{\Pi}=\mathbf{I}_{r_\pi}.
\end{equation}

The matrix $\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}$ is the orthogonal projector onto the retained
predictor subspace, so

\begin{equation}
\mathbf{X}=\mathbf{X}\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}+\mathbf{X}(\mathbf{I}_p-\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}).
\end{equation}

Define the retained predictor scores

\begin{equation}
\mathbf{Z}=\mathbf{X}\mathbf{\Pi}\in\mathbb{R}^{n\times r_\pi}.
\end{equation}

The first term, $\mathbf{Z}\mathbf{\Pi}^{\mathsf T}$, is the rank-$r_\pi$ predictor approximation.
The second term is orthogonal to the retained subspace and contributes a truncation residual.
Substituting the decomposition into the regression relation gives

\begin{equation}
\mathbf{Y}=\mathbf{Z}\mathbf{\Pi}^{\mathsf T}\mathbf{B}+\mathbf{E}',
\qquad
\mathbf{E}'=\mathbf{E}+\mathbf{X}(\mathbf{I}_p-\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T})\mathbf{B}.
\end{equation}

This step is response-independent. Any predictive direction removed by the choice of $r_\pi$ cannot
be recovered later by increasing $h$.

## 2. Response-subspace selection {#response-subspace-selection}

After fixing the retained predictor representation $\mathbf{Z}=\mathbf{X}\mathbf{\Pi}$, the
package supports two criteria for selecting an $h$-dimensional orthonormal response basis
$\mathbf{C}\in\mathbb{R}^{q\times h}$. Both choices feed the same later least-squares coupling
and diagonalization. They differ only in how the intermediate response subspace is chosen.

### 2.1 Cross-covariance selection

The [peer-reviewed companion publication](citation.md#companion-paper) uses the cross-covariance
construction, which is also the package default (`response_subspace="cross_covariance"`). Define

\begin{equation}
\boldsymbol{\Sigma}_{\mathrm{ZY}}=\mathbf{Z}^{\mathsf T}\mathbf{Y}\in\mathbb{R}^{r_\pi\times q}.
\end{equation}

For prescribed $h$, the response basis is chosen by

\begin{equation}
\mathbf{C}_{\mathrm{cov}}
=
\arg\max_{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h}
\left\|\boldsymbol{\Sigma}_{\mathrm{ZY}}\mathbf{C}\right\|_{\mathrm{F}}^2.
\end{equation}

Because

\begin{equation}
\left\|\boldsymbol{\Sigma}_{\mathrm{ZY}}\mathbf{C}\right\|_{\mathrm{F}}^2
=
\operatorname{tr}\!\left(
\mathbf{C}^{\mathsf T}
\boldsymbol{\Sigma}_{\mathrm{ZY}}^{\mathsf T}
\boldsymbol{\Sigma}_{\mathrm{ZY}}
\mathbf{C}
\right),
\end{equation}

this is an orthonormal trace-maximization problem. If

\begin{equation}
\boldsymbol{\Sigma}_{\mathrm{ZY}}=\mathbf{U}_{\mathrm{cov}}\mathbf{S}_{\mathrm{cov}}\mathbf{V}_{\mathrm{cov}}^{\mathsf T},
\end{equation}

then the leading $h$ right singular vectors span an optimum:

\begin{equation}
\mathbf{C}_{\mathrm{cov}}=\mathbf{V}_{\mathrm{cov}(:,1:h)}.
\end{equation}

Thus the default policy selects the response subspace with the largest retained squared
cross-covariance with $\mathbf{Z}$. The maximizing subspace is the meaningful object; individual
basis vectors are not unique when relevant singular values are repeated.

### 2.2 Least-squares selection

The alternative `response_subspace="least_squares"` chooses the response subspace jointly with a
rank-$h$ reduced regression map by solving

\begin{equation}
(\mathbf{C}_{\mathrm{LS}},\mathbf{W}_{\mathrm{LS}})
=
\arg\min_{\substack{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h\\
\mathbf{W}\in\mathbb{R}^{r_\pi\times h}}}
\left\|\mathbf{Y}-\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf T}\right\|_{\mathrm{F}}^2.
\end{equation}

For fixed $\mathbf{C}$, the minimizing map is

\begin{equation}
\mathbf{W}=\mathbf{Z}^{+}\mathbf{Y}\mathbf{C}.
\end{equation}

Let

\begin{equation}
\mathbf{P}_{\mathbf{Z}}=\mathbf{Z}\mathbf{Z}^{+}
\end{equation}

be the orthogonal projector onto the retained predictor-score column space. Eliminating
$\mathbf{W}$ gives the equivalent response-subspace problem

\begin{equation}
\mathbf{C}_{\mathrm{LS}}
=
\arg\max_{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h}
\operatorname{tr}\!\left(
\mathbf{C}^{\mathsf T}
\mathbf{Y}^{\mathsf T}
\mathbf{P}_{\mathbf{Z}}
\mathbf{Y}
\mathbf{C}
\right).
\end{equation}

This is the rank-$h$ reduced-rank-regression response subspace for regression of $\mathbf{Y}$ on
the fixed retained predictor coordinates $\mathbf{Z}$. Consequently, for the same fixed
$(h,r_\pi)$, the least-squares policy minimizes the training Frobenius residual over admissible
rank-$h$ maps. That property does not imply lower cross-validated or external prediction error.

The package evaluates this criterion without forming normal equations. With a reduced QR
factorization

\begin{equation}
\mathbf{Z}=\mathbf{Q}_{\mathbf{Z}}\mathbf{R}_{\mathbf{Z}},
\qquad
\mathbf{Q}_{\mathbf{Z}}^{\mathsf T}\mathbf{Q}_{\mathbf{Z}}=\mathbf{I}_{r_\pi},
\end{equation}

we have

\begin{equation}
\mathbf{P}_{\mathbf{Z}}
=
\mathbf{Q}_{\mathbf{Z}}\mathbf{Q}_{\mathbf{Z}}^{\mathsf T},
\end{equation}

and therefore

\begin{equation}
\mathbf{Y}^{\mathsf T}\mathbf{P}_{\mathbf{Z}}\mathbf{Y}
=
(\mathbf{Q}_{\mathbf{Z}}^{\mathsf T}\mathbf{Y})^{\mathsf T}
(\mathbf{Q}_{\mathbf{Z}}^{\mathsf T}\mathbf{Y}).
\end{equation}

The columns of $\mathbf{C}_{\mathrm{LS}}$ are thus the leading right singular directions of
$\mathbf{Q}_{\mathbf{Z}}^{\mathsf T}\mathbf{Y}$. The response-side QR/SVD route remains exact;
the configurable `svd_solver` applies only to construction of the predictor basis $\mathbf{\Pi}$.

> **Software-extension boundary.** The least-squares response-subspace construction is implemented
> for programming users but is not part of the
> [peer-reviewed companion publication](citation.md#companion-paper).
> Manuscript-aligned analyses use `response_subspace="cross_covariance"`.

The two criteria can differ substantially because they weight the retained predictor directions
differently. Since the SVD-based retained scores can be written as

\begin{equation}
\mathbf{Z}=\mathbf{U}_{r}\mathbf{S}_{r},
\end{equation}

the cross-covariance criterion depends on the eigenspace of

\begin{equation}
\mathbf{Y}^{\mathsf T}
\mathbf{U}_{r}\mathbf{S}_{r}^{2}\mathbf{U}_{r}^{\mathsf T}
\mathbf{Y},
\end{equation}

whereas the least-squares criterion depends on

\begin{equation}
\mathbf{Y}^{\mathsf T}
\mathbf{U}_{r}\mathbf{U}_{r}^{\mathsf T}
\mathbf{Y}.
\end{equation}

The cross-covariance construction therefore retains weighting by predictor singular-value
magnitude, while the least-squares construction depends on the retained predictor column space.
When $q=1$, the two policies give the same fitted regression map. They also give the same fitted map
when $h=q\leq r_\pi$, because the complete response space is retained.

## 3. Least-squares coupling in the selected response subspace

Under either response-subspace policy, project the response matrix onto the selected basis and fit

\begin{equation}
\mathbf{Y}\mathbf{C}=\mathbf{Z}\mathbf{W}+\mathbf{E}'',
\end{equation}

where $\mathbf{W}\in\mathbb{R}^{r_\pi\times h}$. The minimum-norm least-squares solution is

\begin{equation}
\mathbf{W}=\mathbf{Z}^{+}\mathbf{Y}\mathbf{C}.
\end{equation}

Thus the phrase “least-squares response-subspace selection” refers specifically to the criterion
used to choose $\mathbf{C}$; estimation of $\mathbf{W}$ by least squares is common to both
policies.

Before diagonalization, the regression map in the centered coordinates is

\begin{equation}
\mathbf{B}_{\mathrm{cs}}=\mathbf{\Pi}\mathbf{W}\mathbf{C}^{\mathsf T},
\end{equation}

and the fitted responses are

\begin{equation}
\widehat{\mathbf{Y}}=\mathbf{X}\mathbf{B}_{\mathrm{cs}}=\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf T}.
\end{equation}

The implementation uses a least-squares solver rather than forming an explicit inverse.

## 4. Diagonal latent coupling {#diagonal-latent-coupling}

Take the economical singular value decomposition

\begin{equation}
\mathbf{W}=\mathbf{M}\mathbf{D}\mathbf{N}^{\mathsf T},
\end{equation}

where $\mathbf{M}\in\mathbb{R}^{r_\pi\times h}$ and
$\mathbf{N}\in\mathbb{R}^{h\times h}$ have orthonormal columns,
and

\begin{equation}
\mathbf{D}=\operatorname{diag}(D_1,\ldots,D_h),
\qquad
D_1\geq D_2\geq\cdots\geq D_h\geq0.
\end{equation}

Define

\begin{equation}
\mathbf{P}=\mathbf{\Pi}\mathbf{M}\in\mathbb{R}^{p\times h},
\qquad
\mathbf{Q}=\mathbf{C}\mathbf{N}\in\mathbb{R}^{q\times h}.
\end{equation}

Then

\begin{equation}
\mathbf{P}^{\mathsf T}\mathbf{P}=\mathbf{I}_h,
\qquad
\mathbf{Q}^{\mathsf T}\mathbf{Q}=\mathbf{I}_h,
\end{equation}

and the reduced regression relation becomes

\begin{equation}
\mathbf{Y}\mathbf{Q}=\mathbf{X}\mathbf{P}\mathbf{D}+\mathbf{E}_\pi,
\qquad
\mathbf{E}_\pi=\mathbf{E}''\mathbf{N}.
\end{equation}

The predictor score vector $\mathbf{X}P_{:k}$ is coupled only to the response score
vector $\mathbf{Y}Q_{:k}$, with dilation $D_k$. This is the one-to-one, mode-wise interpretation
central to Π-PLS. Orthogonal rotation by $\mathbf{N}$ preserves the Frobenius norm of the
residual and orthogonally transforms its covariance; it preserves covariance
eigenvalues but does not generally leave the covariance matrix itself unchanged.

The equivalent regression-map factorizations are

\begin{equation}
\boxed{
\mathbf{B}_{\mathrm{cs}}
=\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}
=\mathbf{\Pi}\mathbf{W}\mathbf{C}^{\mathsf T}
}.
\end{equation}

For new centered predictors $\mathbf{X}_{\mathrm{new}}$,

\begin{equation}
\widehat{\mathbf{Y}}_{\mathrm{new}}=\mathbf{X}_{\mathrm{new}}\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}.
\end{equation}

## Why the method is panoramic {#why-the-method-is-panoramic}

Standard deflation-based PLS algorithms construct successive components while removing previously
modelled predictor variation. Π-PLS instead fixes one rank-controlled predictor representation
$\mathbf{Z}=\mathbf{X}\mathbf{\Pi}$ and derives all $h$ coupled modes from that undeflated retained
space through closed-form matrix decompositions. The retained predictor space therefore remains
available as a whole during response-subspace selection and regression; in this sense, the view is
panoramic.

## Interpretation of $r_\pi$ and $h$ {#interpretation-of-the-ranks}

The two ranks control different forms of complexity.

The predictor rank $r_\pi$ determines which predictor variation remains available to the model. If
$r_\pi$ is too small, response-relevant variation can be discarded irreversibly. Retaining more
predictor directions does not force all of them to contribute strongly, because the response
subspace and latent least-squares map are estimated afterward.

The component count $h$ determines the rank and interpretive complexity of the diagonal
predictor-response coupling. Increasing $h$ adds paired modes, but it cannot recover predictor
directions excluded by $r_\pi$ and cannot exceed the number of responses.

The phrase “predictor signal rank” is appropriate for a synthetic data-generating model when its
noiseless rank is known. For an observed noisy predictor matrix, $\mathbf{\Pi}$ is more accurately
described as a retained predictor basis: leading observed singular directions are not guaranteed to
separate signal from noise exactly.

## Nominal fitted dimension

After $\mathbf{\Pi}$ has been fixed, the fitted representation is described by $\mathbf{M}$,
$\mathbf{D}$, and $\mathbf{Q}$. These contain $r_\pi h$, $h$, and $qh$ entries. The orthonormality
constraints on $\mathbf{M}$ and $\mathbf{Q}$ each remove $h(h+1)/2$ degrees of freedom. The resulting
nominal fitted dimension is

\begin{equation}
\boxed{(r_\pi+q-h)h}.
\end{equation}

This count concerns the fitted representation after the retained predictor basis has been fixed. It
does not treat the $ph$ entries of $\mathbf{P}=\mathbf{\Pi}\mathbf{M}$ as independently free
parameters.

## Relationships to established methods {#relationships-to-established-methods}

### Ordinary least squares {#ordinary-least-squares}

If predictor truncation preserves the estimable predictor row space and $h$ retains every estimable
response-side direction, then

\begin{equation}
\widehat{\mathbf{Y}}=\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf T}
\end{equation}

coincides with the multivariate OLS fitted response. In that limit, Π-PLS is an orthogonal latent
reparameterization of the same fitted map.

### Reduced-rank regression {#reduced-rank-regression}

Both Π-PLS and reduced-rank regression produce low-rank coefficient structures. Reduced-rank
regression obtains the best rank-$h$ approximation of the OLS fit under its least-squares
criterion. With `response_subspace="least_squares"`, Π-PLS applies exactly that reduced-rank
principle after first fixing the retained predictor representation $\mathbf{Z}$, and then
diagonalizes the resulting reduced map into one-to-one paired modes. With the default
`"cross_covariance"` policy, the response subspace is instead selected by the
[peer-reviewed cross-covariance criterion](citation.md#companion-paper) before the common
least-squares coupling and diagonalization stages.

### Canonical correlation analysis {#canonical-correlation-analysis}

CCA also constructs paired predictor and response variates with a diagonal association structure,
but classical CCA maximizes normalized correlation after within-block whitening. The default
Π-PLS response policy instead uses an unwhitened cross-covariance criterion inside the retained
predictor representation, while the optional least-squares policy uses the fitted-response
least-squares criterion described above. Both policies then form the same diagonal paired-mode
representation. The structural analogy to CCA therefore concerns the final paired relation, not an
identity of objectives.

### PLS and PLS-SVD {#pls-and-pls-svd}

Standard multicomponent PLS is commonly constructed through iterative covariance-guided extraction
and deflation. Π-PLS instead fixes one retained predictor space and then selects a response
subspace before fitting and diagonalizing the reduced least-squares map. Under the publication
default, response-subspace selection is itself cross-covariance-driven; under the optional
least-squares policy, it is RRR-inspired.

PLS-SVD derives predictor and response directions directly from a cross-covariance operator. Under
the default Π-PLS policy, $\mathbf{Z}^{\mathsf T}\mathbf{Y}$ selects only the intermediate response
subspace after $\mathbf{Z}=\mathbf{X}\mathbf{\Pi}$ has already been determined from predictor
singular structure. Under the least-squares policy, that intermediate subspace is instead selected
from the predictable response variation in $\operatorname{col}(\mathbf{Z})$. In both cases, the
final $\mathbf{P}$ and $\mathbf{Q}$ arise only after the common least-squares map $\mathbf{W}$ is
estimated and diagonalized.

## Package realization

`PiPLSRegression` learns predictor and response means from the training data. Predictor and
response scaling are controlled independently after centering. The backward-compatible `scale`
parameter supplies the default for both blocks, while non-`None` `scale_x` and `scale_y` values
override predictor and response scaling respectively. Enabled scaling uses training-sample standard
deviations. `response_subspace` selects either the publication-default `"cross_covariance"`
response basis or the software-extension `"least_squares"` basis. The estimator applies the
corresponding fixed construction above in the resulting centered or centered-and-scaled coordinates,
then transforms the regression map back to original units for `coef_`, `intercept_`, and `predict()`.

A fitted estimator exposes the Π-PLS-specific factorization in `decomposition_`:

- `predictor_directions`: $\mathbf{P}$;
- `dilation`: $(D_1,\ldots,D_h)$;
- `response_directions`: $\mathbf{Q}$;
- `standardized_regression_map`: $\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$;
- numerical-rank and resolved predictor-SVD diagnostics.

The construction matrices $\mathbf{\Pi}$, $\mathbf{C}$, and $\mathbf{W}$ remain private.
Conventional scores, reconstruction loadings, coefficients, and transformations are exposed
separately from the $\mathbf{P}$, $\mathbf{D}$, and $\mathbf{Q}$ factorization.

Singular-vector signs are arbitrary. Repeated or nearly repeated singular values identify invariant
subspaces rather than unique ordered columns. Numerical validation should therefore compare
regression maps, predictions, projections, principal angles, or singular values rather than raw
basis columns.

## Selection, validation, and synthetic-data boundaries {#selection-validation-and-synthetic-data-boundaries}

The equations above define one fixed $(h,r_\pi)$ model under one configured response-subspace
policy. `PiPLSSearchCV` is a package-level facility for evaluating admissible fixed models under
fold-local preprocessing and response-standardized CV-MSE. It searches component count and
predictor rank; it does not automatically compare response-subspace policies. A controlled
programming-user comparison therefore uses two estimator templates with the same materialized CV
splits. Its search policies and the practical real-data workflows documented elsewhere are not
redefined by this theory page and need not duplicate the manuscript’s experimental choices.

`make_synthetic_data()` provides the package's deterministic synthetic-data utility and implements
the [companion manuscript’s](citation.md#companion-paper) Gaussian latent data-generating model
directly. See the [companion-manuscript synthetic-data guide](manuscript_reproduction.md)
for the exact distribution and reproducibility boundary, and the
[dataset and generator guide](datasets.md) for the package data interface.

The [Pulp tutorial’s fixed-rank selection section](tutorials/pulp.md#retrieve-selection-evidence)
and [Path and selection](path_selection.md) define the package’s general path-construction,
validation, and selection contracts.
