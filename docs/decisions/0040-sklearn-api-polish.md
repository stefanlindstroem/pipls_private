# Decision 0040: final scikit-learn API polish

## Status

Accepted and implemented.

## Context

The fixed-estimator and path-search boundary is complete, but a final pre-release review found four
small inconsistencies with ordinary scikit-learn use:

- the complete component path was represented by `n_components_values=None`, which did not state
  the user's intent visibly;
- randomized SVD accepted a narrower `random_state` contract than ordinary scikit-learn
  estimators;
- the default response-standardized scorer was exposed through a package-local string rather than
  directly as its public callable;
- Pi-PLS factorization matrices and numerical diagnostics were duplicated between
  `decomposition_` and many top-level fitted aliases.

The search meta-estimator also needs refit-dependent methods to follow the availability of the
selected refitted estimator without adding an output-configuration layer of its own.

## Decision

`PiPLSSearchCV.n_components_values` accepts only:

- `"all"`, meaning every admissible component count from 1 through
  `min(n_targets, max_predictor_rank_)`;
- a nonempty sequence of distinct positive integers.

`"all"` is the default. It makes the existing complete-path behavior explicit and avoids repeated
user-side dimension calculations. `None` is not retained as an alias because the package is
pre-1.0 and the explicit sentinel is clearer.

`PiPLSRegression.random_state` follows the conventional scikit-learn forms: a valid integer seed,
a NumPy `RandomState` instance, or `None`. The default remains `0`, so the default randomized path
is reproducible. `None` uses NumPy's global random state and is therefore not promised to be
repeatable.

`PiPLSSearchCV.scoring` defaults directly to the public callable
`pipls.metrics.neg_response_standardized_mean_squared_error`. Ordinary scikit-learn scorer names,
other scorer callables, and `None` remain supported. The package does not register a private scorer
name that appears portable to unrelated scikit-learn search objects.

`PiPLSRegression.decomposition_` is the single Pi-PLS-specific factorization and numerical-
diagnostic result. Standard PLS-style fitted attributes remain on the estimator, but duplicate
symbolic and diagnostic aliases are removed:

```text
Pi_, C_, W_, P_, D_, Q_, dilation_
coef_matrix_
x_rank_, x_rank_is_exact_, rank_tolerance_, svd_solver_
```

Their values remain available through `decomposition_`; prediction coefficients remain available
through standard `coef_` and `intercept_`.

Refit-dependent `PiPLSSearchCV` methods are available only when `refit=True` and the estimator
supports the delegated operation. Output-container configuration belongs to the supplied estimator
template, not to the path object. A configured estimator is cloned through the search and its
selected refit retains that configuration.

## Consequences

- `PiPLSSearchCV()` visibly means a complete admissible component path.
- Users can pass the same conventional random-state forms used by scikit-learn estimators.
- The default scorer can be imported and reused as an ordinary callable.
- The fitted estimator has a smaller, less ambiguous attribute surface while retaining both
  standard PLS-style results and direct access to every Pi-PLS matrix and numerical diagnostic.
- `refit=False` path objects expose diagnostics only; prediction and transformation methods are
  absent rather than present and failing later.
- Existing pre-1.0 code using `n_components_values=None` or removed fitted aliases must adopt the
  explicit sentinel or `decomposition_` fields.
