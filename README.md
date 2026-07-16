# Pi-PLS

Development repository for Pi-PLS, a PLS-family method for multivariate regression.

The repository contains the fixed-parameter numerical core and a scikit-learn-style
`PiPLSRegression` estimator with adaptive `"auto"`, exhaustive `"optimal"`, rule-derived
`"max"`, and explicit integer predictor-rank modes. Predictor linear algebra independently supports
full, randomized, and conservative automatic SVD policies. `PiPLSPathCV` provides pipeline-aware
joint path analysis over `n_components` and `predictor_rank`. Datasets and paper reproduction remain
later increments.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make check
```

For a later terminal session, reactivate the existing environment before running commands:

```bash
cd /path/to/pipls
source .venv/bin/activate
git status
make check
```

## Current estimator

```python
from pipls import PiPLSRegression

model = PiPLSRegression(
    n_components=2,
    predictor_rank="auto",
    samples_per_predictor_rank=10,
    cv=5,
    svd_solver="auto",
    random_state=0,
)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
X_scores, Y_scores = model.transform(X_train, Y_train)
print(model.predictor_rank_)
print(model.decomposition_.D)
```

For the complete two-parameter surface or custom learned preprocessing:

```python
from pipls import PiPLSPathCV

search = PiPLSPathCV(cv=5)
search.fit(X_train, Y_train)
print(search.best_params_)
print(search.best_pipls_.coef_)
```

The estimator follows scikit-learn and `PLSRegression` conventions for coefficient orientation,
latent-score transforms, feature names, pandas output containers, and fitted weights/loadings.
Pi-PLS-specific factorization output is grouped in the public read-only `decomposition_` result.

`samples_per_predictor_rank` defaults to 10. Rule-based values below 5 are allowed but emit
`StatisticalSupportWarning` because the resulting rank bound may lack sufficient statistical
support.

Theory navigation starts at `docs/theory.md`; the persistent LLM-facing derivation is in
`.llm/theory.md`, and the concise normative equations are in `.llm/mathematics.md`.

Read `.llm/README.md` before preparing an LLM-assisted change. Create a repository snapshot with
`make snapshot`.
