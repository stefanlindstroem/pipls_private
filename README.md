# Pi-PLS

Development repository for Pi-PLS, a PLS-family method for multivariate regression.

The repository contains the fixed-parameter numerical core and a scikit-learn-style
`PiPLSRegression` estimator with adaptive `"auto"`, exhaustive `"optimal"`, rule-derived
`"max"`, and explicit integer predictor-rank modes. Predictor linear algebra independently supports
full, randomized, and conservative automatic SVD policies. `PiPLSPathCV` provides pipeline-aware
joint path analysis over `n_components` and `predictor_rank`. Both interfaces support ordinary
scikit-learn grouped, repeated, predefined, temporal, and leave-one-out splitters, with optional
ordered out-of-fold reporting. A validated dataset container and deterministic synthetic latent-structure generator are available under `pipls.datasets`. The repository also includes transparent real-data examples for the Linnerud, pulp, and sugarcane multi-output regression tables; further manuscript dataset migration and paper reproduction remain later increments.

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

For explicit validation reporting:

```python
from sklearn.model_selection import LeaveOneOut

search = PiPLSPathCV(
    cv=LeaveOneOut(),
    return_oof_predictions=True,
).fit(X_train, Y_train)
print(search.validation_report_)
print(search.oof_predictions_)
```

For transparent real-data workflows, see `examples/09_linnerud_real_data.py`,
`examples/10_pulp_real_data.py`, and `examples/11_sugarcane_real_data.py`. Each reads predictor
and response tables directly with pandas, verifies their alignment, and then fits:

```python
X = pd.read_csv("datasets/linnerud/X.csv")
Y = pd.read_csv("datasets/linnerud/Y.csv")
model = PiPLSRegression(n_components=2).fit(X, Y)
```

For deterministic synthetic train/test data:

```python
from pipls.datasets import make_pipls_train_test

train, test = make_pipls_train_test(
    n_train=120,
    n_test=40,
    n_features=20,
    n_targets=5,
    n_shared=2,
    n_predictor_specific=2,
    n_response_specific=1,
    random_state=0,
)
model.fit(train.X, train.Y)
print(model.score(test.X, test.Y))
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


The public estimator mirrors scikit-learn PLS conventions, including latent transforms, least-squares inverse reconstruction, standard search diagnostics, and pipeline-aware path selection.
