# Pi-PLS

Development repository for Pi-PLS, a PLS-family method for multivariate regression.

The repository currently contains the fixed-parameter numerical core and a scikit-learn-style
`PiPLSRegression` estimator with explicit integer and rule-derived `"max"` predictor-rank modes.
Automatic predictor-rank selection, path analysis, datasets, and paper reproduction are later
increments.

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
    predictor_rank="max",
    samples_per_predictor_rank=10,
)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
```

Read `.llm/README.md` before preparing an LLM-assisted change. Create a repository snapshot with
`make snapshot`.
