"""Generate deterministic Pi-PLS train/test data and fit the estimator."""

from pipls import PiPLSRegression
from pipls.datasets import make_pipls_train_test

train, test = make_pipls_train_test(
    n_train=120,
    n_test=40,
    n_features=20,
    n_targets=5,
    n_shared=2,
    n_predictor_specific=2,
    n_response_specific=1,
    shared_strength=(2.0, 1.0),
    noise=(0.15, 0.2),
    random_state=0,
)

print(PiPLSRegression(n_components=2, predictor_rank=4).fit(train.X, train.Y).score(test.X, test.Y))
print(train.truth.n_shared)
