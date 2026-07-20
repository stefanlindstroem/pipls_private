"""Generate an independent synthetic train/test problem and evaluate Pi-PLS."""

from pipls import PiPLSRegression
from pipls.datasets import make_pipls_train_test

# Shared directions affect both X and Y and therefore support prediction.
# Predictor- and response-specific directions add structured variation that is
# present in only one block.
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

model = PiPLSRegression(n_components=2, predictor_rank=4).fit(train.X, train.Y)
predictions = model.predict(test.X)

print("Synthetic Pi-PLS train/test example")
print(f"Training data: X{train.X.shape}, Y{train.Y.shape}")
print(f"Independent test data: X{test.X.shape}, Y{test.Y.shape}")
print("Known latent structure:")
print(f"  Shared directions affecting X and Y: {train.truth.n_shared}")
print(
    "  Predictor-specific directions affecting only X: "
    f"{train.truth.n_predictor_specific}"
)
print(
    "  Response-specific directions affecting only Y: "
    f"{train.truth.n_response_specific}"
)
print("Fitted Pi-PLS model:")
print(f"  Components: {model.n_components}")
print(f"  Predictor rank: {model.predictor_rank_}")
print("Held-out evaluation:")
print(f"  Predictions: {predictions.shape}")
print(
    "  Test R^2 (coefficient of determination): "
    f"{model.score(test.X, test.Y):.3f}"
)
