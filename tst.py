import matplotlib.pyplot as plt
from sklearn.datasets import load_linnerud
from sklearn.metrics import root_mean_squared_error

from pipls import PiPLSSearchCV

data = load_linnerud()
X, Y = data.data, data.target

model = (
    PiPLSSearchCV()
    .fit(X, Y)
    .refit(X, Y, rule="one_standard_error")
)

Y_fitted = model.predict(X)
rmse = root_mean_squared_error(Y, Y_fitted, multioutput="raw_values")

fig, axes = plt.subplots(1, Y.shape[1], figsize=(9, 3))

for response, name in enumerate(data.target_names):
    observed = Y[:, response]
    predicted = Y_fitted[:, response]

    axes[response].scatter(observed, predicted)

    limits = [
        min(observed.min(), predicted.min()),
        max(observed.max(), predicted.max()),
    ]
    axes[response].plot(limits, limits, linestyle="--")
    axes[response].set_title(f"{name}\nRMSE = {rmse[response]:.2f}")
    axes[response].set_xlabel("Observed")
    axes[response].set_ylabel("Fitted")

fig.tight_layout()
plt.show()
