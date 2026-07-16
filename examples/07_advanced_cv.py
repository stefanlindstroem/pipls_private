"""Advanced splitter and ordered out-of-fold examples for Pi-PLS."""

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneOut, TimeSeriesSplit

from pipls import PiPLSPathCV

rng = np.random.default_rng(7)
X = rng.normal(size=(30, 8))
Y = X @ rng.normal(size=(8, 2)) + 0.05 * rng.normal(size=(30, 2))
groups = np.repeat(np.arange(10), 3)

# Group-aware path selection.
grouped = PiPLSPathCV(cv=GroupKFold(n_splits=5))
grouped.fit(X, Y, groups=groups)

# Paper-style selection-conditioned LOO reporting.
loo = PiPLSPathCV(cv=LeaveOneOut(), return_oof_predictions=True)
loo.fit(X, Y)
print(loo.validation_report_)

# Temporal validation leaves early rows without OOF predictions.
temporal = PiPLSPathCV(
    cv=TimeSeriesSplit(n_splits=5),
    return_oof_predictions=True,
)
temporal.fit(X, Y)
print(temporal.oof_prediction_counts_)
