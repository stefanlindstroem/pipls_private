"""Evaluate a small-sample Pi-PLS path with leave-one-out validation."""

from sklearn.model_selection import LeaveOneOut

from pipls import PiPLSSearchCV
from pipls.datasets import make_pipls_regression

# Imagine a calibration study with twelve costly specimens. Leave-one-out
# validation gives every specimen one held-out prediction while retaining the
# other eleven observations for fitting.
data = make_pipls_regression(
    n_samples=12,
    n_features=5,
    n_targets=2,
    n_shared=1,
    n_predictor_specific=1,
    shared_strength=2.0,
    predictor_specific_strength=1.0,
    noise=(0.15, 0.2),
    random_state=7,
)

search = PiPLSSearchCV(
    n_components_values=[1, 2],
    predictor_rank_values=[1, 2],
    search_method="exhaustive",
    cv=LeaveOneOut(),
    n_jobs=1,
).fit(data.X, data.Y)

selection = search.select(rule="best_score")
report = search.oof_report(data.X, data.Y, selection=selection)
if report.pooled_oof_r2 is None:
    raise RuntimeError("Leave-one-out validation did not produce pooled OOF R2.")

print(f"Observations: {data.n_samples}")
print(f"Leave-one-out splits: {selection.n_splits}")
print(
    "Selected pair: "
    f"n_components={selection.n_components}, predictor_rank={selection.predictor_rank}"
)
print(f"Complete OOF coverage: {report.has_complete_oof_coverage}")
print(f"OOF prediction shape: {report.oof_predictions.shape}")
print(
    "Mean response-standardized CV-MSE: "
    f"{selection.cv_mse_mean:.4f}"
)
print(
    "Pooled OOF R2 (not mean foldwise R2): "
    f"{report.pooled_oof_r2:.4f}"
)
print("OOF interpretation: selection-conditioned")
