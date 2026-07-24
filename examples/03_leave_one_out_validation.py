"""Evaluate a small-sample Pi-PLS path with leave-one-out validation."""

from sklearn.model_selection import LeaveOneOut

from pipls import PiPLSPathCV
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

search = PiPLSPathCV(
    n_components_values=[1, 2],
    predictor_rank_values=[1, 2],
    search_method="optimal",
    cv=LeaveOneOut(),
    return_oof_predictions=True,
    n_jobs=1,
).fit(data.X, data.Y)

report = search.validation_report_
if report.oof_predictions is None or report.pooled_oof_r2 is None:
    raise RuntimeError("Leave-one-out validation did not produce complete OOF diagnostics.")

print(f"Observations: {data.n_samples}")
print(f"Leave-one-out splits: {report.n_splits}")
print(
    "Selected pair: "
    f"n_components={report.n_components}, predictor_rank={report.predictor_rank}"
)
print(f"Complete OOF coverage: {report.complete_oof_coverage}")
print(f"OOF prediction shape: {report.oof_predictions.shape}")
print(
    "Mean response-standardized CV-MSE: "
    f"{report.mean_response_standardized_mse:.4f}"
)
print(
    "Pooled OOF R2 (not mean foldwise R2): "
    f"{report.pooled_oof_r2:.4f}"
)
print(f"Estimate kind: {report.estimate_kind}")
