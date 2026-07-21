"""Create a Pulp Pi-PLS component path and fixed-model post-analysis."""

from pathlib import Path

from _support.plot_component_path import plot_pipls_component_path
from _support.post_analysis_artifacts import (
    build_post_analysis_tables,
    render_post_analysis_report,
    write_post_analysis_tables,
)
from _support.pulp_workflow import PULP_CHOSEN_N_COMPONENTS, run_pulp_workflow

ANALYSIS_DIR = Path(__file__).resolve().parent / "results" / "pulp_post_analysis"
CHOSEN_N_COMPONENTS = PULP_CHOSEN_N_COMPONENTS

workflow = run_pulp_workflow(chosen_n_components=CHOSEN_N_COMPONENTS)
X = workflow.X
Y = workflow.Y
model = workflow.model

# Export the component path and its derived figure.
workflow.component_path.to_csv(ANALYSIS_DIR / "component_path.csv", index=False)
plot_pipls_component_path(
    ANALYSIS_DIR / "component_path.csv",
    ANALYSIS_DIR / "component_path.pdf",
    title="Pulp Pi-PLS component path",
)

# Export OOF predictions and fitted-model inspection tables.
write_post_analysis_tables(
    ANALYSIS_DIR,
    build_post_analysis_tables(
        factors=workflow.factors,
        diagnostics=workflow.diagnostics,
        structure=workflow.structure,
        predictor_names=X.columns.tolist(),
        response_names=Y.columns.tolist(),
        sample_names=[str(index) for index in range(1, len(X) + 1)],
        fold_index=workflow.oof.fold_index,
    ),
)
render_post_analysis_report(
    ANALYSIS_DIR,
    ANALYSIS_DIR / "post_analysis.pdf",
    dataset_name="Pulp",
    score_components=(1, 2),
    biplot_components=(1, 2),
    loading_components=(1, 2, 3),
    coefficient_responses=("CSF", "Density", "TI"),
)

print(f"X shape: {X.shape}; Y shape: {Y.shape}")
print(
    "Selected Pi-PLS: "
    f"n_components={model.n_components}, predictor_rank={model.predictor_rank_}"
)
print(f"Wrote results to {ANALYSIS_DIR}")
