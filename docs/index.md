# Pi-PLS documentation

The current authoritative development specification is the tracked `.llm/` layer together with
`docs/publication_repository_plan.md` and the accepted decision records.

Current navigation:

- `theory.md`: user-facing theory overview and navigation to the persistent LLM theory reference;
- `parameter_selection.md`: implemented predictor-rank selection and accepted search-policy
  roadmap;
- `estimator_api.md`: current `PiPLSRegression` interface;
- `path_analysis.md`: pipeline-aware joint path search and diagnostics;
- `cross_validation.md`: grouped, repeated, predefined, temporal, LOO, and OOF contracts;
- `preprocessing.md`: preprocessing semantics;
- `decisions/0005-leave-one-out-protocol.md`: implemented LOO and OOF reporting contract;
- `decisions/0010-path-analysis-api.md`: accepted `PiPLSPathCV` boundary;
- `decisions/0011-shared-selection-engine.md`: shared private search/evaluation architecture;
- `decisions/0012-sklearn-api-alignment.md`: PLS-style estimator and path API contracts;
- `decisions/0007-predictor-rank-search-policies.md`: accepted `"optimal"` versus adaptive
  `"auto"` semantics and the separate randomized-SVD policy;
- `publication_repository_plan.md`: detailed revision-5 architecture, subject to later accepted
  decision records where explicitly noted.

- [Decision 0013: final scikit-learn cleanup boundary](decisions/0013-sklearn-cleanup-boundary.md)
