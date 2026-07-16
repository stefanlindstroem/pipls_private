# Pi-PLS documentation

The current authoritative development specification is the tracked `.llm/` layer together with
`docs/publication_repository_plan.md` and the accepted decision records.

Current navigation:

- `theory.md`: user-facing theory overview and navigation to the persistent LLM theory reference;
- `parameter_selection.md`: implemented predictor-rank selection and accepted search-policy
  roadmap;
- `estimator_api.md`: current `PiPLSRegression` interface;
- `path_analysis.md`: pipeline-aware joint path search and diagnostics;
- `preprocessing.md`: preprocessing semantics;
- `decisions/0010-path-analysis-api.md`: accepted `PiPLSPathCV` boundary;
- `decisions/0007-predictor-rank-search-policies.md`: accepted `"optimal"` versus adaptive
  `"auto"` semantics and the separate randomized-SVD policy;
- `publication_repository_plan.md`: detailed revision-5 architecture, subject to later accepted
  decision records where explicitly noted.
