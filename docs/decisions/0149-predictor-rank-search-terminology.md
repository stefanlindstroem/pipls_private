# Decision 0149: predictor-rank search terminology

## Status

Accepted and implemented. Patches 1--3 of the authorized six-patch sequence are complete. The
runtime API, maintained examples, public documentation, retained decisions, tests, and distribution
checks now use `search_method="adaptive"` and `search_method="exhaustive"`. The former values are
rejected without aliases, and the mapped candidate sets and numerical evidence remain unchanged.
Patches 4--6 establish and integrate Decision 0150 computational-performance guidance.

## Context

`PiPLSSearchCV.search_method` controls how predictor-rank candidates are covered for each component
count. It does not control the component-count path, the configured scorer, or the later
predictor-rank and component-count tolerance rules.

The existing values do not describe that responsibility precisely:

- `"auto"` is vague and can suggest automatic choice among unrelated algorithms;
- `"optimal"` overstates the guarantee, because the mode evaluates all admissible candidates but
  does not guarantee an optimal model outside the declared candidate set, scorer, validation
  protocol, or hierarchical tolerance rules.

The implemented algorithms already have clearer established descriptions: one is deterministic
adaptive candidate coverage, and the other is exhaustive candidate coverage. The public values
should state those guarantees directly.

## Decision

### Keep the parameter name and rename only its alternatives

The constructor parameter remains:

```python
search_method=...
```

Its accepted values become:

```python
search_method="adaptive"    # default
search_method="exhaustive"
```

The direct pre-release replacement is:

```text
"auto"     -> "adaptive"
"optimal"  -> "exhaustive"
```

The parameter name remains stable because the requested correction concerns the alternatives, and
`search_method` is already scoped by `PiPLSSearchCV` and documented as predictor-rank candidate
coverage. Renaming the parameter in the same increment would enlarge the migration without changing
the numerical contract.

### Define the two guarantees

`"adaptive"` uses the existing deterministic coarse-to-fine predictor-rank procedure independently
for every component count. It may leave admissible predictor ranks unevaluated. Candidate refinement
continues to use the exact configured-score reference, not the public predictor-rank tolerance.

`"exhaustive"` evaluates every admissible predictor rank for every requested component count. It
guarantees complete candidate coverage of the declared path, not statistical or scientific
optimality.

Neither rename changes candidate generation, scoring, numerical ties, predictor-rank tolerance,
component-count selection, refitting, OOF reporting, or fitted numerical results.

### Keep fitted exhaustiveness diagnostic terminology

The fitted attribute remains:

```python
search.search_is_exhaustive_
```

It reports the candidate coverage actually achieved, not merely the requested method. An adaptive
search can become exhaustive when the admissible interval is small or when refinement evaluates all
remaining ranks. Therefore this is a valid outcome:

```python
search.search_method == "adaptive"
search.search_is_exhaustive_ is True
```

No `search_is_adaptive_` attribute or duplicate mode diagnostic is added.

### Do not retain compatibility aliases

`"auto"` and `"optimal"` are rejected after the runtime migration. The package is unreleased and
its accepted compatibility policy does not retain aliases for replaced pre-release values unless
the owner explicitly requests them.

Migration errors should state the complete current choice set:

```text
search_method must be "adaptive" or "exhaustive".
```

Historical decisions and changelog entries may retain the former values when describing the
migration. Current source, tests, examples, API documentation, and maintainer contracts must use the
new values.

### Treat one-candidate rank policies explicitly

A maximum-rank policy and a one-element explicit sequence provide one predictor-rank candidate per
component count:

```python
PiPLSSearchCV(predictor_rank_values="max")
PiPLSSearchCV(predictor_rank_values=[20])
```

Their examples omit `search_method` because there is no predictor-rank coverage choice to explain.
The default `search_method="adaptive"` remains accepted for scikit-learn constructor consistency;
the implementation cannot distinguish an omitted default from an explicitly supplied default.

The nondefault value `search_method="exhaustive"` is rejected for these one-candidate policies
because it has no operative meaning. This matches the existing rejection of nondefault
predictor-rank tolerances when predictor-rank optimization is inapplicable.

A multi-rank explicit sequence remains an optimized policy. `"adaptive"` may evaluate only an
adaptive subset of the supplied admissible ranks, while `"exhaustive"` evaluates every supplied
admissible rank.

### Preserve the separation from predictor-rank tolerance

Search method answers which candidates are evaluated. Decision 0148 answers which evaluated
candidate is retained at each component count. The order remains:

1. generate candidates by adaptive or exhaustive coverage;
2. identify the exact configured-score reference among evaluated candidates;
3. retain the smallest evaluated rank satisfying the predictor-rank tolerances;
4. construct `component_path_`;
5. apply component-count selection to the conditioned path.

Thus an adaptive search returns the smallest **evaluated** qualifying rank, while an exhaustive
search returns the smallest admissible qualifying rank.

### Add computational-performance guidance after the rename

The computational-performance guide must be written against the final terminology. It will explain
candidate-fit scaling, development versus final CV repetitions, adaptive versus exhaustive coverage,
fixed and maximum rank policies, restricted component paths, randomized predictor SVD, parallelism,
OOF recomputation, and available timing diagnostics.

The guide must distinguish controls that reduce wall time without intended statistical change from
controls that alter candidate coverage, validation evidence, rank policy, or numerical
approximation.

## Authorized six-patch sequence

1. Establish this terminology and migration contract in Decision 0149 and synchronize the active
   maintainer layer -- complete.
2. Rename the runtime values, validate one-candidate policies, and prove old/new numerical
   equivalence under the direct value mapping -- complete.
3. Migrate remaining source prose, examples, public documentation, retained decisions, distribution checks,
   and tests; reject stale current uses and complete Decision 0149 -- complete.
4. Add Decision 0150 and the central computational-performance guide structure.
5. Write the complete training-performance guidance using `"adaptive"` and `"exhaustive"`.
6. Integrate links, troubleshooting, examples, changelog, documentation tests, and distribution
   audits; complete Decision 0150 and return `.llm` to current-state wording.

Do not introduce the performance guide before the search terminology migration is complete.

## Validation obligations

The runtime rename must verify that:

- the default is `"adaptive"`;
- old `"auto"` candidate sets and numerical results are identical to new `"adaptive"` results;
- old `"optimal"` candidate sets and numerical results are identical to new `"exhaustive"`
  results;
- old values are rejected without aliases;
- cloning, parameter inspection, parameter setting, repr, pipelines, and pickling use the new
  values;
- adaptive determinism and exhaustive coverage remain unchanged;
- `search_is_exhaustive_` still reports achieved coverage;
- maximum and one-rank policies accept the default and reject the nondefault exhaustive method;
- multi-rank explicit sequences retain both coverage choices;
- predictor-rank tolerance and conditioned-path behavior remain unchanged.

The mapped before-and-after manifest must compare evaluated pairs, split scores, configured-score
summaries, CV-MSE summaries, `rank_test_score`, component-path rows, predictor-rank evidence, timing
array shapes, and `search_is_exhaustive_`.

The documentation phase must additionally verify that current examples use only `"adaptive"` and
`"exhaustive"`, fixed and maximum-rank examples omit `search_method`, and unrelated values such as
`svd_solver="auto"` remain intact.

## Consequences

The public names state the actual coverage guarantees and no longer imply a broader optimization
claim. The migration is intentionally breaking and alias-free. `search_method` remains stable,
`search_is_exhaustive_` remains meaningful, and the later computational-performance guide can
compare the two modes without explaining misleading legacy terminology.
