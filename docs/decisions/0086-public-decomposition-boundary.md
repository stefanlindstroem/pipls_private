# Decision 0086: public decomposition boundary

## Status

Accepted and implemented.

## Context

`PiPLSDecomposition` mirrored the complete private `PiPLSCoreResult`. It exposed the truncated
predictor basis $\Pi$, response basis $C$, least-squares map $W$, predictor rotations $P$, diagonal
matrix $D$, response rotations $Q$, a duplicate dilation vector, and descriptive aliases for most of
those arrays.

The maintained user workflows consume only the final Pi-PLS factorization and numerical solver
information. The intermediate matrices $\Pi$, $C$, and $W$ are required by the algorithm and some
internal numerical checks, but are not fitted-model interpretation results. Exposing both $D$ and
its diagonal vector also duplicated the same information.

## Decision

The public `PiPLSDecomposition` exposes only:

- `predictor_rotations`, corresponding to $P$;
- `dilation`, corresponding to the diagonal of $D$;
- `response_rotations`, corresponding to $Q$;
- `predictor_numerical_rank` and whether that rank is exact;
- `rank_tolerance`;
- `predictor_svd_solver`;
- the derived `standardized_regression_map` $PDQ^{\mathsf T}$.

The public object uses descriptive Python names rather than parallel symbolic and descriptive
aliases. The matrices $\Pi$, $C$, $W$, and the redundant matrix form $D$ remain private in
`PiPLSCoreResult`.

The fixed-structure benchmark reconstructs the retained predictor basis from the fitted training
coordinates because that quantity is specific to the benchmark question and does not justify a
package-wide public field.

## Consequences

- Programming users see only quantities used for interpretation, prediction auditing, or solver
  provenance.
- The private numerical core retains the complete construction required by the method.
- `pipls_display_factors()` consumes descriptive decomposition fields and the dilation vector.
- Existing pre-release code using public `Pi`, `C`, `W`, `P`, `D`, `Q`, `regression_map`, `x_rank`,
  or their descriptive aliases must migrate to the reduced field set.
- Standard PLS-family fitted attributes remain available on `PiPLSRegression`.
