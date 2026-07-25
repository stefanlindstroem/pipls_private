# Decision 0106: fold-based CV standard error

## Status

Accepted.

## Context

The concise Pi-PLS component-path records store the mean response-standardized CV-MSE, the
population standard deviation of the realized fold-specific MSE values, and the number of
validation splits. Maintained component-path figures have used the fold standard deviation as a
descriptive error bar.

The owner wants the figures to support visual use of the conventional one-standard-error rule,
usually abbreviated the 1-SE rule. That rule requires an estimate of the standard error of the
mean CV error rather than the raw dispersion of the fold errors. The public records already contain
all information needed to derive that quantity, so another stored field or constructor argument
would duplicate state.

## Decision

`PiPLSComponentResult`, `PiPLSPredictorRankProfile`, and `PiPLSComponentPath` expose a derived
read-only property named `cv_mse_standard_error`. The example-local ordinary-PLS
`PLSComponentPath` exposes the same property so the maintained comparison can use a symmetric
contract.

For $K$ realized validation splits, let

\[
\widehat{\sigma}_{\mathrm{pop}}
=
\left[
\frac{1}{K}
\sum_{k=1}^{K}
(e_k-\bar e)^2
\right]^{1/2}
\]

be the stored population fold standard deviation. The derived standard error is

\[
\widehat{\mathrm{SE}}(\bar e)
=
\frac{\widehat{\sigma}_{\mathrm{pop}}}{\sqrt{K-1}}.
\]

This is algebraically identical to converting the stored population standard deviation to the
sample standard deviation and dividing by $\sqrt{K}$. The derived property requires at least two
validation splits. Paths produced by a valid one-split protocol remain constructible, but accessing
their standard error raises an explicit `ValueError`.

When defined, the derived arrays are finite `float64`, read-only, and aligned with their owning
path or profile. They are calculated on access and are not added to constructor signatures, pickle
payloads, or `cv_results_`.

This quantity is a conventional fold-based resampling heuristic. Overlapping CV training sets mean
that the fold estimates are not independent, so the property is not presented as a confidence
interval or a formal uncertainty guarantee.

This increment does not change candidate scores, ranking, selected parameters, refitting, or any
maintained figure. A following increment will migrate the CV-MSE error bars and explanatory
documentation. Automatic 1-SE component selection remains outside the present demonstration.

## Consequences

Programming users can obtain one consistently defined uncertainty quantity from every concise
Pi-PLS path result without reconstructing it from stored fields. Existing fold-SD access and
serialization remain unchanged. The next plotting patch can replace descriptive fold-SD bars with
one fold-based standard error while leaving component-count choice explicit.
