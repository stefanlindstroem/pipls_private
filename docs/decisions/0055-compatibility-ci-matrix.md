# Decision 0055: compatibility CI matrix

## Status

Accepted and implemented.

## Context

Decision 0054 defined the supported Python and runtime dependency ranges. The initial policy patch
added Python 3.14 to the ordinary test matrix and made the existing minimum-dependency job consume
the complete constraint file, but the workflow still mixed responsibilities under a generic test
job. It did not explicitly test an upgraded latest-compatible environment, and failures did not
report the resolved interpreter and runtime dependency versions.

Testing every Python and dependency-version combination would add substantial CI cost while
including combinations that cannot be installed because the oldest dependency lines do not publish
artifacts for every newer interpreter.

## Decision

The test workflow has three explicit compatibility responsibilities:

- `minimum-dependencies` runs on Python 3.10 and installs the complete stack from
  `constraints/minimum.txt`;
- `supported-python` runs normal dependency resolution on Python 3.10 through 3.14;
- `latest-dependencies` runs on Python 3.14 and explicitly upgrades NumPy, scikit-learn, and joblib
  within the declared upper major-version limits.

Every job prints the resolved Python, NumPy, scikit-learn, and joblib versions before running the
standard repository checks. The supported-Python matrix disables fail-fast behavior so failures on
one interpreter do not hide the status of the others.

The workflow does not test a Cartesian product of interpreter and dependency versions. Clean wheel
and source-distribution installation checks remain a separate packaging increment.

## Consequences

The lower bounds, supported interpreter range, and newest admitted runtime dependency releases now
have separate, diagnosable CI ownership. A failure identifies both the compatibility responsibility
and the concrete versions involved. The matrix remains small enough for ordinary push and pull-
request validation.

The latest-compatible job is not a lock file or a promise that every historical combination inside
the declared ranges is continuously exercised. Reproducible incompatibilities still require either
a code correction or a coordinated narrowing of metadata and documentation.
