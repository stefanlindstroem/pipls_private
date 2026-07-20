# Compatibility

Pi-PLS supports Python 3.10 through 3.14. Package metadata deliberately declares
`requires-python = ">=3.10"` without an upper bound: newer Python versions may be installable, but
they are not part of the supported range until they are added to the documented and tested matrix.
The Python classifiers enumerate the currently supported versions.

## Runtime dependency ranges

The runtime dependencies use lower bounds that define the oldest supported API lines and
conservative upper major-version limits:

| Dependency | Supported range |
|---|---|
| NumPy | `>=1.26,<3` |
| scikit-learn | `>=1.4,<2` |
| joblib | `>=1.2,<2` |

These are compatibility ranges rather than reproducible-environment pins. A normal installation
selects mutually compatible releases within them. Newer compatible minor releases are supported
through continuous integration rather than by exact package pins. A new dependency major version
is admitted only after its compatibility has been reviewed and tested.

## Minimum-dependency environment

The repository records the minimum tested dependency lines in `constraints/minimum.txt`:

```text
numpy==1.26.*
scikit-learn==1.4.*
joblib==1.2.*
```

This minimum combination is tested on Python 3.10. The supported-Python matrix uses normal
dependency resolution on Python 3.10 through 3.14. This is necessary on Python 3.13 and 3.14
because NumPy 1.26 and scikit-learn 1.4 support Python only through 3.12, and it keeps one consistent
resolution policy across the interpreter matrix.

The constraint file is a maintainer test input, not a user lock file. Users should normally install
Pi-PLS without it and use their own application-level lock or environment-management policy when
exact reproducibility is required.

## Continuous-integration environments

The test workflow separates three compatibility responsibilities:

| Job | Environment | Purpose |
|---|---|---|
| `minimum-dependencies` | Python 3.10 and `constraints/minimum.txt` | Verify all declared lower runtime dependency lines together |
| `supported-python` | Python 3.10 through 3.14 with normal resolution | Verify every supported interpreter |
| `latest-dependencies` | Python 3.14 with explicit runtime dependency upgrades | Verify the newest releases admitted by the declared major-version limits |

Each job prints the resolved Python, NumPy, scikit-learn, and joblib versions before running the
standard repository checks. The supported-Python matrix continues after an individual interpreter
failure so the workflow reports the state of the complete supported range.

## What support means

The three environments avoid an unnecessary Cartesian product while checking the declared lower
bounds, the active interpreter range, and the newest admitted runtime dependencies. They do not
claim that every historical combination inside the declared ranges is continuously exercised.

A combination inside the declared ranges is expected to work. When a reproducible incompatibility
is found, the project should either correct it or narrow the metadata and documentation in the same
change. Optional documentation, example, data, and plotting dependencies have separate ranges in
`pyproject.toml`; they do not expand the core runtime support claim.
