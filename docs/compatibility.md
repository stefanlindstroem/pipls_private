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

## What support means

The supported Python matrix is exercised with normally resolved dependencies. The minimum stack is
exercised separately on Python 3.10. This avoids an unnecessary Cartesian product while checking
both the declared lower bounds and the active interpreter range.

A combination inside the declared ranges is expected to work. When a reproducible incompatibility
is found, the project should either correct it or narrow the metadata and documentation in the same
change. Optional documentation, example, data, and plotting dependencies have separate ranges in
`pyproject.toml`; they do not expand the core runtime support claim.
