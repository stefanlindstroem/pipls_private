# Decision 0117: commercial license, authorship, and citation

## Status

Accepted and implemented.

## Context

The repository already declared `BSD-3-Clause` in package metadata, but the root license named only
"Pi-PLS authors" and contained an abbreviated disclaimer rather than the complete standard text.
The public documentation did not identify the three code and documentation authors or provide a
stable citation for the software and its companion manuscript.

The owner requires commercial and noncommercial use, redistribution, and modification to remain
permitted while the copyright notice, conditions, and disclaimer must be retained. The package also
contains separately licensed reference datasets, so the repository license must not be described as
relicensing those assets.

The companion manuscript is under revision at *Computers & Chemical Engineering* as manuscript
CACE-D-26-00847. It has the same three authors as the software.

## Decision

Retain the BSD 3-Clause License. Replace the abbreviated root text with the complete standard
BSD-3-Clause terms and name Vishal Agrawal, Fritjof Nilsson, and Stefan B. Lindström as the 2026
copyright holders of the repository-authored code and documentation.

Use the same three names in package metadata and public documentation. Add `docs/citation.md` as the
human-readable ownership, license-scope, software-citation, and companion-paper page. Add a root
`CITATION.cff` using Citation File Format 1.2.0 for machine-readable software metadata and the
companion-paper preferred citation.

The companion-paper reference is:

> Agrawal, V., Nilsson, F., and Lindström, S. B. (2026). Panoramic Partial Least Squares
> (Pi-PLS): Transparent, parsimonious, and more interpretable multivariate regression model.
> Manuscript under revision at *Computers & Chemical Engineering*, manuscript
> CACE-D-26-00847.

The CFF publication status is `submitted`, with an explicit note that the manuscript is under
revision. The citation must be updated when final publication metadata and a DOI become available.

Dataset-specific license and attribution files remain authoritative for the reference datasets.
The BSD-3-Clause repository license does not relicense those datasets. This decision does not add
paper-reproduction assets, claim publication or acceptance, or authorize package release work.

## Consequences

- Commercial use, redistribution, and modification remain permitted under BSD-3-Clause.
- Source and binary redistributors must preserve the notice, conditions, and disclaimer as stated
  in the license.
- The code and documentation authors and current copyright holders are explicit.
- Human and machine users have one maintained citation route.
- Dataset-specific licenses remain visible and legally separate.
- Publication metadata will require a focused update after the companion paper is published.
