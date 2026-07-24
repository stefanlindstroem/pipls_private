# Decision 0099: example artifact and source-distribution boundary

## Context

Numbered examples write final PDF files below `examples/results/`. Repository policy already treats
those PDFs as generated outputs and keeps only `.gitkeep` directory placeholders under version
control. The committed tree nevertheless contained generated PDFs, so clean Git snapshots included
artifacts that the contributor and example documentation described as ignored.

The source distribution had the opposite defect. `MANIFEST.in` included the example scripts but not
the `.gitkeep` placeholders. After extraction, example 01 failed when it tried to write its first
PDF because `examples/results/` did not exist.

## Decision

1. Commit only `.gitkeep` files below `examples/results/`; generated example outputs remain ignored
   and must not be included in repository snapshots.
2. Refuse snapshot creation when committed `HEAD` contains any regular example-result file other
   than `.gitkeep`, even when the worktree itself is clean.
3. Include every example-result `.gitkeep` file in the source distribution so the documented output
   directories exist after extraction.
4. Extend installed-distribution validation for the source distribution: install its `examples`
   extra, run example 01 from the extracted source tree with a headless Matplotlib backend, and
   require its PDF output.
5. Test the repository result tree, source-distribution manifest, snapshot refusal, and extracted
   example execution without committing generated PDFs.

## Consequences

A clean repository snapshot now contains only example output-directory placeholders. An unpacked
source distribution preserves the same directory structure and can run the introductory example
without manual directory creation. Numbered examples continue to trust repository-owned output
directories; their numerical calculations and output filenames are unchanged.
