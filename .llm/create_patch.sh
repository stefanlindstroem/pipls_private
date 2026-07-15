#!/usr/bin/env bash
set -euo pipefail

output="${1:-pipls-change.patch}"
git rev-parse --show-toplevel >/dev/null 2>&1 || { printf 'Not inside a Git repository.\n' >&2; exit 1; }
git diff --binary --no-ext-diff --src-prefix=a/ --dst-prefix=b/ > "$output"
[[ -s "$output" ]] || { rm -f "$output"; printf 'No unstaged changes to write.\n' >&2; exit 1; }
git apply --check --reverse "$output"
printf 'Created %s\n' "$output"
