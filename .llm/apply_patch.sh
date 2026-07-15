#!/usr/bin/env bash
set -euo pipefail

allow_dirty=false
if [[ "${1:-}" == "--allow-dirty" ]]; then allow_dirty=true; shift; fi
patch="${1:?Usage: .llm/apply_patch.sh [--allow-dirty] PATCH_FILE}"
[[ -r "$patch" ]] || { printf 'Patch is not readable: %s\n' "$patch" >&2; exit 1; }
git rev-parse --show-toplevel >/dev/null 2>&1 || { printf 'Not inside a Git repository.\n' >&2; exit 1; }
if [[ "$allow_dirty" == false ]] && { ! git diff --quiet || ! git diff --cached --quiet; }; then
    printf 'Refusing to apply a patch to a dirty tracked worktree.\n' >&2
    exit 1
fi
if grep -E '^(---|\+\+\+) (/[[:graph:]]+|[ab]/\.\./|\.\./)' "$patch" >/dev/null; then
    printf 'Patch contains an unsafe path.\n' >&2
    exit 1
fi
git apply --check "$patch"
git apply "$patch"
printf 'Applied %s\nRun: make check, inspect git diff, then commit with .llm/commit.sh \"MESSAGE\"\n' "$patch"
