#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    printf 'Usage: .llm/commit.sh COMMIT_MESSAGE\n' >&2
    exit 2
fi

message="$*"
root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    printf 'Not inside a Git repository.\n' >&2
    exit 1
}
cd "$root"

if git diff --quiet && git diff --cached --quiet && [[ -z "$(git ls-files --others --exclude-standard)" ]]; then
    printf 'Nothing to commit.\n' >&2
    exit 1
fi

printf 'Running repository validation before commit...\n'
make check

git diff --check
git add -A

if git diff --cached --quiet; then
    printf 'Nothing to commit after validation.\n' >&2
    exit 1
fi

printf 'Changes to be committed:\n'
git diff --cached --stat
git commit -m "$message"
