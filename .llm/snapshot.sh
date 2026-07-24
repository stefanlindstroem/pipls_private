#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    printf 'Snapshot creation requires a Git worktree.\n' >&2
    exit 1
}
project_name="$(basename "$root")"
output="${1:-$(dirname "$root")/${project_name}-snapshot.tar.gz}"
if [[ "$output" != /* ]]; then
    output="$(pwd)/$output"
fi
metadata="$(mktemp)"
staging="$(mktemp -d)"
trap 'rm -f "$metadata"; rm -rf "$staging"' EXIT

status="$(git -C "$root" status --porcelain=v1 --untracked-files=normal --ignore-submodules=none)"
if [[ -n "$status" ]]; then
    printf 'Refusing to create a snapshot from a dirty worktree:\n%s\n' "$status" >&2
    exit 1
fi

committed_example_outputs=()
while IFS= read -r -d '' path; do
    if [[ "$(basename "$path")" != ".gitkeep" ]]; then
        committed_example_outputs+=("$path")
    fi
done < <(git -C "$root" ls-tree -r -z --name-only HEAD -- examples/results)
if (( ${#committed_example_outputs[@]} > 0 )); then
    printf 'Refusing to create a snapshot with committed generated example outputs:\n' >&2
    printf '  %s\n' "${committed_example_outputs[@]}" >&2
    exit 1
fi

commit="$(git -C "$root" rev-parse HEAD)"
branch="$(git -C "$root" symbolic-ref --quiet --short HEAD || printf 'detached')"
version="$(PYTHONPATH="$root/src" python3 -c 'import pipls; print(pipls.__version__)' 2>/dev/null || printf 'unknown')"
cat > "$metadata" <<META
project: $project_name
commit: $commit
branch: $branch
dirty: false
created_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
python: $(python3 --version 2>&1)
package_version: $version
META

# Start from the committed tree rather than copying the worktree. Ignored build products,
# generated figures, caches, and other local files therefore cannot enter the snapshot.
git -C "$root" archive --format=tar HEAD | tar -xf - -C "$staging"
cp "$metadata" "$staging/.llm/SNAPSHOT_INFO"

# Archive the contents of the repository root, not an enclosing project directory.
# This keeps archive paths identical to Git paths and allows extraction directly
# into an existing checkout directory.
(
    cd "$staging"
    find . -mindepth 1 -print0 | LC_ALL=C sort -z | \
        tar --null --no-recursion --mtime='UTC 2020-01-01' \
            --owner=0 --group=0 --numeric-owner --transform='s|^\./||' \
            -czf "$output" -T -
)
printf 'Created %s\n' "$output"
