#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
project_name="$(basename "$root")"
output="${1:-$(dirname "$root")/${project_name}-snapshot.tar.gz}"
metadata="$(mktemp)"
staging="$(mktemp -d)"
trap 'rm -f "$metadata"; rm -rf "$staging"' EXIT

commit="uncommitted"
branch="no-git"
dirty="unknown"
if git -C "$root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    commit="$(git -C "$root" rev-parse HEAD 2>/dev/null || printf 'uncommitted')"
    branch="$(git -C "$root" branch --show-current 2>/dev/null || printf 'detached')"
    if git -C "$root" diff --quiet && git -C "$root" diff --cached --quiet; then dirty=false; else dirty=true; fi
fi
version="$(PYTHONPATH="$root/src" python3 -c 'import pipls; print(pipls.__version__)' 2>/dev/null || printf 'unknown')"
cat > "$metadata" <<META
project: $project_name
commit: $commit
branch: $branch
dirty: $dirty
created_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
python: $(python3 --version 2>&1)
package_version: $version
META

mkdir -p "$staging/$project_name/.llm"
rsync -a \
  --exclude='.git/' --exclude='.venv/' --exclude='venv/' \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.pytest_cache/' \
  --exclude='.mypy_cache/' --exclude='.ruff_cache/' --exclude='.coverage' \
  --exclude='coverage.xml' --exclude='htmlcov/' --exclude='build/' --exclude='dist/' \
  --exclude='*.egg-info/' --exclude='docs/_build/' --exclude='.ipynb_checkpoints/' \
  --exclude='.DS_Store' --exclude='._*' --exclude='*~' --exclude='*.patch' \
  --exclude='*-snapshot.tar.gz' "$root/" "$staging/$project_name/"
cp "$metadata" "$staging/$project_name/.llm/SNAPSHOT_INFO"

tar --sort=name --mtime='UTC 2020-01-01' --owner=0 --group=0 --numeric-owner \
    -czf "$output" -C "$staging" "$project_name"
printf 'Created %s\n' "$output"
