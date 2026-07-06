#!/bin/bash
# A0 — snapshot the Claude Code transcript corpus before 30-day GC erodes it.
# Copies ~/.claude/projects into data/corpus/ and writes a dated manifest.
# Re-run weekly during Phase 0; rsync makes repeat runs incremental.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$HOME/.claude/projects/"
DEST="$REPO_ROOT/data/corpus/"
MANIFEST_DIR="$REPO_ROOT/data/manifests"
STAMP="$(date +%Y-%m-%d_%H%M%S)"

mkdir -p "$DEST" "$MANIFEST_DIR"

rsync -a --exclude '.DS_Store' "$SRC" "$DEST"

MANIFEST="$MANIFEST_DIR/corpus-$STAMP.txt"
{
  echo "snapshot: $STAMP"
  echo "source: $SRC"
  echo "jsonl files: $(find "$DEST" -name '*.jsonl' | wc -l | tr -d ' ')"
  echo "total size: $(du -sh "$DEST" | cut -f1)"
  echo "oldest mtime: $(find "$DEST" -name '*.jsonl' -exec stat -f '%Sm %N' -t '%Y-%m-%d' {} + | sort | head -1)"
  echo "newest mtime: $(find "$DEST" -name '*.jsonl' -exec stat -f '%Sm %N' -t '%Y-%m-%d' {} + | sort | tail -1)"
  echo "--- per-project file counts ---"
  find "$DEST" -name '*.jsonl' | sed "s|$DEST||" | cut -d/ -f1 | sort | uniq -c | sort -rn
} > "$MANIFEST"

echo "snapshot complete → $DEST"
echo "manifest → $MANIFEST"
head -6 "$MANIFEST"
