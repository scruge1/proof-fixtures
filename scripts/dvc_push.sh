#!/usr/bin/env bash
# v0.4.1 Stage 9 — corpus DVC push helper.
#
# Cron-friendly: idempotent, exits 0 on no-change, logs to stderr.
# Add to AX52 systemd timer or cron.d for nightly push to Hetzner.
#
# Required env (set in cron / Coolify scheduled task / .env):
#   PROJECT_ROOT             absolute path to proof-fixtures clone
#   AWS_ACCESS_KEY_ID         Hetzner Object Storage access key
#   AWS_SECRET_ACCESS_KEY     Hetzner Object Storage secret key
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/opt/callmeie/proof-fixtures}"
CORPUS_FILE="${CORPUS_FILE:-corpus/corrections.jsonl}"

cd "$PROJECT_ROOT"

if [[ ! -f "$CORPUS_FILE" ]]; then
    echo "[dvc_push] $CORPUS_FILE missing; nothing to push." >&2
    exit 0
fi

# Track new corpus state under DVC (idempotent — DVC only updates pointer if
# content hash changed).
dvc add "$CORPUS_FILE"

# Stage the .dvc pointer in git so the commit log shows what shipped.
POINTER="${CORPUS_FILE}.dvc"
if [[ -f "$POINTER" ]]; then
    git add "$POINTER" "corpus/.gitignore" 2>/dev/null || true
    if ! git diff --cached --quiet -- "$POINTER"; then
        git commit -m "data(corpus): push $(date -u +%Y-%m-%dT%H:%M:%SZ) snapshot" -- "$POINTER" "corpus/.gitignore" \
            || true   # may have nothing staged on hash-stable runs
    fi
fi

# Push the actual bytes to Hetzner Object Storage. -j 4 = parallel uploads.
dvc push -j 4

echo "[dvc_push] OK $(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
