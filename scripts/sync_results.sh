#!/usr/bin/env bash
# Periodic push of GPU-week results from the T4 box. Called by run_week.py every
# 30 min when GITHUB_TOKEN is set. Never commits credentials; token is used only
# for the push URL and never written to disk.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "sync_results: GITHUB_TOKEN unset — skipping push (use scripts/pack_results.sh for manual upload)"
  exit 0
fi

REMOTE_URL="$(git remote get-url origin | sed -E 's#https://[^@]*@#https://#')"
AUTH_URL="$(echo "$REMOTE_URL" | sed -E "s#https://#https://x-access-token:${GITHUB_TOKEN}@#")"

git add experiments_gpu research/week logs 2>/dev/null || true
if git diff --cached --quiet; then
  echo "sync_results: nothing new to sync"
  exit 0
fi
git -c user.name="gpu-week-bot" -c user.email="nafis5341800@gmail.com" \
    commit -q -m "gpu-week: results sync $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git push -q "$AUTH_URL" HEAD:main
echo "sync_results: pushed at $(date -u +%H:%M:%SZ)"
