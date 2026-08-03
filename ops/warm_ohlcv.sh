#!/bin/bash
# Weekly OHLCV cache refresh, pushed to main ahead of the Sunday backtest.
#
# Yahoo blocks the GitHub runners, so CI cannot refill data/_ohlcv_cache on its
# own. This runs here instead, where the fetch actually works. Scheduled by
# ops/com.appfin.warm-ohlcv.plist.
#
# All work happens in a dedicated worktree — never the interactive checkout,
# which routinely holds uncommitted work that a scripted reset would destroy.

set -uo pipefail

REPO="/Users/philip.sierpinski/Documents/app_fin"
WORKTREE="$HOME/.cache/app_fin_ohlcv_warm"
PYTHON="$REPO/.venv/bin/python"
LOG="$REPO/data/ohlcv_warm.log"
ATTEMPTS=3

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== warm start ==="

if [ ! -x "$PYTHON" ]; then
  log "FATAL: no interpreter at $PYTHON"
  exit 1
fi

if [ ! -d "$WORKTREE/.git" ] && [ ! -f "$WORKTREE/.git" ]; then
  log "creating worktree at $WORKTREE"
  mkdir -p "$(dirname "$WORKTREE")"
  git -C "$REPO" worktree add --detach "$WORKTREE" origin/main >>"$LOG" 2>&1 || {
    log "FATAL: could not create worktree"
    exit 1
  }
fi

for attempt in $(seq 1 $ATTEMPTS); do
  log "attempt $attempt/$ATTEMPTS"

  git -C "$WORKTREE" fetch origin main --quiet >>"$LOG" 2>&1
  git -C "$WORKTREE" reset --hard origin/main --quiet >>"$LOG" 2>&1 || {
    log "could not sync worktree to origin/main"
    continue
  }

  if ! (cd "$WORKTREE" && "$PYTHON" scripts/warm_ohlcv_cache.py >>"$LOG" 2>&1); then
    log "warm script failed (see above) — leaving the committed cache alone"
    continue
  fi

  git -C "$WORKTREE" add data/_ohlcv_cache >>"$LOG" 2>&1
  if git -C "$WORKTREE" diff --cached --quiet; then
    log "cache already current, nothing to push"
    exit 0
  fi

  git -C "$WORKTREE" commit -q -m "chore(data): weekly OHLCV cache warm" >>"$LOG" 2>&1

  # A losing push means the hourly scan landed first; the retry re-warms on top
  # of its commit rather than trying to reconcile hundreds of CSVs by hand.
  if git -C "$WORKTREE" push origin HEAD:main >>"$LOG" 2>&1; then
    log "pushed $(git -C "$WORKTREE" rev-parse --short HEAD)"
    log "=== warm ok ==="
    exit 0
  fi
  log "push rejected, retrying"
done

log "=== warm FAILED after $ATTEMPTS attempts ==="
exit 1
