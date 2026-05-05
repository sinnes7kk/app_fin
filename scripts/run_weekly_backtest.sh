#!/usr/bin/env bash
# Weekly replay-backtest + recalibration runner intended for cron.
#
# What it does
# ------------
# 1. cd into the repo
# 2. Activate the project venv if present (.venv/ or venv/)
# 3. Run scripts/build_replay_backtest.py
# 4. If the replay step exits 0, run scripts/recalibrate_conviction.py
# 5. Append all stdout+stderr to data/cron_backtest.log
#
# Concurrency / safety
# --------------------
# - Uses a flock-style lock at data/.cron_backtest.lock so two cron firings
#   never run simultaneously (e.g. if you forgot you also clicked the UI button).
# - Exit 75 (EX_TEMPFAIL) when another run is already in progress, which cron
#   logs as a transient failure rather than a real error.
#
# Suggested crontab line (Sundays at 18:00 ET ≈ 22:00 UTC during DST)
# -------------------------------------------------------------------
#     0 22 * * 0  /Users/<you>/Documents/app_fin/scripts/run_weekly_backtest.sh
#
# Or if you prefer a launchd LaunchAgent on macOS, see the README section
# at the bottom of this file (search for "launchd").

set -u
set -o pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/data"
LOG_FILE="$LOG_DIR/cron_backtest.log"
LOCK_FILE="$LOG_DIR/.cron_backtest.lock"
mkdir -p "$LOG_DIR"

ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }

log() { echo "[$(ts)] $*" >> "$LOG_FILE"; }

# Lock — only one weekly run at a time.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "weekly-backtest: another run is already in progress, skipping"
  exit 75
fi

log "weekly-backtest: starting"

# Pick a python interpreter. Prefer the project venv if present.
PY=""
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PY="$REPO_ROOT/.venv/bin/python"
elif [ -x "$REPO_ROOT/venv/bin/python" ]; then
  PY="$REPO_ROOT/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  log "weekly-backtest: no python interpreter found"
  exit 70
fi
log "weekly-backtest: using python: $PY"

# Step 1 — replay
log "weekly-backtest: starting replay"
"$PY" "$REPO_ROOT/scripts/build_replay_backtest.py" >>"$LOG_FILE" 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then
  log "weekly-backtest: replay step exited rc=$RC, aborting"
  exit "$RC"
fi
log "weekly-backtest: replay done"

# Step 2 — recalibration
log "weekly-backtest: starting recalibration"
"$PY" "$REPO_ROOT/scripts/recalibrate_conviction.py" >>"$LOG_FILE" 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then
  log "weekly-backtest: recalibration step exited rc=$RC"
  exit "$RC"
fi
log "weekly-backtest: recalibration done"

log "weekly-backtest: success"
exit 0

# ---------------------------------------------------------------------------
# launchd LaunchAgent (macOS) — alternative to cron
# ---------------------------------------------------------------------------
# Save as ~/Library/LaunchAgents/com.vector-alpha.weekly-backtest.plist:
#
# <?xml version="1.0" encoding="UTF-8"?>
# <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
#  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
# <plist version="1.0">
# <dict>
#   <key>Label</key>
#   <string>com.vector-alpha.weekly-backtest</string>
#   <key>ProgramArguments</key>
#   <array>
#     <string>/bin/bash</string>
#     <string>/Users/<you>/Documents/app_fin/scripts/run_weekly_backtest.sh</string>
#   </array>
#   <key>StartCalendarInterval</key>
#   <dict>
#     <key>Weekday</key><integer>0</integer>
#     <key>Hour</key><integer>18</integer>
#     <key>Minute</key><integer>0</integer>
#   </dict>
#   <key>StandardOutPath</key>
#   <string>/Users/<you>/Documents/app_fin/data/cron_backtest.log</string>
#   <key>StandardErrorPath</key>
#   <string>/Users/<you>/Documents/app_fin/data/cron_backtest.log</string>
# </dict>
# </plist>
#
# Then load with: launchctl load ~/Library/LaunchAgents/com.vector-alpha.weekly-backtest.plist
