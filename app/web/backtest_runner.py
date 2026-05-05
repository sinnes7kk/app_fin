"""Web-triggered backtest runner.

Wires the existing CLI scripts (`scripts/build_replay_backtest.py` and
`scripts/recalibrate_conviction.py`) so the dashboard can trigger them
from a button instead of a terminal session.

Design notes
------------
- **Single source of truth**: the existing CLI scripts. We never
  reimplement their logic here; we only spawn them as subprocesses.
- **Status file** (`data/backtest_status.json`) is the only piece of
  state shared between the spawned subprocess and the Flask routes.
  Each subprocess writes progress to it; the Flask routes read it.
- **Concurrency**: a single in-process lock plus a state-machine check
  on the status file prevents two clicks from racing.
- **Failure mode**: if the subprocess crashes, we record the last 4 KB
  of its stderr in the status so the UI can surface it. We never raise
  back into the Flask request handler.
- **History**: we keep the last 10 runs inline in the status file so
  the UI can render a small drift-watch table without a separate store.

Status file shape::

    {
      "state": "idle" | "running" | "completed" | "failed",
      "mode": "replay" | "recal" | "both",
      "started_at": "2026-05-05T09:00:00",
      "completed_at": "2026-05-05T09:03:24" | null,
      "duration_sec": 204 | null,
      "rows_total": 104 | null,
      "rows_done": 67 | null,
      "current_step": "replaying" | "recalibrating" | "done",
      "error": "..." | null,
      "report_path": "data/diagnostic_replay_2026-05-05.md" | null,
      "history": [{...}, ...]
    }
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
STATUS_FILE = DATA_DIR / "backtest_status.json"
LOG_FILE = DATA_DIR / "backtest_run.log"

VALID_MODES = {"replay", "recal", "both"}
HISTORY_MAX = 10
STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"

_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None


# ---- status file helpers ----------------------------------------------


def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _empty_status() -> dict[str, Any]:
    return {
        "state": STATE_IDLE,
        "mode": None,
        "started_at": None,
        "completed_at": None,
        "duration_sec": None,
        "rows_total": None,
        "rows_done": None,
        "current_step": None,
        "error": None,
        "report_path": None,
        "headline": None,
        "history": [],
    }


def read_status() -> dict[str, Any]:
    """Return the current status dict (always a valid shape)."""
    if not STATUS_FILE.exists():
        return _empty_status()
    try:
        data = json.loads(STATUS_FILE.read_text())
        # Defensive: stamp any missing keys against the canonical shape so
        # the UI doesn't have to defend against partial writes from older
        # versions.
        empty = _empty_status()
        for k, v in empty.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return _empty_status()


def _write_status(status: dict[str, Any]) -> None:
    """Atomic-ish write so polling reads never see a partial JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2, default=str))
    os.replace(tmp, STATUS_FILE)


def _update_status(**fields) -> dict[str, Any]:
    """Merge ``fields`` into the current status and persist."""
    cur = read_status()
    cur.update(fields)
    _write_status(cur)
    return cur


def _record_history(status: dict[str, Any]) -> None:
    """Append the current run summary to the rolling history list."""
    summary = {
        "started_at": status.get("started_at"),
        "completed_at": status.get("completed_at"),
        "duration_sec": status.get("duration_sec"),
        "mode": status.get("mode"),
        "state": status.get("state"),
        "headline": status.get("headline"),
        "report_path": status.get("report_path"),
        "error": status.get("error"),
    }
    history = list(status.get("history") or [])
    history.insert(0, summary)
    history = history[:HISTORY_MAX]
    status["history"] = history
    _write_status(status)


# ---- runtime helpers --------------------------------------------------


def is_running() -> bool:
    return read_status().get("state") == STATE_RUNNING


def latest_report_path() -> Path | None:
    """Return the most recent ``diagnostic_replay_*.md`` file, if any."""
    candidates = sorted(
        DATA_DIR.glob("diagnostic_replay_*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def latest_recalibration_report() -> Path | None:
    candidates = sorted(
        DATA_DIR.glob("diagnostic_recalibration_*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# ---- subprocess execution --------------------------------------------


_PROGRESS_RE = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]")
_REPLAYING_TOTAL_RE = re.compile(r"Replaying\s+(\d+)\s+rows")


def _stream_subprocess(
    cmd: list[str],
    *,
    step_label: str,
    log_handle,
) -> tuple[int, str]:
    """Run ``cmd``, stream stdout to log + status, return (returncode, tail).

    The subprocess is launched with line-buffered stdout. Each line is:
      - written to the run log
      - parsed for progress markers like "[12/104] ..." which become
        ``rows_done``/``rows_total`` in the status file
      - kept (last 4 KB) so we can return a tail on failure
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        bufsize=1,
        text=True,
    )

    tail_lines: list[str] = []
    last_status_write = 0.0

    if proc.stdout is not None:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            log_handle.write(line + "\n")
            log_handle.flush()
            tail_lines.append(line)
            if len(tail_lines) > 200:
                tail_lines = tail_lines[-200:]

            now = time.time()
            updates: dict[str, Any] = {"current_step": step_label}

            m_total = _REPLAYING_TOTAL_RE.search(line)
            if m_total:
                updates["rows_total"] = int(m_total.group(1))
                updates["rows_done"] = 0

            m = _PROGRESS_RE.search(line)
            if m:
                updates["rows_done"] = int(m.group(1))
                if not read_status().get("rows_total"):
                    updates["rows_total"] = int(m.group(2))

            # Throttle status writes to ~once a second; a bursty
            # subprocess can otherwise hammer the disk.
            if (now - last_status_write) >= 0.75 or m_total is not None:
                _update_status(**updates)
                last_status_write = now

    proc.wait()
    return proc.returncode, "\n".join(tail_lines[-40:])


def _parse_replay_headline() -> str | None:
    """Best-effort: read the freshest replay report and pull a Grade-A line."""
    rp = latest_report_path()
    if not rp:
        return None
    try:
        text = rp.read_text(errors="replace")
    except Exception:
        return None
    # Section 2 of the report has a per-grade table; grab the row whose
    # first column (after the leading pipe) is "A" — coarse tier.
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Grade") and "n" in line:
            in_table = True
            continue
        if in_table:
            if line.startswith("| A "):
                return line.strip().strip("|").strip()
            if not line.startswith("|"):
                in_table = False
    return None


def _run_chain(mode: str) -> None:
    """Execute the requested chain. Always updates the status file with
    ``state="completed"`` or ``state="failed"`` before returning, even
    if a step explodes — the UI relies on that contract to re-enable
    the run button.
    """
    started_at = _now_iso()
    started_ts = time.time()
    _update_status(
        state=STATE_RUNNING,
        mode=mode,
        started_at=started_at,
        completed_at=None,
        duration_sec=None,
        rows_total=None,
        rows_done=None,
        current_step="starting",
        error=None,
        report_path=None,
        headline=None,
    )

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(LOG_FILE, "w")
    try:
        log_handle.write(f"==== Backtest run mode={mode} started {started_at} ====\n")
        log_handle.flush()
        py = sys.executable or "python3"

        if mode in ("replay", "both"):
            _update_status(current_step="replaying")
            rc, tail = _stream_subprocess(
                [py, str(ROOT / "scripts" / "build_replay_backtest.py")],
                step_label="replaying",
                log_handle=log_handle,
            )
            if rc != 0:
                _finalize_failure(
                    started_ts,
                    f"replay step exited rc={rc}",
                    tail,
                )
                return

        if mode in ("recal", "both"):
            _update_status(current_step="recalibrating")
            rc, tail = _stream_subprocess(
                [py, str(ROOT / "scripts" / "recalibrate_conviction.py")],
                step_label="recalibrating",
                log_handle=log_handle,
            )
            if rc != 0:
                _finalize_failure(
                    started_ts,
                    f"recalibration step exited rc={rc}",
                    tail,
                )
                return

        # success path
        rp = latest_report_path()
        cur = _update_status(
            state=STATE_COMPLETED,
            current_step="done",
            completed_at=_now_iso(),
            duration_sec=int(time.time() - started_ts),
            report_path=str(rp.relative_to(ROOT)) if rp else None,
            headline=_parse_replay_headline(),
            error=None,
        )
        _record_history(cur)
        log_handle.write(f"==== completed {_now_iso()} ====\n")
    except Exception as e:
        _finalize_failure(started_ts, f"unexpected: {e!r}", "")
    finally:
        try:
            log_handle.close()
        except Exception:
            pass


def _finalize_failure(started_ts: float, error_msg: str, tail: str) -> None:
    cur = _update_status(
        state=STATE_FAILED,
        completed_at=_now_iso(),
        duration_sec=int(time.time() - started_ts),
        current_step="failed",
        error=f"{error_msg}\n--- tail ---\n{tail}" if tail else error_msg,
    )
    _record_history(cur)


# ---- public entrypoint ------------------------------------------------


def start_backtest(mode: str) -> dict[str, Any]:
    """Kick off a backtest in a background thread.

    Returns a dict with ``ok`` (bool), ``error`` (str|None), and a
    snapshot of the status the UI should display immediately.
    """
    global _THREAD
    if mode not in VALID_MODES:
        return {"ok": False, "error": f"invalid mode: {mode}", "status": read_status()}

    with _LOCK:
        # Two layers of defense:
        #   1) the in-process thread we may have spawned earlier
        #   2) the status file (covers the case where the previous
        #      flask process crashed mid-run)
        if _THREAD is not None and _THREAD.is_alive():
            return {"ok": False, "error": "backtest already running", "status": read_status()}
        if is_running():
            # Likely stale — but be conservative; force the user to ack.
            return {
                "ok": False,
                "error": "status file shows a run already in progress",
                "status": read_status(),
            }

        t = threading.Thread(target=_run_chain, args=(mode,), daemon=True)
        _THREAD = t
        t.start()

    # Give the thread a short moment to flip the status to running so the
    # immediate response to the caller already shows the new state.
    time.sleep(0.05)
    return {"ok": True, "error": None, "status": read_status()}


def reset_status_for_tests() -> None:
    """Drop the status file. Test-only helper."""
    if STATUS_FILE.exists():
        STATUS_FILE.unlink()
    global _THREAD
    _THREAD = None
