"""Web-triggered backtest runner — GitHub Actions edition.

The dashboard's "Run backtest" button POSTs to ``/api/run-backtest``;
this module dispatches the ``backtest.yml`` workflow on GitHub Actions
via the REST API. We do not run the replay locally — the existing repo
pattern is "Actions does heavy work, commits to main, local server
auto-pulls every 5 min" (see ``_git_auto_pull`` in ``server.py`` and
``.github/workflows/hourly_scan.yml``).

Lifecycle of a click
--------------------
1. ``start_backtest(mode)``  → resolves the GitHub repo from the local
   clone's ``origin`` remote, POSTs the workflow_dispatch with input
   ``mode``, writes ``state="running"`` + ``started_at`` to the
   shared status file. Returns ~immediately.
2. The frontend polls ``/api/backtest-status`` every 2s. Each call
   eventually triggers ``_sync_with_github()`` (rate-limited to once
   per ``GITHUB_POLL_INTERVAL_S``), which lists recent runs of
   ``backtest.yml`` on ``main``, finds the most recent run created
   ≥ ``started_at``, and updates the status to mirror its GitHub state.
3. When the run reports ``conclusion`` (success / failure / cancelled /
   skipped), the local status flips to ``completed`` or ``failed``,
   the run summary is recorded into ``history``, and polling stops.
4. The local server's existing auto-pull then syncs the new diagnostic
   markdown / panel CSVs into ``data/`` within ~5 min, after which the
   "View latest report" button picks up the fresh file.

Auth (in resolution order — the runner takes the first that works)
-------------------------------------------------------------------
1. **gh CLI** — if ``gh`` is installed and ``gh auth status`` reports
   a logged-in user, the runner shells out to ``gh api`` for both the
   dispatch and the polling. This is the zero-config path: nothing to
   set in the env, no PAT to manage. Just run ``gh auth login`` once.
2. **Personal access token** in env, in this order:
   ``BACKTEST_GITHUB_TOKEN`` → ``GITHUB_TOKEN``. Token needs the
   ``actions:write`` fine-grained scope (or classic ``workflow``+``repo``).

The workflow's own ``permissions:`` block (declared in backtest.yml)
governs what the workflow does *once it starts running* (e.g. commit
back to main). It cannot grant permission to whatever triggers it
from outside, which is why this auth layer is needed.

Status file shape (unchanged from the local-only version, with two
new GitHub-specific fields)::

    {
      "state": "idle" | "running" | "completed" | "failed",
      "mode": "replay" | "recal" | "both",
      "started_at": "..." | null,
      "completed_at": "..." | null,
      "duration_sec": int | null,
      "current_step": "queued" | "in_progress" | "done" | "failed",
      "error": "..." | null,
      "report_path": "data/diagnostic_replay_..." | null,
      "headline": "..." | null,
      "external_run_id": int | null,    # GitHub run id
      "external_run_url": str | null,   # link to the run page
      "history": [...]
    }
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
STATUS_FILE = DATA_DIR / "backtest_status.json"

VALID_MODES = {"replay", "recal", "both"}
HISTORY_MAX = 10
WORKFLOW_FILE = "backtest.yml"
WORKFLOW_REF = "main"
GITHUB_POLL_INTERVAL_S = 5.0
GITHUB_API_BASE = "https://api.github.com"

STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"

_LOCK = threading.Lock()


# ---- auth method resolution ------------------------------------------
#
# Two paths in priority order:
#   - "gh"    — shell out to the GitHub CLI; uses its keyring auth
#   - "token" — direct REST via urllib using a PAT from env
#
# Probe results are cached for the lifetime of the process — `gh` install
# state and PAT presence don't change between requests.


_AUTH_METHOD: str | None = None  # "gh" | "token" | None
_GH_BIN: str | None = None       # absolute path to gh, when found


def _get_token() -> str | None:
    return (
        os.environ.get("BACKTEST_GITHUB_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
        or None
    )


def _probe_gh_cli() -> str | None:
    """Return the gh binary path if it's installed AND authenticated."""
    import shutil

    binp = shutil.which("gh")
    if not binp:
        return None
    try:
        out = subprocess.run(
            [binp, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    # `gh auth status` exits 0 when at least one host is logged in. We
    # additionally check that the active token is *currently* valid —
    # gh prints "The token in keyring is invalid." with a non-zero exit
    # in that case (see the user's stale-keyring scenario).
    if out.returncode != 0:
        return None
    return binp


def _resolve_auth_method() -> str | None:
    """Pick the auth method to use, with caching. Returns "gh" | "token" | None."""
    global _AUTH_METHOD, _GH_BIN
    if _AUTH_METHOD is not None:
        return _AUTH_METHOD

    # Allow the env to force a specific path; useful for tests and for
    # users who explicitly want to bypass gh and use a PAT.
    forced = (os.environ.get("BACKTEST_AUTH_METHOD") or "").strip().lower()
    if forced in ("gh", "token"):
        _AUTH_METHOD = forced
        if forced == "gh":
            _GH_BIN = _probe_gh_cli()
        return _AUTH_METHOD

    _GH_BIN = _probe_gh_cli()
    if _GH_BIN:
        _AUTH_METHOD = "gh"
        return _AUTH_METHOD
    if _get_token():
        _AUTH_METHOD = "token"
        return _AUTH_METHOD
    return None


def _reset_auth_cache_for_tests() -> None:
    """Test-only: clear the cached auth probe."""
    global _AUTH_METHOD, _GH_BIN
    _AUTH_METHOD = None
    _GH_BIN = None


_REPO_RE = re.compile(
    r"(?:github\.com[:/])([^/]+)/([^/]+?)(?:\.git)?$"
)


def _resolve_repo() -> tuple[str, str] | None:
    """Read ``origin`` and parse out (owner, repo).

    Supports both HTTPS (``https://github.com/owner/repo.git``) and
    SSH (``git@github.com:owner/repo.git``) remote URLs.
    """
    override = os.environ.get("BACKTEST_GITHUB_REPO")
    if override and "/" in override:
        owner, _, repo = override.partition("/")
        return owner.strip(), repo.strip().removesuffix(".git")
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        m = _REPO_RE.search(out.stdout.strip())
        if not m:
            return None
        return m.group(1), m.group(2)
    except Exception:
        return None


# ---- status file helpers ---------------------------------------------


def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _empty_status() -> dict[str, Any]:
    return {
        "state": STATE_IDLE,
        "mode": None,
        "started_at": None,
        "completed_at": None,
        "duration_sec": None,
        "current_step": None,
        "error": None,
        "report_path": None,
        "headline": None,
        "external_run_id": None,
        "external_run_url": None,
        "_last_polled_at": None,
        "history": [],
    }


def _read_raw_status() -> dict[str, Any]:
    if not STATUS_FILE.exists():
        return _empty_status()
    try:
        data = json.loads(STATUS_FILE.read_text())
        empty = _empty_status()
        for k, v in empty.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return _empty_status()


def _write_status(status: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2, default=str))
    os.replace(tmp, STATUS_FILE)


def _update_status(**fields) -> dict[str, Any]:
    cur = _read_raw_status()
    cur.update(fields)
    _write_status(cur)
    return cur


def _record_history(status: dict[str, Any]) -> None:
    summary = {
        "started_at": status.get("started_at"),
        "completed_at": status.get("completed_at"),
        "duration_sec": status.get("duration_sec"),
        "mode": status.get("mode"),
        "state": status.get("state"),
        "headline": status.get("headline"),
        "report_path": status.get("report_path"),
        "external_run_url": status.get("external_run_url"),
        "error": status.get("error"),
    }
    history = list(status.get("history") or [])
    history.insert(0, summary)
    history = history[:HISTORY_MAX]
    status["history"] = history
    _write_status(status)


# ---- GitHub REST helpers ---------------------------------------------


class GitHubError(RuntimeError):
    pass


_NO_AUTH_MSG = (
    "no GitHub auth available. Either run `gh auth login` (preferred — "
    "zero-config; the runner uses its keyring) OR set "
    "BACKTEST_GITHUB_TOKEN / GITHUB_TOKEN to a PAT with the "
    "'actions:write' fine-grained scope (or classic 'workflow'+'repo')."
)


def _gh_request(
    method: str,
    path: str,
    *,
    json_body: dict | None = None,
    timeout: float = 15.0,
) -> tuple[int, dict | list | None]:
    """Make a GitHub REST call via the active auth method.

    Routes through the gh CLI when authenticated (no PAT needed),
    otherwise falls back to direct urllib + Bearer token. Returns
    (status_code, body).
    """
    method_choice = _resolve_auth_method()
    if method_choice == "gh":
        return _gh_request_via_cli(method, path, json_body=json_body, timeout=timeout)
    if method_choice == "token":
        return _gh_request_via_token(method, path, json_body=json_body, timeout=timeout)
    raise GitHubError(_NO_AUTH_MSG)


def _gh_request_via_cli(
    method: str,
    path: str,
    *,
    json_body: dict | None = None,
    timeout: float = 15.0,
) -> tuple[int, dict | list | None]:
    """Shell out to ``gh api``. The gh CLI handles auth via its keyring."""
    if not _GH_BIN:
        raise GitHubError("gh CLI not available")
    args = [_GH_BIN, "api", "--method", method, path,
            "-H", "Accept: application/vnd.github+json",
            "-H", "X-GitHub-Api-Version: 2022-11-28"]
    if json_body is not None:
        for k, v in (json_body.items() if isinstance(json_body, dict) else []):
            # gh api uses -f for string fields and -F for raw fields.
            # We always want raw JSON to preserve nested objects (e.g.
            # `inputs={"mode": "both"}`), so we serialize each top-level
            # key as a JSON value through the --raw-field convention.
            if isinstance(v, (dict, list)):
                args += ["--raw-field", f"{k}={json.dumps(v)}"]
            else:
                args += ["-f", f"{k}={v}"]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as te:
        raise GitHubError(f"gh api {method} {path} timed out after {timeout}s") from te
    if out.returncode != 0:
        # gh prints API errors to stderr in JSON-ish form. Surface tail.
        msg = (out.stderr or out.stdout or "").strip().splitlines()[-1:] or [""]
        raise GitHubError(f"gh api {method} {path} → rc={out.returncode}: {msg[0]}")
    raw = (out.stdout or "").strip()
    body = None
    if raw:
        try:
            body = json.loads(raw)
        except Exception:
            body = None
    return 200, body


def _gh_request_via_token(
    method: str,
    path: str,
    *,
    json_body: dict | None = None,
    timeout: float = 15.0,
) -> tuple[int, dict | list | None]:
    """urllib-based REST call using a PAT from the env."""
    token = _get_token()
    if not token:
        raise GitHubError(_NO_AUTH_MSG)

    url = GITHUB_API_BASE + path
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "vector-alpha-backtest-runner",
    }
    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            raw = resp.read().decode("utf-8") if status not in (204,) else ""
            body = json.loads(raw) if raw else None
            return status, body
    except urllib.error.HTTPError as he:
        try:
            body = json.loads(he.read().decode("utf-8"))
        except Exception:
            body = {"message": he.reason}
        raise GitHubError(f"GitHub {method} {path} → {he.code}: {body.get('message') or body}")
    except urllib.error.URLError as ue:
        raise GitHubError(f"GitHub {method} {path} network error: {ue}")


def _dispatch_workflow(owner: str, repo: str, mode: str) -> None:
    """POST a workflow_dispatch event with our input."""
    path = f"/repos/{owner}/{repo}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    _gh_request("POST", path, json_body={
        "ref": WORKFLOW_REF,
        "inputs": {"mode": mode},
    })


def _list_recent_runs(owner: str, repo: str, *, per_page: int = 10) -> list[dict]:
    path = (
        f"/repos/{owner}/{repo}/actions/workflows/{WORKFLOW_FILE}/runs"
        f"?per_page={per_page}&branch={WORKFLOW_REF}"
    )
    _, body = _gh_request("GET", path)
    if not body or not isinstance(body, dict):
        return []
    return body.get("workflow_runs") or []


# ---- public state-machine helpers ------------------------------------


def is_running() -> bool:
    return read_status().get("state") == STATE_RUNNING


def latest_report_path() -> Path | None:
    """Most recent ``diagnostic_replay_*.md`` file in data/, if any.

    Surfaces a freshly-pulled GitHub-Actions artifact: the local
    auto-pull syncs files committed by the workflow into ``data/``,
    so this just resolves whatever's on disk now.
    """
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


# ---- run-state mapping & polling -------------------------------------


def _map_github_to_local(run: dict) -> tuple[str, str | None]:
    """Translate (status, conclusion) into (local_state, current_step)."""
    gh_status = (run.get("status") or "").lower()
    gh_conclusion = (run.get("conclusion") or "").lower()

    if gh_status == "completed":
        if gh_conclusion == "success":
            return STATE_COMPLETED, "done"
        if gh_conclusion == "":
            # Race: status flipped before conclusion was written. Treat
            # as still running; next poll will resolve it.
            return STATE_RUNNING, "completing"
        return STATE_FAILED, gh_conclusion or "failed"

    if gh_status in ("queued", "waiting", "pending", "requested"):
        return STATE_RUNNING, "queued"
    if gh_status in ("in_progress", "running"):
        return STATE_RUNNING, "in_progress"
    return STATE_RUNNING, gh_status or "unknown"


def _parse_replay_headline() -> str | None:
    """Best-effort: read the freshest replay report and pull a Grade-A line.

    Same parser as before — works against whatever the auto-pull
    landed in ``data/``.
    """
    rp = latest_report_path()
    if not rp:
        return None
    try:
        text = rp.read_text(errors="replace")
    except Exception:
        return None
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


def _find_matching_run(runs: list[dict], started_at_iso: str) -> dict | None:
    """Pick the most recent run whose ``created_at`` is at or after our
    locally-recorded ``started_at``. We give 60 s of slack to absorb
    clock skew between the local laptop and GitHub.
    """
    if not runs:
        return None
    started_dt = _parse_iso(started_at_iso)
    if started_dt is None:
        return runs[0]
    threshold = started_dt - 60.0
    for run in runs:
        ca = _parse_iso(run.get("created_at"))
        if ca is None:
            continue
        if ca >= threshold:
            return run
    return runs[0]


def _parse_iso(s: str | None) -> float | None:
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def _sync_with_github() -> dict[str, Any]:
    """Refresh the cached status from GitHub if we're currently running.

    Throttled to ``GITHUB_POLL_INTERVAL_S`` per call so a chatty UI
    doesn't burn through API quota. Always returns a status dict, never
    raises — GitHub-side errors are recorded into ``status['error']``
    so the UI can surface them.
    """
    cur = _read_raw_status()
    if cur.get("state") != STATE_RUNNING:
        return cur

    last_polled = _parse_iso(cur.get("_last_polled_at"))
    if last_polled is not None and (time.time() - last_polled) < GITHUB_POLL_INTERVAL_S:
        return cur

    repo = _resolve_repo()
    if repo is None:
        cur["error"] = "could not resolve GitHub repo from origin remote"
        _write_status(cur)
        return cur
    owner, name = repo

    try:
        runs = _list_recent_runs(owner, name)
    except GitHubError as e:
        cur["error"] = str(e)
        cur["_last_polled_at"] = _now_iso()
        _write_status(cur)
        return cur

    started_at = cur.get("started_at") or _now_iso()
    run = _find_matching_run(runs, started_at)
    if run is None:
        # Dispatch may not have shown up yet; just record the poll
        # timestamp and let the next tick try again.
        cur["_last_polled_at"] = _now_iso()
        _write_status(cur)
        return cur

    new_state, step = _map_github_to_local(run)
    cur["external_run_id"] = run.get("id")
    cur["external_run_url"] = run.get("html_url")
    cur["current_step"] = step
    cur["_last_polled_at"] = _now_iso()
    cur["state"] = new_state

    if new_state in (STATE_COMPLETED, STATE_FAILED):
        cur["completed_at"] = _now_iso()
        started_ts = _parse_iso(cur.get("started_at"))
        if started_ts is not None:
            cur["duration_sec"] = max(0, int(time.time() - started_ts))
        if new_state == STATE_COMPLETED:
            # The auto-pull is what actually delivers the new files;
            # if it hasn't run yet, the report path resolves to whatever
            # was on disk before. Best-effort.
            rp = latest_report_path()
            cur["report_path"] = str(rp.relative_to(ROOT)) if rp else None
            cur["headline"] = _parse_replay_headline()
            cur["error"] = None
        else:
            cur["error"] = (
                f"workflow failed (conclusion={run.get('conclusion')}). "
                f"See: {run.get('html_url')}"
            )
        _record_history(cur)
    else:
        _write_status(cur)

    return cur


def read_status() -> dict[str, Any]:
    """Public read: reflects the latest state, refreshing from GitHub if needed."""
    return _sync_with_github()


# ---- public entrypoint -----------------------------------------------


def start_backtest(mode: str) -> dict[str, Any]:
    """Dispatch the workflow on GitHub Actions.

    Returns a dict with ``ok``, ``error``, and a snapshot of the status
    the UI should display immediately. Never raises.
    """
    if mode not in VALID_MODES:
        return {"ok": False, "error": f"invalid mode: {mode}", "status": _read_raw_status()}

    with _LOCK:
        cur = read_status()
        if cur.get("state") == STATE_RUNNING:
            return {
                "ok": False,
                "error": "a backtest is already in progress on GitHub Actions",
                "status": cur,
            }

        repo = _resolve_repo()
        if repo is None:
            return {
                "ok": False,
                "error": (
                    "could not resolve GitHub repo from 'origin' remote. "
                    "Set BACKTEST_GITHUB_REPO=owner/name to override."
                ),
                "status": cur,
            }

        try:
            _dispatch_workflow(repo[0], repo[1], mode)
        except GitHubError as e:
            return {"ok": False, "error": str(e), "status": cur}

        started_at = _now_iso()
        new_status = _update_status(
            state=STATE_RUNNING,
            mode=mode,
            started_at=started_at,
            completed_at=None,
            duration_sec=None,
            current_step="queued",
            error=None,
            report_path=None,
            headline=None,
            external_run_id=None,
            external_run_url=None,
            _last_polled_at=None,
        )
        return {"ok": True, "error": None, "status": new_status}


def auth_method_label() -> str:
    """Human-friendly label for the resolved auth path (for UI hints)."""
    m = _resolve_auth_method()
    if m == "gh":
        return "gh CLI"
    if m == "token":
        return "PAT (env var)"
    return "none"


def reset_status_for_tests() -> None:
    if STATUS_FILE.exists():
        STATUS_FILE.unlink()
