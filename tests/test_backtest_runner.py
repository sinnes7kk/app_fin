"""Unit tests for app/web/backtest_runner.py (GitHub Actions edition).

The runner now dispatches a workflow_dispatch event on GitHub Actions
instead of running the CLI scripts locally. These tests stub out the
GitHub HTTP client (``_gh_request``) so they don't hit the network.

Coverage:
  - status-file shape, recovery from missing/corrupt/partial files
  - mode validation
  - dispatch refused without a token
  - dispatch refused when origin remote can't be parsed
  - dispatch happy path → status flips to "running" with started_at
  - second click while running is rejected
  - poll → in_progress, queued, completed-success, completed-failure
  - poll throttles to GITHUB_POLL_INTERVAL_S
  - history rotates at MAX
  - repo URL parser handles HTTPS + SSH + override env var
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.web import backtest_runner as br


def _setenv(token: str = "fake-token", repo: str = "owner/repo"):
    os.environ["BACKTEST_GITHUB_TOKEN"] = token
    os.environ["BACKTEST_GITHUB_REPO"] = repo
    # Force PAT path in tests so we don't accidentally invoke a real `gh`
    # binary on the dev's machine. Tests that exercise the gh path set
    # BACKTEST_AUTH_METHOD=gh and patch _probe_gh_cli.
    os.environ["BACKTEST_AUTH_METHOD"] = "token"
    br._reset_auth_cache_for_tests()


def _clearenv():
    for k in (
        "BACKTEST_GITHUB_TOKEN", "GITHUB_TOKEN", "BACKTEST_GITHUB_REPO",
        "BACKTEST_AUTH_METHOD",
    ):
        os.environ.pop(k, None)
    br._reset_auth_cache_for_tests()


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ---- repo + token resolution -----------------------------------------


def test_repo_resolution_from_https_remote():
    _clearenv()
    with patch.object(br.subprocess, "run") as mock_run:
        mock_run.return_value = type("R", (), {
            "returncode": 0,
            "stdout": "https://github.com/sinnes7kk/app_fin.git\n",
            "stderr": "",
        })()
        owner, name = br._resolve_repo()
    assert owner == "sinnes7kk" and name == "app_fin"
    print("  PASS: test_repo_resolution_from_https_remote")


def test_repo_resolution_from_ssh_remote():
    _clearenv()
    with patch.object(br.subprocess, "run") as mock_run:
        mock_run.return_value = type("R", (), {
            "returncode": 0,
            "stdout": "git@github.com:sinnes7kk/app_fin.git\n",
            "stderr": "",
        })()
        owner, name = br._resolve_repo()
    assert owner == "sinnes7kk" and name == "app_fin"
    print("  PASS: test_repo_resolution_from_ssh_remote")


def test_repo_override_env_var_wins():
    _setenv(repo="custom_owner/custom_repo")
    # subprocess shouldn't even be called
    with patch.object(br.subprocess, "run") as mock_run:
        owner, name = br._resolve_repo()
        mock_run.assert_not_called()
    assert owner == "custom_owner" and name == "custom_repo"
    print("  PASS: test_repo_override_env_var_wins")


def test_repo_resolution_returns_none_on_unknown_remote():
    _clearenv()
    with patch.object(br.subprocess, "run") as mock_run:
        mock_run.return_value = type("R", (), {
            "returncode": 0,
            "stdout": "ssh://example.com/x/y\n",
            "stderr": "",
        })()
        out = br._resolve_repo()
    assert out is None
    print("  PASS: test_repo_resolution_returns_none_on_unknown_remote")


def test_get_token_resolution_order():
    _clearenv()
    assert br._get_token() is None
    os.environ["GITHUB_TOKEN"] = "fallback"
    assert br._get_token() == "fallback"
    os.environ["BACKTEST_GITHUB_TOKEN"] = "preferred"
    assert br._get_token() == "preferred"
    _clearenv()
    print("  PASS: test_get_token_resolution_order")


# ---- auth method resolution ------------------------------------------


def test_auth_method_prefers_gh_when_available():
    _clearenv()
    with patch.object(br, "_probe_gh_cli", return_value="/fake/gh"):
        m = br._resolve_auth_method()
    assert m == "gh"
    print("  PASS: test_auth_method_prefers_gh_when_available")


def test_auth_method_falls_back_to_token_when_gh_unauthed():
    _clearenv()
    os.environ["GITHUB_TOKEN"] = "tok"
    with patch.object(br, "_probe_gh_cli", return_value=None):
        m = br._resolve_auth_method()
    assert m == "token"
    _clearenv()
    print("  PASS: test_auth_method_falls_back_to_token_when_gh_unauthed")


def test_auth_method_returns_none_when_neither_available():
    _clearenv()
    with patch.object(br, "_probe_gh_cli", return_value=None):
        m = br._resolve_auth_method()
    assert m is None
    print("  PASS: test_auth_method_returns_none_when_neither_available")


def test_auth_method_env_override_forces_token():
    _clearenv()
    os.environ["BACKTEST_AUTH_METHOD"] = "token"
    os.environ["GITHUB_TOKEN"] = "tok"
    with patch.object(br, "_probe_gh_cli", return_value="/fake/gh") as p:
        m = br._resolve_auth_method()
        # Probe should NOT be invoked since the override forces a method.
        p.assert_not_called()
    assert m == "token"
    _clearenv()
    print("  PASS: test_auth_method_env_override_forces_token")


def test_auth_method_label_user_facing():
    _clearenv()
    with patch.object(br, "_probe_gh_cli", return_value="/fake/gh"):
        assert br.auth_method_label() == "gh CLI"
    _clearenv()
    os.environ["GITHUB_TOKEN"] = "x"
    with patch.object(br, "_probe_gh_cli", return_value=None):
        assert br.auth_method_label() == "PAT (env var)"
    _clearenv()
    with patch.object(br, "_probe_gh_cli", return_value=None):
        assert br.auth_method_label() == "none"
    print("  PASS: test_auth_method_label_user_facing")


# ---- gh CLI path -----------------------------------------------------


def test_dispatch_via_gh_cli_calls_subprocess_with_correct_args():
    br.reset_status_for_tests()
    _clearenv()
    os.environ["BACKTEST_GITHUB_REPO"] = "owner/repo"
    os.environ["BACKTEST_AUTH_METHOD"] = "gh"

    captured: dict = {}

    def fake_run(args, capture_output=False, text=False, timeout=None, **kw):
        captured["args"] = args
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with patch.object(br, "_probe_gh_cli", return_value="/fake/gh"), \
         patch.object(br.subprocess, "run", side_effect=fake_run):
        r = br.start_backtest("replay")

    assert r["ok"] is True, r
    args = captured["args"]
    # gh api --method POST /repos/owner/repo/actions/workflows/backtest.yml/dispatches ...
    assert args[0] == "/fake/gh"
    assert args[1] == "api"
    assert "--method" in args and "POST" in args
    assert any("/dispatches" in a for a in args)
    # Inputs preserved as raw JSON
    assert any("ref=" in a and "main" in a for a in args)
    assert any('inputs={"mode": "replay"}' in a for a in args)
    _clearenv()
    print("  PASS: test_dispatch_via_gh_cli_calls_subprocess_with_correct_args")


def test_gh_cli_failure_surfaces_clean_error():
    br.reset_status_for_tests()
    _clearenv()
    os.environ["BACKTEST_GITHUB_REPO"] = "owner/repo"
    os.environ["BACKTEST_AUTH_METHOD"] = "gh"

    def fake_run(args, capture_output=False, text=False, timeout=None, **kw):
        return type("R", (), {
            "returncode": 1,
            "stdout": "",
            "stderr": '{"message":"Bad credentials"}',
        })()

    with patch.object(br, "_probe_gh_cli", return_value="/fake/gh"), \
         patch.object(br.subprocess, "run", side_effect=fake_run):
        r = br.start_backtest("replay")

    assert r["ok"] is False
    assert "rc=1" in (r["error"] or "")
    _clearenv()
    print("  PASS: test_gh_cli_failure_surfaces_clean_error")


def test_no_auth_returns_helpful_error():
    br.reset_status_for_tests()
    _clearenv()
    os.environ["BACKTEST_GITHUB_REPO"] = "owner/repo"
    with patch.object(br, "_probe_gh_cli", return_value=None):
        r = br.start_backtest("replay")
    assert r["ok"] is False
    assert "no GitHub auth" in (r["error"] or "")
    assert "gh auth login" in (r["error"] or "")
    _clearenv()
    print("  PASS: test_no_auth_returns_helpful_error")


# ---- status-file shape -----------------------------------------------


def test_read_status_default_shape_when_missing():
    br.reset_status_for_tests()
    s = br._read_raw_status()
    assert s["state"] == "idle"
    assert s["history"] == []
    print("  PASS: test_read_status_default_shape_when_missing")


def test_read_status_normalizes_partial_status_file():
    br.reset_status_for_tests()
    br.STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    br.STATUS_FILE.write_text(json.dumps({"state": "completed"}))
    s = br._read_raw_status()
    assert s["state"] == "completed"
    for k in ("mode", "started_at", "history", "external_run_id", "external_run_url"):
        assert k in s
    print("  PASS: test_read_status_normalizes_partial_status_file")


def test_read_status_recovers_from_corrupt_file():
    br.reset_status_for_tests()
    br.STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    br.STATUS_FILE.write_text("{not valid json")
    s = br._read_raw_status()
    assert s["state"] == "idle"
    print("  PASS: test_read_status_recovers_from_corrupt_file")


# ---- start_backtest --------------------------------------------------


def test_start_backtest_rejects_invalid_mode():
    br.reset_status_for_tests()
    _setenv()
    r = br.start_backtest("not_a_mode")
    assert r["ok"] is False and "invalid mode" in r["error"]
    _clearenv()
    print("  PASS: test_start_backtest_rejects_invalid_mode")


def test_start_backtest_fails_without_any_auth():
    br.reset_status_for_tests()
    _clearenv()
    os.environ["BACKTEST_GITHUB_REPO"] = "owner/repo"
    with patch.object(br, "_probe_gh_cli", return_value=None):
        r = br.start_backtest("both")
    assert r["ok"] is False and "no GitHub auth" in r["error"]
    _clearenv()
    print("  PASS: test_start_backtest_fails_without_any_auth")


def test_start_backtest_fails_when_repo_unresolvable():
    br.reset_status_for_tests()
    _clearenv()
    os.environ["BACKTEST_GITHUB_TOKEN"] = "tok"
    with patch.object(br, "_resolve_repo", return_value=None):
        r = br.start_backtest("both")
    assert r["ok"] is False and "could not resolve" in r["error"]
    _clearenv()
    print("  PASS: test_start_backtest_fails_when_repo_unresolvable")


def test_start_backtest_dispatches_and_writes_running_status():
    br.reset_status_for_tests()
    _setenv()
    captured: dict = {}

    def _fake(method, path, *, json_body=None, timeout=15.0):
        captured["method"] = method
        captured["path"] = path
        captured["body"] = json_body
        return 204, None

    with patch.object(br, "_gh_request", side_effect=_fake):
        r = br.start_backtest("replay")
    assert r["ok"] is True
    assert captured["method"] == "POST"
    assert captured["path"].endswith("/actions/workflows/backtest.yml/dispatches")
    assert captured["body"] == {"ref": "main", "inputs": {"mode": "replay"}}

    s = br._read_raw_status()
    assert s["state"] == "running"
    assert s["mode"] == "replay"
    assert s["started_at"] is not None
    assert s["current_step"] == "queued"
    _clearenv()
    print("  PASS: test_start_backtest_dispatches_and_writes_running_status")


def test_start_backtest_rejects_second_call_while_running():
    br.reset_status_for_tests()
    _setenv()
    with patch.object(br, "_gh_request", return_value=(204, None)):
        first = br.start_backtest("both")
        assert first["ok"] is True
        # `read_status` will try to refresh from GitHub; stub the listing to
        # return the run as still in_progress so the lock holds.
        run = {
            "id": 1, "html_url": "https://gh/run/1",
            "status": "in_progress", "conclusion": None,
            "created_at": br._now_iso() + "Z",
        }
        with patch.object(br, "_list_recent_runs", return_value=[run]):
            second = br.start_backtest("both")
    assert second["ok"] is False
    assert "already in progress" in (second["error"] or "")
    _clearenv()
    print("  PASS: test_start_backtest_rejects_second_call_while_running")


# ---- polling lifecycle -----------------------------------------------


def _make_run(status, conclusion=None, run_id=42, url="https://gh/run/42"):
    return {
        "id": run_id,
        "html_url": url,
        "status": status,
        "conclusion": conclusion,
        "created_at": br._now_iso() + "Z",
    }


def test_sync_marks_state_running_with_in_progress_run():
    br.reset_status_for_tests()
    _setenv()
    with patch.object(br, "_gh_request", return_value=(204, None)):
        br.start_backtest("both")
    run = _make_run("in_progress")
    with patch.object(br, "_list_recent_runs", return_value=[run]):
        s = br.read_status()
    assert s["state"] == "running"
    assert s["current_step"] == "in_progress"
    assert s["external_run_id"] == 42
    assert s["external_run_url"] == "https://gh/run/42"
    _clearenv()
    print("  PASS: test_sync_marks_state_running_with_in_progress_run")


def test_sync_handles_queued_run():
    br.reset_status_for_tests()
    _setenv()
    with patch.object(br, "_gh_request", return_value=(204, None)):
        br.start_backtest("both")
    run = _make_run("queued")
    with patch.object(br, "_list_recent_runs", return_value=[run]):
        s = br.read_status()
    assert s["state"] == "running"
    assert s["current_step"] == "queued"
    _clearenv()
    print("  PASS: test_sync_handles_queued_run")


def test_sync_marks_completed_on_success():
    br.reset_status_for_tests()
    _setenv()
    with patch.object(br, "_gh_request", return_value=(204, None)):
        br.start_backtest("both")
    run = _make_run("completed", conclusion="success")
    with patch.object(br, "_list_recent_runs", return_value=[run]), \
         patch.object(br, "latest_report_path", return_value=None), \
         patch.object(br, "_parse_replay_headline", return_value="A | n=10 | +0.50R"):
        s = br.read_status()
    assert s["state"] == "completed"
    assert s["headline"] == "A | n=10 | +0.50R"
    assert s["error"] is None
    assert len(s["history"]) == 1
    _clearenv()
    print("  PASS: test_sync_marks_completed_on_success")


def test_sync_marks_failed_on_failure_conclusion():
    br.reset_status_for_tests()
    _setenv()
    with patch.object(br, "_gh_request", return_value=(204, None)):
        br.start_backtest("both")
    run = _make_run("completed", conclusion="failure", url="https://gh/run/99")
    with patch.object(br, "_list_recent_runs", return_value=[run]):
        s = br.read_status()
    assert s["state"] == "failed"
    assert "https://gh/run/99" in (s["error"] or "")
    assert len(s["history"]) == 1
    assert s["history"][0]["state"] == "failed"
    _clearenv()
    print("  PASS: test_sync_marks_failed_on_failure_conclusion")


def test_sync_throttles_polling_to_interval():
    br.reset_status_for_tests()
    _setenv()
    with patch.object(br, "_gh_request", return_value=(204, None)):
        br.start_backtest("both")

    counter = {"n": 0}

    def fake_list(owner, name):
        counter["n"] += 1
        return [_make_run("in_progress")]

    with patch.object(br, "_list_recent_runs", side_effect=fake_list):
        # Three rapid-fire reads should produce a single GitHub call thanks
        # to the throttle.
        br.read_status()
        br.read_status()
        br.read_status()
    assert counter["n"] == 1, counter
    _clearenv()
    print("  PASS: test_sync_throttles_polling_to_interval")


def test_sync_records_github_error_into_status():
    br.reset_status_for_tests()
    _setenv()
    with patch.object(br, "_gh_request", return_value=(204, None)):
        br.start_backtest("both")
    with patch.object(br, "_list_recent_runs",
                      side_effect=br.GitHubError("boom: 500")):
        s = br.read_status()
    assert s["state"] == "running"  # still running, but error noted
    assert "boom: 500" in (s["error"] or "")
    _clearenv()
    print("  PASS: test_sync_records_github_error_into_status")


def test_sync_no_op_when_state_is_idle():
    br.reset_status_for_tests()
    _setenv()
    counter = {"n": 0}

    def fake_list(owner, name):
        counter["n"] += 1
        return []

    with patch.object(br, "_list_recent_runs", side_effect=fake_list):
        br.read_status()
        br.read_status()
    assert counter["n"] == 0
    _clearenv()
    print("  PASS: test_sync_no_op_when_state_is_idle")


# ---- history rotation -------------------------------------------------


def test_history_rotates_to_max_entries():
    br.reset_status_for_tests()
    _setenv()
    for i in range(br.HISTORY_MAX + 3):
        with patch.object(br, "_gh_request", return_value=(204, None)):
            br.start_backtest("replay")
        run = _make_run("completed", conclusion="success", run_id=1000 + i)
        with patch.object(br, "_list_recent_runs", return_value=[run]), \
             patch.object(br, "latest_report_path", return_value=None), \
             patch.object(br, "_parse_replay_headline", return_value=None):
            br.read_status()
    s = br._read_raw_status()
    assert len(s["history"]) == br.HISTORY_MAX
    _clearenv()
    print("  PASS: test_history_rotates_to_max_entries")


# ---- runner ---------------------------------------------------------


def main():
    tests = [
        test_repo_resolution_from_https_remote,
        test_repo_resolution_from_ssh_remote,
        test_repo_override_env_var_wins,
        test_repo_resolution_returns_none_on_unknown_remote,
        test_get_token_resolution_order,
        test_auth_method_prefers_gh_when_available,
        test_auth_method_falls_back_to_token_when_gh_unauthed,
        test_auth_method_returns_none_when_neither_available,
        test_auth_method_env_override_forces_token,
        test_auth_method_label_user_facing,
        test_dispatch_via_gh_cli_calls_subprocess_with_correct_args,
        test_gh_cli_failure_surfaces_clean_error,
        test_no_auth_returns_helpful_error,
        test_read_status_default_shape_when_missing,
        test_read_status_normalizes_partial_status_file,
        test_read_status_recovers_from_corrupt_file,
        test_start_backtest_rejects_invalid_mode,
        test_start_backtest_fails_without_any_auth,
        test_start_backtest_fails_when_repo_unresolvable,
        test_start_backtest_dispatches_and_writes_running_status,
        test_start_backtest_rejects_second_call_while_running,
        test_sync_marks_state_running_with_in_progress_run,
        test_sync_handles_queued_run,
        test_sync_marks_completed_on_success,
        test_sync_marks_failed_on_failure_conclusion,
        test_sync_throttles_polling_to_interval,
        test_sync_records_github_error_into_status,
        test_sync_no_op_when_state_is_idle,
        test_history_rotates_to_max_entries,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failures += 1
        except Exception as e:
            print(f"  ERROR: {t.__name__}: {type(e).__name__}: {e}")
            failures += 1
        finally:
            br.reset_status_for_tests()
            _clearenv()
    if failures:
        print(f"\n{failures} test(s) failed.")
        return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
