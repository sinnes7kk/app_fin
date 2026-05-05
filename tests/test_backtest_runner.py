"""Unit tests for app/web/backtest_runner.py.

Covers the moving parts that aren't covered by simply running the CLI
scripts: status-file shape, lock semantics (start_backtest rejecting a
second run), success / failure finalization, history rotation, and
progress-line parsing.

We patch the subprocess execution so the tests don't actually run
build_replay_backtest.py (which would need yfinance + 5 minutes).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.web import backtest_runner as br


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    """Poll until predicate returns truthy or timeout elapses."""
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _stub_stream_subprocess_factory(rc: int = 0, lines: list[str] | None = None):
    """Build a fake _stream_subprocess that simulates subprocess output."""
    lines = lines or ["Replaying 4 rows…", "  [1/4] processed", "  [2/4] processed",
                      "  [3/4] processed", "  [4/4] processed", "Wrote: data/x.csv"]

    def _fake(cmd, *, step_label, log_handle):
        for line in lines:
            log_handle.write(line + "\n")
        # trigger one progress update so the runner records rows_total/rows_done
        try:
            br._update_status(rows_total=4, rows_done=len(lines), current_step=step_label)
        except Exception:
            pass
        return rc, "\n".join(lines[-4:])

    return _fake


# ---- status file shape -----------------------------------------------


def test_read_status_returns_default_shape_when_missing():
    br.reset_status_for_tests()
    s = br.read_status()
    assert s["state"] == "idle"
    assert s["history"] == []
    assert s["report_path"] is None
    print("  PASS: test_read_status_returns_default_shape_when_missing")


def test_read_status_normalizes_partial_old_status():
    br.reset_status_for_tests()
    # Simulate an older partial status file lacking some keys.
    br.STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    br.STATUS_FILE.write_text(json.dumps({"state": "completed"}))
    s = br.read_status()
    assert s["state"] == "completed"
    # All canonical keys are present even if missing from disk
    for k in ("mode", "started_at", "history", "rows_total", "headline"):
        assert k in s
    print("  PASS: test_read_status_normalizes_partial_old_status")


def test_read_status_recovers_from_corrupt_file():
    br.reset_status_for_tests()
    br.STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    br.STATUS_FILE.write_text("{not valid json")
    s = br.read_status()
    assert s["state"] == "idle"
    print("  PASS: test_read_status_recovers_from_corrupt_file")


# ---- mode validation -------------------------------------------------


def test_start_backtest_rejects_invalid_mode():
    br.reset_status_for_tests()
    r = br.start_backtest("not_a_mode")
    assert r["ok"] is False
    assert "invalid mode" in r["error"]
    print("  PASS: test_start_backtest_rejects_invalid_mode")


# ---- success path ----------------------------------------------------


def test_start_backtest_replay_only_succeeds():
    br.reset_status_for_tests()
    fake = _stub_stream_subprocess_factory(rc=0)
    with patch.object(br, "_stream_subprocess", side_effect=fake), \
         patch.object(br, "latest_report_path", return_value=None), \
         patch.object(br, "_parse_replay_headline", return_value="A | n=10 | +0.50R"):
        r = br.start_backtest("replay")
        assert r["ok"] is True
        ok = _wait_for(lambda: br.read_status()["state"] in ("completed", "failed"))
        assert ok, f"timed out, status={br.read_status()}"

    s = br.read_status()
    assert s["state"] == "completed", s
    assert s["mode"] == "replay"
    assert s["headline"] == "A | n=10 | +0.50R"
    assert s["error"] is None
    assert s["duration_sec"] is not None and s["duration_sec"] >= 0
    assert len(s["history"]) == 1
    print("  PASS: test_start_backtest_replay_only_succeeds")


def test_start_backtest_both_runs_two_subprocesses():
    br.reset_status_for_tests()
    calls: list[str] = []

    def _fake(cmd, *, step_label, log_handle):
        calls.append(step_label)
        log_handle.write(f"step {step_label} done\n")
        return 0, ""

    with patch.object(br, "_stream_subprocess", side_effect=_fake), \
         patch.object(br, "latest_report_path", return_value=None), \
         patch.object(br, "_parse_replay_headline", return_value=None):
        br.start_backtest("both")
        _wait_for(lambda: br.read_status()["state"] in ("completed", "failed"))

    s = br.read_status()
    assert s["state"] == "completed"
    assert calls == ["replaying", "recalibrating"], calls
    print("  PASS: test_start_backtest_both_runs_two_subprocesses")


def test_start_backtest_recal_only_skips_replay():
    br.reset_status_for_tests()
    calls: list[str] = []

    def _fake(cmd, *, step_label, log_handle):
        calls.append(step_label)
        return 0, ""

    with patch.object(br, "_stream_subprocess", side_effect=_fake), \
         patch.object(br, "latest_report_path", return_value=None):
        br.start_backtest("recal")
        _wait_for(lambda: br.read_status()["state"] in ("completed", "failed"))

    assert calls == ["recalibrating"], calls
    print("  PASS: test_start_backtest_recal_only_skips_replay")


# ---- failure path ----------------------------------------------------


def test_start_backtest_records_failure_when_replay_returns_nonzero():
    br.reset_status_for_tests()

    def _fake(cmd, *, step_label, log_handle):
        log_handle.write("boom\n")
        return 7, "boom"

    with patch.object(br, "_stream_subprocess", side_effect=_fake):
        br.start_backtest("replay")
        _wait_for(lambda: br.read_status()["state"] in ("completed", "failed"))

    s = br.read_status()
    assert s["state"] == "failed"
    assert "rc=7" in (s["error"] or "")
    assert s["history"] and s["history"][0]["state"] == "failed"
    print("  PASS: test_start_backtest_records_failure_when_replay_returns_nonzero")


def test_start_backtest_records_failure_when_recal_fails_after_replay_succeeds():
    br.reset_status_for_tests()

    def _fake(cmd, *, step_label, log_handle):
        return (0 if step_label == "replaying" else 9), step_label

    with patch.object(br, "_stream_subprocess", side_effect=_fake), \
         patch.object(br, "latest_report_path", return_value=None):
        br.start_backtest("both")
        _wait_for(lambda: br.read_status()["state"] in ("completed", "failed"))

    s = br.read_status()
    assert s["state"] == "failed"
    assert "rc=9" in (s["error"] or "")
    print("  PASS: test_start_backtest_records_failure_when_recal_fails_after_replay_succeeds")


# ---- concurrency -----------------------------------------------------


def test_start_backtest_rejects_second_call_while_running():
    br.reset_status_for_tests()
    gate_open = {"v": False}

    def _fake(cmd, *, step_label, log_handle):
        # Block until the test releases us, simulating a long-running run.
        for _ in range(40):
            if gate_open["v"]:
                break
            time.sleep(0.05)
        return 0, ""

    with patch.object(br, "_stream_subprocess", side_effect=_fake), \
         patch.object(br, "latest_report_path", return_value=None):
        first = br.start_backtest("replay")
        assert first["ok"] is True

        # While the first one is "running", a second call should be rejected.
        _wait_for(lambda: br.read_status()["state"] == "running")
        second = br.start_backtest("replay")
        assert second["ok"] is False
        assert "running" in (second["error"] or "")

        # Release the worker so the test can finish cleanly.
        gate_open["v"] = True
        _wait_for(lambda: br.read_status()["state"] in ("completed", "failed"))

    print("  PASS: test_start_backtest_rejects_second_call_while_running")


# ---- history rotation -------------------------------------------------


def test_history_rotates_to_max_entries():
    br.reset_status_for_tests()

    def _fake(cmd, *, step_label, log_handle):
        return 0, ""

    with patch.object(br, "_stream_subprocess", side_effect=_fake), \
         patch.object(br, "latest_report_path", return_value=None):
        for _ in range(br.HISTORY_MAX + 3):
            br.start_backtest("replay")
            _wait_for(lambda: br.read_status()["state"] in ("completed", "failed"))

    s = br.read_status()
    assert len(s["history"]) == br.HISTORY_MAX, len(s["history"])
    print("  PASS: test_history_rotates_to_max_entries")


# ---- progress regex ---------------------------------------------------


def test_progress_regex_parses_bracketed_counts():
    m = br._PROGRESS_RE.search("  [12/104] processed (skipped so far: 0)")
    assert m is not None
    assert m.group(1) == "12"
    assert m.group(2) == "104"
    print("  PASS: test_progress_regex_parses_bracketed_counts")


def test_replaying_total_regex_parses_initial_count():
    m = br._REPLAYING_TOTAL_RE.search("Replaying 104 rows…")
    assert m is not None
    assert m.group(1) == "104"
    print("  PASS: test_replaying_total_regex_parses_initial_count")


# ---- runner ---------------------------------------------------------


def main():
    tests = [
        test_read_status_returns_default_shape_when_missing,
        test_read_status_normalizes_partial_old_status,
        test_read_status_recovers_from_corrupt_file,
        test_start_backtest_rejects_invalid_mode,
        test_start_backtest_replay_only_succeeds,
        test_start_backtest_both_runs_two_subprocesses,
        test_start_backtest_recal_only_skips_replay,
        test_start_backtest_records_failure_when_replay_returns_nonzero,
        test_start_backtest_records_failure_when_recal_fails_after_replay_succeeds,
        test_start_backtest_rejects_second_call_while_running,
        test_history_rotates_to_max_entries,
        test_progress_regex_parses_bracketed_counts,
        test_replaying_total_regex_parses_initial_count,
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
    if failures:
        print(f"\n{failures} test(s) failed.")
        return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
