"""The post-main handoff must refuse incomplete work and never simulate in check-only mode."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import handoff_regime_after_main as handoff  # noqa: E402


def test_check_only_validates_report_without_launching(tmp_path, monkeypatch):
    main = tmp_path / "main.json"
    report = tmp_path / "comparison.json"
    main.write_text(json.dumps({"status": "complete", "counts": {"passed": 20},
                                "execution_source_fingerprint": "pinned"}), encoding="utf-8")
    saved = {"status": "complete", "expected_jobs": 20, "validated_jobs": 20,
             "validation_errors": [], "missing_or_unfinished_jobs": [],
             "manifest": str(main.resolve()), "source_execution_fingerprint": "pinned"}
    report.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(handoff, "MAIN", main)
    monkeypatch.setattr(handoff, "REPORT", report)
    monkeypatch.setattr(handoff, "_execution_source_fingerprint", lambda: "pinned")
    monkeypatch.setattr(handoff, "summarize", lambda path: saved)
    monkeypatch.setattr(handoff.subprocess, "run", lambda *args, **kwargs: 1 / 0)
    monkeypatch.setattr(sys, "argv", ["handoff", "--check-only"])

    assert handoff.main() == 0
    saved["status"] = "incomplete"
    report.write_text(json.dumps(saved), encoding="utf-8")
    assert handoff.main() == 2
    report.unlink()
    assert handoff.main() == 2


def test_completed_but_stale_report_is_refused(tmp_path, monkeypatch):
    main = tmp_path / "main.json"
    report = tmp_path / "comparison.json"
    main.write_text(json.dumps({"status": "complete", "counts": {"passed": 20},
                                "execution_source_fingerprint": "pinned"}), encoding="utf-8")
    saved = {"status": "complete", "expected_jobs": 20, "validated_jobs": 20,
             "validation_errors": [], "missing_or_unfinished_jobs": [],
             "manifest": str(main.resolve()), "source_execution_fingerprint": "pinned"}
    report.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(handoff, "MAIN", main)
    monkeypatch.setattr(handoff, "REPORT", report)
    monkeypatch.setattr(handoff, "_execution_source_fingerprint", lambda: "pinned")
    monkeypatch.setattr(handoff, "summarize", lambda path: {**saved, "manifest_sha256": "fresh"})
    monkeypatch.setattr(sys, "argv", ["handoff", "--check-only"])

    assert handoff.main() == 2


def test_active_simulator_blocks_normal_handoff(monkeypatch):
    from contextlib import nullcontext

    monkeypatch.setattr(handoff, "_require_current_wrapper", lambda pid, ticks: 123)
    monkeypatch.setattr(handoff, "_wrapper_start_ticks", lambda pid: None)
    monkeypatch.setattr(handoff, "_queue_lock", lambda path: nullcontext())
    monkeypatch.setattr(handoff, "_active_simulator_pids", lambda: [456])
    monkeypatch.setattr(handoff, "_validate_report", lambda: 1 / 0)
    monkeypatch.setattr(handoff.subprocess, "run", lambda *args, **kwargs: 1 / 0)
    monkeypatch.setattr(sys, "argv", ["handoff", "--main-pid", "123"])

    assert handoff.main() == 2


def test_watched_wrapper_disappearing_during_proc_read_counts_as_exit(monkeypatch):
    original = Path.read_text

    def vanished(path, *args, **kwargs):
        if path == Path("/proc/123/stat"):
            raise ProcessLookupError("process exited")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", vanished)
    assert handoff._wrapper_start_ticks(123) is None
