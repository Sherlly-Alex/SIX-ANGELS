from __future__ import annotations

import json
from pathlib import Path
import tempfile
import sys

import pytest

from learning.heldout_gate import PolicySummary, assess_heldout_gate
from learning.replay_split import split_replay_dataset
from learning.event_replay import replay_event_logs, write_replay_dataset
from scheduler.candidate_generator import CandidateAction
from scheduler.decision import SchedulerDecisionService
from scheduler.events import EventLog, JsonlEventSink

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from validate_rl1_heldout import validate_split_manifest
import run_rl1_pipeline as rl1_pipeline


def _dataset(path: Path, sessions: int = 5) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    records = []
    for session in range(sessions):
        event = path.parent / f"events-{session}.jsonl"
        log = EventLog([JsonlEventSink(event)], clock=lambda: 0.0)
        log.emit({"event_type": "scheduler_started", "engine": "v2"})
        service = SchedulerDecisionService(event_log=log)
        service.decide((CandidateAction("best", "rescan", expected_score=2, success_probability=.9),
                        CandidateAction("low", "rescan", expected_score=1, success_probability=.9)),
                       now_s=1.0, world_state={}, event_fields={"task_id": 1, "attempt": 1,
                       "step_id": "x", "task_run_id": f"task-{session}",
                       "attempt_run_id": f"attempt-{session}", "step_run_id": f"step-{session}"})
        service.close()
        summary, values = replay_event_logs([event], min_decisions=1, require_training_ready=True)
        assert summary.passed
        record = values[0].to_json_dict()
        record["source_sha256"] = f"{session + 1:064x}"
        record["session_index"] = 1
        record["decision_id"] = f"decision-{session}"
        records.append(record)
    dataset = path / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n" for value in records), encoding="utf-8")
    return dataset


def test_split_is_deterministic_and_session_disjoint() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = _dataset(root)
        first = split_replay_dataset(dataset, root / "one", seed=7)
        second = split_replay_dataset(dataset, root / "two", seed=7)
        assert first.manifest["sessions"] == second.manifest["sessions"]
        sets = [set((x["source_sha256"], x["session_index"]) for x in first.manifest["sessions"][name]) for name in ("train", "validation", "test")]
        assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
        assert sum(first.manifest["record_counts"].values()) == 5
        different = split_replay_dataset(dataset, root / "different", seed=8)
        assert first.manifest["sessions"] != different.manifest["sessions"]
        for count in (6, 7):
            extra = _dataset(root / f"extra-{count}", sessions=count)
            result = split_replay_dataset(extra, root / f"split-{count}", seed=7)
            assert result.manifest["session_counts"] == {
                "train": count - 2, "validation": 1, "test": 1
            }
            assert sum(result.manifest["record_counts"].values()) == count


def test_split_rejects_too_few_sessions_and_nonempty_output() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = _dataset(root, sessions=4)
        with pytest.raises(ValueError, match="at least 5"):
            split_replay_dataset(dataset, root / "out")
        dataset = _dataset(root / "more", sessions=5)
        out = root / "existing"
        out.mkdir()
        (out / "keep").write_text("x")
        with pytest.raises(FileExistsError):
            split_replay_dataset(dataset, out)


def test_heldout_gate_boundaries_are_fail_closed() -> None:
    good = PolicySummary(10, 10, 0, 0, 0.98)
    result = assess_heldout_gate(good, good, validation_baseline_return=.95,
                                 test_baseline_return=.95, validation_oracle_return=1,
                                 test_oracle_return=1)
    assert result["passed"]
    bad = PolicySummary(10, 9, 1, 1, 0.2)
    result = assess_heldout_gate(bad, good, validation_baseline_return=.95,
                                 test_baseline_return=.95, validation_oracle_return=1,
                                 test_oracle_return=1)
    assert not result["passed"]
    assert result["failures"]


def test_split_validator_rejects_manifest_and_data_tampering() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        split_replay_dataset(_dataset(root / "source"), root / "split")
        manifest_path = root / "split" / "split_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["sessions"]["validation"] = manifest["sessions"]["test"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="session manifest"):
            validate_split_manifest(root / "split")

        split_replay_dataset(_dataset(root / "source2"), root / "split2")
        manifest_path = root / "split2" / "split_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_path = root / "split2" / "train.jsonl"
        validation_path = root / "split2" / "validation.jsonl"
        validation_path.write_text(train_path.read_text(encoding="utf-8"), encoding="utf-8")
        manifest["files"]["validation"]["sha256"] = __import__("hashlib").sha256(validation_path.read_bytes()).hexdigest()
        manifest["record_counts"]["validation"] = manifest["record_counts"]["train"]
        manifest["session_counts"]["validation"] = manifest["session_counts"]["train"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="session manifest"):
            validate_split_manifest(root / "split2")


def test_split_validator_rejects_path_escape() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        split_replay_dataset(_dataset(root / "source"), root / "split")
        manifest_path = root / "split" / "split_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["test"]["path"] = "../train.jsonl"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="escapes split directory"):
            validate_split_manifest(root / "split")


def test_pipeline_fail_closed_without_sb3(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = _dataset(root / "source")
        nonempty = root / "nonempty"
        nonempty.mkdir()
        marker = nonempty / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with pytest.raises(SystemExit):
            rl1_pipeline.main(["--dataset", str(dataset), "--output-dir", str(nonempty),
                               "--code-revision", "test"])
        assert marker.read_text(encoding="utf-8") == "keep"

        def fail_step(*args, **kwargs):
            raise RuntimeError("injected missing optional dependency")

        monkeypatch.setattr(rl1_pipeline, "_run", fail_step)
        output = root / "failed"
        assert rl1_pipeline.main(["--dataset", str(dataset), "--output-dir", str(output),
                                  "--code-revision", "test"]) == 1
        report = json.loads((output / "rl1_acceptance.json").read_text(encoding="utf-8"))
        assert report["passed"] is False
        assert report["next_allowed_mode"] == "heuristic"
        assert report["rl_guarded_authorized"] is False
