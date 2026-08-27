"""Deterministic, session-isolated splits for replay training data.

The split boundary is intentionally at ``(source_sha256, session_index)``.  A
decision is never allowed to move independently of the session that produced
it, preventing a single robot run from leaking into held-out evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .replay_env import load_replay_dataset


SPLIT_SCHEMA_VERSION = "scheduler-replay-session-split-v1"
SPLIT_NAMES = ("train", "validation", "test")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            value = dict(record)
            if hasattr(value.get("observation"), "tolist"):
                value["observation"] = value["observation"].tolist()
            if hasattr(value.get("action_mask"), "tolist"):
                value["action_mask"] = value["action_mask"].tolist()
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _session_key(record: Mapping[str, Any]) -> tuple[str, int]:
    source = record.get("source_sha256")
    try:
        index = int(record.get("session_index"))
    except (TypeError, ValueError):
        raise ValueError("record has invalid session_index")
    if not isinstance(source, str) or not _SHA256_RE.fullmatch(source.lower()):
        raise ValueError("record has invalid source_sha256")
    return source.lower(), index


@dataclass(frozen=True)
class SplitResult:
    manifest: dict[str, Any]
    records: dict[str, tuple[dict[str, Any], ...]]


def split_replay_dataset(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    training_only_dataset_path: str | Path | None = None,
    seed: int = 0,
    train_sessions: int = 3,
    validation_sessions: int = 1,
    test_sessions: int = 1,
) -> SplitResult:
    """Validate and split a replay dataset, failing before writing on errors."""

    source = Path(dataset_path)
    target = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"output directory is non-empty: {target}")
    counts = (train_sessions, validation_sessions, test_sessions)
    if any(isinstance(value, bool) or int(value) <= 0 for value in counts):
        raise ValueError("each split must contain at least one session")
    try:
        records = tuple(dict(item) for item in load_replay_dataset(source))
        training_only_records = (
            tuple(
                dict(item)
                for item in load_replay_dataset(training_only_dataset_path)
            )
            if training_only_dataset_path is not None
            else ()
        )
    except Exception:
        # Do not create a directory or manifest for malformed input.
        raise
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    seen_ids: set[tuple[tuple[str, int], str]] = set()
    for record in records:
        key = _session_key(record)
        decision = record.get("decision_id")
        if not isinstance(decision, str) or not decision:
            raise ValueError("record has missing decision_id")
        if (key, decision) in seen_ids:
            raise ValueError(f"duplicate decision_id in session {key}")
        seen_ids.add((key, decision))
        groups.setdefault(key, []).append(record)
    training_only_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in training_only_records:
        key = _session_key(record)
        if key in groups:
            raise ValueError(
                f"training-only session overlaps primary dataset: {key}"
            )
        decision = record.get("decision_id")
        if not isinstance(decision, str) or not decision:
            raise ValueError("training-only record has missing decision_id")
        if (key, decision) in seen_ids:
            raise ValueError(f"duplicate decision_id in session {key}")
        seen_ids.add((key, decision))
        training_only_groups.setdefault(key, []).append(record)
    if training_only_records:
        primary_schema = (
            records[0].get("dataset_schema_version"),
            records[0].get("observation_schema_hash"),
            records[0].get("max_candidates"),
        )
        training_only_schema = (
            training_only_records[0].get("dataset_schema_version"),
            training_only_records[0].get("observation_schema_hash"),
            training_only_records[0].get("max_candidates"),
        )
        if training_only_schema != primary_schema:
            raise ValueError("training-only dataset schema mismatch")
    required = sum(counts)
    if len(groups) < required:
        raise ValueError(f"at least {required} sessions are required; got {len(groups)}")
    ordered = sorted(
        groups,
        key=lambda key: hashlib.sha256(
            f"{int(seed)}|{key[0]}|{key[1]}".encode("utf-8")
        ).hexdigest(),
    )
    validation_end = validation_sessions
    test_end = validation_end + test_sessions
    assignment = {
        "train": ordered[test_end:] + sorted(training_only_groups),
        "validation": ordered[:validation_end],
        "test": ordered[validation_end:test_end],
    }
    if any(not assignment[name] for name in SPLIT_NAMES):
        raise ValueError("a split is empty")
    output_records = {
        name: tuple(
            item
            for key in assignment[name]
            for item in (
                groups[key] if key in groups else training_only_groups[key]
            )
        )
        for name in SPLIT_NAMES
    }
    if sum(len(value) for value in output_records.values()) != (
        len(records) + len(training_only_records)
    ):
        raise ValueError("split record count does not match input")
    manifest = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "dataset_schema_version": str(records[0].get("dataset_schema_version")),
        "input": {"path": str(source), "sha256": _sha256(source), "record_count": len(records)},
        "training_only_input": (
            {
                "path": str(Path(training_only_dataset_path)),
                "sha256": _sha256(Path(training_only_dataset_path)),
                "record_count": len(training_only_records),
                "session_count": len(training_only_groups),
            }
            if training_only_dataset_path is not None
            else None
        ),
        "seed": int(seed),
        "requested_session_counts": {
            "train": int(train_sessions),
            "validation": int(validation_sessions),
            "test": int(test_sessions),
        },
        "session_counts": {name: len(assignment[name]) for name in SPLIT_NAMES},
        "record_counts": {name: len(output_records[name]) for name in SPLIT_NAMES},
        "sessions": {
            name: [{"source_sha256": key[0], "session_index": key[1]} for key in assignment[name]]
            for name in SPLIT_NAMES
        },
        "files": {name: f"{name}.jsonl" for name in SPLIT_NAMES},
    }
    target.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        _write_jsonl(target / f"{name}.jsonl", output_records[name])
        manifest["files"][name] = {"path": f"{name}.jsonl", "sha256": _sha256(target / f"{name}.jsonl")}
    (target / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return SplitResult(manifest=manifest, records=output_records)


__all__ = ["SPLIT_NAMES", "SPLIT_SCHEMA_VERSION", "SplitResult", "split_replay_dataset"]
