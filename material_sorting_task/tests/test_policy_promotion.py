from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from learning.observation import OBSERVATION_SCHEMA_VERSION, observation_schema_hash
from learning.promotion import (
    APPROVAL_SCHEMA_VERSION,
    build_guarded_approval,
    file_sha256,
    validate_guarded_approval,
)
from learning.train_maskable_ppo import MODEL_METADATA_SCHEMA_VERSION


def _canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evidence(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    model = tmp_path / "scheduler.zip"
    model.write_bytes(b"guarded-policy")
    model_hash = file_sha256(model)
    config = {
        "seed": 7,
        "total_timesteps": 1000,
        "learning_rate": 0.0003,
        "n_steps": 256,
        "batch_size": 64,
        "max_candidates": 8,
        "environment_factory": "learning.simulation_backend:build_project_sim_env",
        "code_revision": "deadbeef",
    }
    metadata = {
        "metadata_schema_version": MODEL_METADATA_SCHEMA_VERSION,
        "algorithm": "MaskablePPO",
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_schema_hash": observation_schema_hash(8),
        "model_sha256": model_hash,
        "training_config": config,
        "training_config_sha256": _canonical_hash(config),
        "provenance_files": [
            {"name": "events.jsonl", "sha256": "a" * 64, "size_bytes": 10}
        ],
    }
    Path(f"{model}.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "passed": True,
                "failures": [],
                "model_sha256": model_hash,
                "seeds": list(range(10000, 10100)),
                "rl_inference_p95_ms": 4.0,
                "limits": {
                    "max_inference_p95_ms": 25.0,
                    "minimum_relative_improvement": 0.02,
                    "bootstrap_samples": 2000,
                },
            }
        ),
        encoding="utf-8",
    )
    shadow = tmp_path / "shadow.json"
    shadow.write_text(
        json.dumps(
            {
                "passed": True,
                "failures": [],
                "model_sha256": [model_hash],
                "shadow_sessions": 5,
                "suggestion_count": 1200,
                "actual_rl_takeovers": 0,
                "masked_suggestion_violations": 0,
                "fallback_rate": 0.0,
                "inference_p95_ms": 5.0,
                "limits": {
                    "min_suggestions": 1000,
                    "max_inference_p95_ms": 25.0,
                    "max_fallback_rate": 0.01,
                },
            }
        ),
        encoding="utf-8",
    )
    return model, benchmark, shadow, model_hash


def test_build_and_validate_guarded_approval(tmp_path: Path) -> None:
    model, benchmark, shadow, model_hash = _evidence(tmp_path)
    manifest = build_guarded_approval(model, benchmark, shadow)
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    report = validate_guarded_approval(
        approval,
        expected_manifest_sha256=file_sha256(approval),
        model_path=model,
        expected_model_sha256=model_hash,
        expected_schema_hash=observation_schema_hash(8),
    )

    assert manifest["schema_version"] == APPROVAL_SCHEMA_VERSION
    assert manifest["benchmark"]["blind_seed_count"] == 100
    assert report.passed


def test_build_rejects_lax_or_mismatched_gate(tmp_path: Path) -> None:
    model, benchmark, shadow, _ = _evidence(tmp_path)
    payload = json.loads(shadow.read_text(encoding="utf-8"))
    payload["limits"]["max_fallback_rate"] = 0.5
    shadow.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="weaker than 1 percent"):
        build_guarded_approval(model, benchmark, shadow)

    payload["limits"]["max_fallback_rate"] = 0.01
    payload["inference_p95_ms"] = float("nan")
    shadow.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        build_guarded_approval(model, benchmark, shadow)


def test_runtime_validation_rejects_tampering_and_missing_hashes(tmp_path: Path) -> None:
    model, benchmark, shadow, model_hash = _evidence(tmp_path)
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(build_guarded_approval(model, benchmark, shadow)),
        encoding="utf-8",
    )
    approved_hash = file_sha256(approval)
    approval.write_text(approval.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    tampered = validate_guarded_approval(
        approval,
        expected_manifest_sha256=approved_hash,
        model_path=model,
        expected_model_sha256=model_hash,
        expected_schema_hash=observation_schema_hash(8),
    )
    missing = validate_guarded_approval(
        tmp_path / "missing.json",
        expected_manifest_sha256="",
        model_path=model,
        expected_model_sha256="",
        expected_schema_hash=observation_schema_hash(8),
    )

    assert not tampered.passed
    assert "approval manifest SHA256 mismatch" in tampered.failures
    assert not missing.passed
    assert "approval manifest is missing" in missing.failures
    assert "expected manifest SHA256 is missing or malformed" in missing.failures


def test_runtime_validation_rechecks_model_metadata_chain(tmp_path: Path) -> None:
    model, benchmark, shadow, model_hash = _evidence(tmp_path)
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(build_guarded_approval(model, benchmark, shadow)),
        encoding="utf-8",
    )
    metadata_path = Path(f"{model}.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_config"]["seed"] = 999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = validate_guarded_approval(
        approval,
        expected_manifest_sha256=file_sha256(approval),
        model_path=model,
        expected_model_sha256=model_hash,
        expected_schema_hash=observation_schema_hash(8),
    )

    assert not report.passed
    assert "model package: training_config SHA256 mismatch" in report.failures
