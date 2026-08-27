from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from learning.model_package import validate_model_package
from learning.observation import OBSERVATION_SCHEMA_VERSION, observation_schema_hash
from learning.train_maskable_ppo import MODEL_METADATA_SCHEMA_VERSION


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def create_package(tmp_path: Path) -> tuple[Path, str, str]:
    model = tmp_path / "scheduler.zip"
    model.write_bytes(b"approved deterministic model bytes")
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    dataset_hash = "b" * 64
    config = {
        "seed": 7,
        "total_timesteps": 1000,
        "learning_rate": 0.0003,
        "n_steps": 256,
        "batch_size": 64,
        "max_candidates": 8,
        "environment_factory": "training_env:build",
        "code_revision": "deadbeef",
    }
    metadata = {
        "metadata_schema_version": MODEL_METADATA_SCHEMA_VERSION,
        "algorithm": "MaskablePPO",
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_schema_hash": observation_schema_hash(8),
        "model_sha256": model_hash,
        "training_config": config,
        "training_config_sha256": canonical_hash(config),
        "provenance_files": [
            {"name": "dataset.jsonl", "sha256": dataset_hash, "size_bytes": 123}
        ],
    }
    Path(f"{model}.metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return model, model_hash, dataset_hash


def test_model_package_accepts_matching_integrity_chain(tmp_path: Path) -> None:
    model, model_hash, dataset_hash = create_package(tmp_path)

    report = validate_model_package(
        model,
        expected_model_sha256=model_hash,
        expected_provenance_sha256=dataset_hash,
    )

    assert report.passed
    assert report.model_sha256 == model_hash
    assert report.provenance_file_count == 1


def test_model_package_rejects_tampered_model(tmp_path: Path) -> None:
    model, model_hash, _ = create_package(tmp_path)
    model.write_bytes(model.read_bytes() + b"tampered")

    report = validate_model_package(model, expected_model_sha256=model_hash)

    assert not report.passed
    assert "model SHA256 disagrees with metadata" in report.failures
    assert "actual model SHA256 does not match approved SHA256" in report.failures


def test_model_package_rejects_tampered_training_config(tmp_path: Path) -> None:
    model, _, _ = create_package(tmp_path)
    metadata_path = Path(f"{model}.metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["training_config"]["seed"] = 999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = validate_model_package(model)

    assert not report.passed
    assert "training_config SHA256 mismatch" in report.failures


def test_training_writes_model_config_and_provenance_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    training = importlib.import_module("learning.train_maskable_ppo")
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"sample":1}\n', encoding="utf-8")

    constructor_kwargs = {}

    class FakeModel:
        def __init__(self, policy, env, **kwargs):
            del policy, env
            constructor_kwargs.update(kwargs)

        def learn(self, **kwargs):
            del kwargs

        def save(self, path):
            Path(path).write_bytes(b"trained-model")

    monkeypatch.setattr(training, "_maskable_ppo_class", lambda: FakeModel)
    env = SimpleNamespace(
        action_space=SimpleNamespace(n=8),
        action_masks=lambda: [True] * 8,
    )
    output = tmp_path / "trained.zip"

    result = training.train_maskable_ppo(
        env,
        output,
        total_timesteps=10,
        provenance_files=[dataset],
        environment_factory="training_env:build",
        code_revision="deadbeef",
        gamma=0.0,
        gae_lambda=1.0,
        device="cpu",
        verbose=0,
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_sha256"] == result.model_sha256
    assert metadata["training_config_sha256"] == result.training_config_sha256
    assert metadata["training_config"]["environment_factory"] == "training_env:build"
    assert metadata["training_config"]["code_revision"] == "deadbeef"
    assert metadata["training_config"]["gamma"] == 0.0
    assert metadata["training_config"]["gae_lambda"] == 1.0
    assert metadata["training_config"]["device"] == "cpu"
    assert constructor_kwargs["gamma"] == 0.0
    assert constructor_kwargs["gae_lambda"] == 1.0
    assert constructor_kwargs["device"] == "cpu"
    assert metadata["provenance_files"][0]["sha256"] == hashlib.sha256(
        dataset.read_bytes()
    ).hexdigest()
    assert validate_model_package(output).passed


@pytest.mark.parametrize(
    ("name", "value"),
    (("gamma", -0.01), ("gamma", 1.01), ("gae_lambda", -0.01), ("gae_lambda", 1.01)),
)
def test_training_rejects_invalid_discount_parameters(
    tmp_path: Path, name: str, value: float
) -> None:
    training = importlib.import_module("learning.train_maskable_ppo")
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    env = SimpleNamespace(
        action_space=SimpleNamespace(n=8),
        action_masks=lambda: [True] * 8,
    )
    with pytest.raises(ValueError, match=name):
        training.train_maskable_ppo(
            env,
            tmp_path / "model.zip",
            provenance_files=[dataset],
            code_revision="deadbeef",
            **{name: value},
        )
