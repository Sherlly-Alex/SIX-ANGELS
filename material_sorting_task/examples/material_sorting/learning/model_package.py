"""Integrity validation for an offline-trained scheduler model package."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .observation import OBSERVATION_SCHEMA_VERSION, observation_schema_hash
from .train_maskable_ppo import MODEL_METADATA_SCHEMA_VERSION


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ModelPackageReport:
    model_path: str
    metadata_path: str
    model_sha256: str | None
    observation_schema_hash: str | None
    training_config_sha256: str | None
    provenance_file_count: int
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passed"] = self.passed
        value["failures"] = list(self.failures)
        return value


def validate_model_package(
    model_path: str | Path,
    *,
    expected_model_sha256: str | None = None,
    expected_provenance_sha256: str | None = None,
) -> ModelPackageReport:
    model = Path(model_path)
    metadata_path = Path(f"{model}.metadata.json")
    failures: list[str] = []
    if expected_model_sha256 is not None and not _SHA256_RE.fullmatch(
        expected_model_sha256
    ):
        raise ValueError("expected_model_sha256 must be a SHA256 hex digest")
    if expected_provenance_sha256 is not None and not _SHA256_RE.fullmatch(
        expected_provenance_sha256
    ):
        raise ValueError("expected_provenance_sha256 must be a SHA256 hex digest")
    actual_model_hash = None
    if not model.is_file():
        failures.append("model file is missing")
    else:
        actual_model_hash = _file_sha256(model)
    metadata: Mapping[str, Any] = {}
    if not metadata_path.is_file():
        failures.append("model metadata file is missing")
    else:
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, Mapping):
                raise TypeError("metadata root is not an object")
            metadata = loaded
        except (OSError, TypeError, ValueError) as exc:
            failures.append(f"model metadata is unreadable: {type(exc).__name__}")

    if metadata:
        if metadata.get("metadata_schema_version") != MODEL_METADATA_SCHEMA_VERSION:
            failures.append("model metadata schema version mismatch")
        if metadata.get("algorithm") != "MaskablePPO":
            failures.append("model algorithm is not MaskablePPO")
        declared_hash = metadata.get("model_sha256")
        if actual_model_hash is not None and declared_hash != actual_model_hash:
            failures.append("model SHA256 disagrees with metadata")
        config = metadata.get("training_config")
        declared_config_hash = metadata.get("training_config_sha256")
        if not isinstance(config, Mapping):
            failures.append("training_config is missing")
        elif declared_config_hash != _canonical_sha256(config):
            failures.append("training_config SHA256 mismatch")
        try:
            maximum = int(config.get("max_candidates")) if isinstance(config, Mapping) else 0
        except (TypeError, ValueError):
            maximum = 0
        expected_schema = observation_schema_hash(maximum) if maximum > 0 else None
        if metadata.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION:
            failures.append("observation schema version mismatch")
        if expected_schema is None or metadata.get("observation_schema_hash") != expected_schema:
            failures.append("observation schema SHA256 mismatch")
        provenance = metadata.get("provenance_files")
        if not isinstance(provenance, list):
            failures.append("provenance_files is missing")
            provenance = []
        for index, item in enumerate(provenance):
            if (
                not isinstance(item, Mapping)
                or not isinstance(item.get("name"), str)
                or not _SHA256_RE.fullmatch(str(item.get("sha256", "")))
                or not isinstance(item.get("size_bytes"), int)
                or int(item.get("size_bytes", -1)) < 0
            ):
                failures.append(f"provenance_files[{index}] is malformed")
        if expected_provenance_sha256 is not None and not any(
            isinstance(item, Mapping)
            and str(item.get("sha256", "")).lower()
            == expected_provenance_sha256.lower()
            for item in provenance
        ):
            failures.append("approved provenance SHA256 is absent")
    else:
        provenance = []

    if expected_model_sha256 is not None and actual_model_hash != expected_model_sha256.lower():
        failures.append("actual model SHA256 does not match approved SHA256")
    return ModelPackageReport(
        model_path=str(model),
        metadata_path=str(metadata_path),
        model_sha256=actual_model_hash,
        observation_schema_hash=(
            str(metadata.get("observation_schema_hash")) if metadata else None
        ),
        training_config_sha256=(
            str(metadata.get("training_config_sha256")) if metadata else None
        ),
        provenance_file_count=len(provenance),
        failures=tuple(dict.fromkeys(failures)),
    )


__all__ = ["ModelPackageReport", "validate_model_package"]
