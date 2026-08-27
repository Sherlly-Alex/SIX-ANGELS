"""Fail-closed approval artifact for enabling guarded RL scheduling.

The benchmark and shadow validators deliberately produce independent reports.
This module binds those reports, the model package and the observation schema
into one immutable manifest.  The production Client accepts ``rl_guarded``
only when the manifest file hash is supplied explicitly and every bound value
still matches the deployed model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .model_package import validate_model_package


APPROVAL_SCHEMA_VERSION = "scheduler-policy-approval-v1"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} root must be a JSON object")
    return value


def _passed_report(report: Mapping[str, Any], name: str) -> None:
    if report.get("passed") is not True or report.get("failures") not in ([], ()):  # noqa: E501
        raise ValueError(f"{name} report did not pass")


def build_guarded_approval(
    model_path: str | Path,
    benchmark_report_path: str | Path,
    shadow_report_path: str | Path,
    *,
    minimum_blind_seeds: int = 100,
    minimum_shadow_suggestions: int = 1000,
) -> dict[str, Any]:
    """Build a reproducible approval manifest from already-passed gates."""

    if minimum_blind_seeds <= 0 or minimum_shadow_suggestions <= 0:
        raise ValueError("approval evidence minimums must be positive")
    model = Path(model_path)
    benchmark_path = Path(benchmark_report_path)
    shadow_path = Path(shadow_report_path)
    package = validate_model_package(model)
    if not package.passed or package.model_sha256 is None:
        raise ValueError("model package did not pass integrity validation")
    benchmark = _load_object(benchmark_path)
    shadow = _load_object(shadow_path)
    _passed_report(benchmark, "benchmark")
    _passed_report(shadow, "shadow")

    model_hash = package.model_sha256.lower()
    if str(benchmark.get("model_sha256", "")).lower() != model_hash:
        raise ValueError("benchmark model SHA256 does not match the package")
    shadow_hashes = shadow.get("model_sha256")
    if not isinstance(shadow_hashes, list) or [str(v).lower() for v in shadow_hashes] != [model_hash]:
        raise ValueError("shadow model SHA256 does not match the package")
    seeds = benchmark.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < minimum_blind_seeds:
        raise ValueError("benchmark has insufficient blind seeds")
    if len({int(seed) for seed in seeds}) != len(seeds):
        raise ValueError("benchmark blind seeds are not unique")
    try:
        benchmark_limits = benchmark["limits"]
        shadow_limits = shadow["limits"]
        if not isinstance(benchmark_limits, Mapping) or not isinstance(shadow_limits, Mapping):
            raise TypeError("gate limits are not objects")
        benchmark_inference_limit = float(benchmark_limits["max_inference_p95_ms"])
        benchmark_improvement_limit = float(
            benchmark_limits["minimum_relative_improvement"]
        )
        shadow_inference_limit = float(shadow_limits["max_inference_p95_ms"])
        shadow_fallback_limit = float(
            shadow_limits.get(
                "max_runtime_fallback_rate",
                shadow_limits.get("max_fallback_rate"),
            )
        )
        observed_benchmark_inference = float(benchmark["rl_inference_p95_ms"])
        observed_shadow_inference = float(shadow["inference_p95_ms"])
        observed_shadow_fallback = float(shadow["fallback_rate"])
        finite_values = (
            benchmark_inference_limit,
            benchmark_improvement_limit,
            shadow_inference_limit,
            shadow_fallback_limit,
            observed_benchmark_inference,
            observed_shadow_inference,
            observed_shadow_fallback,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("gate limits or observations are non-finite")
        if benchmark_inference_limit > 25.0:
            raise ValueError("benchmark inference limit is weaker than 25 ms")
        if benchmark_improvement_limit < 0.02:
            raise ValueError("benchmark improvement limit is weaker than 2 percent")
        if int(benchmark_limits["bootstrap_samples"]) < 2000:
            raise ValueError("benchmark bootstrap sample count is below 2000")
        if shadow_inference_limit > 25.0:
            raise ValueError("shadow inference limit is weaker than 25 ms")
        if shadow_fallback_limit > 0.01:
            raise ValueError("shadow fallback limit is weaker than 1 percent")
        if int(shadow_limits["min_suggestions"]) < minimum_shadow_suggestions:
            raise ValueError("shadow configured suggestion minimum is too low")
        if observed_benchmark_inference > 25.0 or observed_shadow_inference > 25.0:
            raise ValueError("observed inference p95 exceeds 25 ms")
        if observed_shadow_fallback > 0.01:
            raise ValueError("observed shadow fallback rate exceeds 1 percent")
        suggestions = int(shadow.get("suggestion_count", 0))
        takeovers = int(shadow.get("actual_rl_takeovers", -1))
        masked = int(shadow.get("masked_suggestion_violations", -1))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"approval evidence is malformed or unsafe: {exc}"
        ) from exc
    if suggestions < minimum_shadow_suggestions:
        raise ValueError("shadow report has insufficient accepted suggestions")
    if takeovers != 0 or masked != 0:
        raise ValueError("shadow report contains takeover or mask violations")

    return {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "model_sha256": model_hash,
        "observation_schema_hash": package.observation_schema_hash,
        "training_config_sha256": package.training_config_sha256,
        "benchmark": {
            "report_sha256": file_sha256(benchmark_path),
            "blind_seed_count": len(seeds),
            "rl_inference_p95_ms": benchmark.get("rl_inference_p95_ms"),
        },
        "shadow": {
            "report_sha256": file_sha256(shadow_path),
            "session_count": shadow.get("shadow_sessions"),
            "suggestion_count": suggestions,
            "fallback_rate": shadow.get("fallback_rate"),
            "inference_p95_ms": shadow.get("inference_p95_ms"),
        },
    }


@dataclass(frozen=True)
class GuardedApprovalReport:
    manifest_path: str
    manifest_sha256: str | None
    model_sha256: str | None
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passed"] = self.passed
        value["failures"] = list(self.failures)
        return value


def validate_guarded_approval(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    model_path: str | Path,
    expected_model_sha256: str,
    expected_schema_hash: str,
) -> GuardedApprovalReport:
    """Validate the explicit operator-approved manifest and deployed model."""

    path = Path(manifest_path)
    failures: list[str] = []
    for value, name in (
        (expected_manifest_sha256, "expected manifest SHA256"),
        (expected_model_sha256, "expected model SHA256"),
        (expected_schema_hash, "expected observation schema SHA256"),
    ):
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            failures.append(f"{name} is missing or malformed")
    actual_manifest_hash = None
    manifest: Mapping[str, Any] = {}
    if not path.is_file():
        failures.append("approval manifest is missing")
    else:
        try:
            actual_manifest_hash = file_sha256(path)
            manifest = _load_object(path)
        except (OSError, TypeError, ValueError) as exc:
            failures.append(f"approval manifest is unreadable: {type(exc).__name__}")
    if actual_manifest_hash is not None and _SHA256_RE.fullmatch(expected_manifest_sha256 or ""):
        if actual_manifest_hash != expected_manifest_sha256.lower():
            failures.append("approval manifest SHA256 mismatch")
    actual_model_hash = None
    model = Path(model_path)
    if not model.is_file():
        failures.append("approved model file is missing")
    else:
        try:
            actual_model_hash = file_sha256(model)
        except OSError as exc:
            failures.append(f"approved model is unreadable: {type(exc).__name__}")
        package = validate_model_package(
            model,
            expected_model_sha256=(
                expected_model_sha256
                if _SHA256_RE.fullmatch(expected_model_sha256 or "")
                else None
            ),
        )
        failures.extend(f"model package: {item}" for item in package.failures)
    if manifest:
        if manifest.get("schema_version") != APPROVAL_SCHEMA_VERSION:
            failures.append("approval manifest schema version mismatch")
        if str(manifest.get("model_sha256", "")).lower() != str(expected_model_sha256).lower():
            failures.append("approval manifest model SHA256 mismatch")
        if manifest.get("observation_schema_hash") != expected_schema_hash:
            failures.append("approval manifest observation schema mismatch")
        if model.is_file() and package.passed:
            if package.observation_schema_hash != manifest.get("observation_schema_hash"):
                failures.append("model package observation schema differs from approval")
            if package.training_config_sha256 != manifest.get("training_config_sha256"):
                failures.append("model package training config differs from approval")
    if actual_model_hash is not None and actual_model_hash != str(expected_model_sha256).lower():
        failures.append("deployed model SHA256 mismatch")
    return GuardedApprovalReport(
        manifest_path=str(path),
        manifest_sha256=actual_manifest_hash,
        model_sha256=actual_model_hash,
        failures=tuple(dict.fromkeys(failures)),
    )


__all__ = [
    "APPROVAL_SCHEMA_VERSION",
    "GuardedApprovalReport",
    "build_guarded_approval",
    "file_sha256",
    "validate_guarded_approval",
]
