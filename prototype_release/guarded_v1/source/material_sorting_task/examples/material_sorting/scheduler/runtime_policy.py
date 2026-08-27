"""Resolve the optional RL overlay without dismantling V2 Heuristic control.

``MATERIAL_SCHEDULER_RL_ENABLED`` defaults to off.  A missing model, broken
approval chain or warmup failure must keep the already-accepted Heuristic
decision service and candidate provider alive.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


ALLOWED_POLICIES = frozenset({"heuristic", "rl_shadow", "rl_guarded"})


def parse_rl_enabled(value: str | None) -> bool:
    """Return True only for an explicit ``1``. Unset or any other value is off."""

    return str(value or "0").strip() == "1"


def requested_policy(value: str | None) -> tuple[str, str | None]:
    """Return ``(normalized_policy, invalid_original_or_none)``."""

    raw = str(value or "heuristic").strip().lower()
    if raw not in ALLOWED_POLICIES:
        return "heuristic", raw
    return raw, None


def resolve_effective_policy(*, rl_enabled: bool, requested: str) -> str:
    policy, _ = requested_policy(requested)
    if not rl_enabled or policy == "heuristic":
        return "heuristic"
    return policy


@dataclass
class SchedulerStack:
    """Live V2 scheduler objects. ``decision_service`` is never None on success."""

    scheduler_policy: str
    rl_enabled: bool
    decision_service: Any
    candidate_provider: Any
    rl_policy: Any = None
    isolated_policy_supervisor: Any = None
    rl_load_error: str | None = None

    @property
    def loads_model(self) -> bool:
        return self.rl_policy is not None or self.isolated_policy_supervisor is not None


def _env(environ: Mapping[str, str], name: str, default: str = "") -> str:
    return str(environ.get(name, default)).strip()


def _load_optional_rl_policy(
    *,
    environ: Mapping[str, str],
    effective_policy: str,
) -> tuple[Any, str | None]:
    """Load and warm a model. Any failure returns ``(None, error)``."""

    if effective_policy == "heuristic":
        return None, None
    from learning.observation import ObservationBuilder
    from scheduler.policies import RLPolicy

    model_path = _env(environ, "MATERIAL_SCHEDULER_MODEL")
    expected_hash = _env(environ, "MATERIAL_SCHEDULER_MODEL_SHA256")
    builder = ObservationBuilder(8)
    if effective_policy == "rl_guarded":
        from learning.promotion import validate_guarded_approval

        approval = validate_guarded_approval(
            _env(environ, "MATERIAL_RL_GUARDED_APPROVAL"),
            expected_manifest_sha256=_env(
                environ, "MATERIAL_RL_GUARDED_APPROVAL_SHA256"
            ),
            model_path=model_path,
            expected_model_sha256=expected_hash,
            expected_schema_hash=builder.schema_hash,
        )
        if not approval.passed:
            return None, "rl_guarded approval rejected: " + "; ".join(
                approval.failures
            )
    device = _env(environ, "MATERIAL_RL_DEVICE", "cpu") or "cpu"
    policy = RLPolicy(
        model_path=model_path or None,
        expected_sha256=expected_hash or None,
        expected_schema_hash=builder.schema_hash,
        device=device,
    )
    try:
        policy.warmup(
            observation_size=builder.size,
            action_count=builder.max_candidates,
        )
    except Exception as exc:
        return None, f"scheduler RL model unavailable ({type(exc).__name__}: {exc})"
    return policy, None


def _build_isolated_shadow_supervisor(
    *, environ: Mapping[str, str]
) -> tuple[Any, str | None]:
    """Validate immutable model provenance without loading it in the client."""

    from learning.isolated_inference import (
        InferenceSupervisor,
        InferenceSupervisorConfig,
        PolicyWorkerConfig,
        ProcessPolicyTransport,
    )
    from learning.observation import ObservationBuilder

    model_path = Path(_env(environ, "MATERIAL_SCHEDULER_MODEL"))
    expected_hash = _env(environ, "MATERIAL_SCHEDULER_MODEL_SHA256")
    if not model_path.is_file() or not expected_hash:
        return None, "isolated Shadow model path or SHA256 is missing"
    try:
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest.casefold() != expected_hash.casefold():
            return None, "isolated Shadow model SHA256 mismatch"
        metadata_path = Path(f"{model_path}.metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        builder = ObservationBuilder(8)
        schema = builder.schema_hash
        if (
            metadata.get("metadata_schema_version") != "scheduler-model-metadata-v1"
            or metadata.get("algorithm") != "MaskablePPO"
            or metadata.get("model_sha256") != digest
            or metadata.get("observation_schema_hash") != schema
        ):
            return None, "isolated Shadow model metadata is not approved"
        timeout_ms = float(_env(environ, "MATERIAL_RL_TIMEOUT_MS", "50") or "50")
        stale_ms = float(_env(environ, "MATERIAL_RL_RESULT_STALE_MS", "1000") or "1000")
        faults = int(
            _env(environ, "MATERIAL_RL_QUARANTINE_AFTER_TIMEOUTS", "3") or "3"
        )
        transport = ProcessPolicyTransport(
            PolicyWorkerConfig(
                model_path=str(model_path),
                expected_sha256=digest,
                expected_schema_hash=schema,
                device=_env(environ, "MATERIAL_RL_DEVICE", "cpu") or "cpu",
                observation_size=builder.size,
                action_count=builder.max_candidates,
            ),
            start_method="spawn",
        )
        if not transport.start():
            return None, "isolated Shadow worker could not start: " + str(
                transport.startup_error or "unknown error"
            )
        return (
            InferenceSupervisor(
                transport,
                config=InferenceSupervisorConfig(
                    # PolicyGuard enforces the inference performance budget.
                    # Isolated results are collected on a later control tick,
                    # so a 50 ms delivery deadline would turn normal async
                    # scheduling into a false worker fault. Delivery never
                    # blocks control; wait boundedly, then fail closed.
                    deadline_s=max(0.5, (stale_ms / 1000.0) * 2.0),
                    stale_after_s=stale_ms / 1000.0,
                    isolate_after_consecutive_faults=faults,
                ),
            ),
            None,
        )
    except Exception as exc:
        return None, f"isolated Shadow unavailable ({type(exc).__name__}: {exc})"


def build_v2_scheduler_stack(
    *,
    environ: Mapping[str, str] | None = None,
    event_log: Any = None,
) -> SchedulerStack:
    """Build V2 Heuristic first; attach RL only after the overlay is valid."""

    environ = os.environ if environ is None else environ
    from navigation.costmap import WorldCostmap
    from scheduler.decision import DecisionConfig, SchedulerDecisionService
    from scheduler.policies import PolicyGuard, PolicyGuardConfig
    from scheduler.project_candidates import ProjectCandidateProvider

    rl_enabled = parse_rl_enabled(_env(environ, "MATERIAL_SCHEDULER_RL_ENABLED", "0"))
    requested, _invalid = requested_policy(
        _env(environ, "MATERIAL_SCHEDULER_POLICY", "heuristic")
    )
    effective = resolve_effective_policy(rl_enabled=rl_enabled, requested=requested)
    isolated_shadow = (
        effective == "rl_shadow"
        and parse_rl_enabled(_env(environ, "MATERIAL_RL_SHADOW_ISOLATED", "0"))
    )

    switch_margin = float(_env(environ, "MATERIAL_POLICY_SWITCH_MARGIN", "0.25") or "0.25")
    minimum_hold_s = float(_env(environ, "MATERIAL_POLICY_MIN_HOLD_S", "0.75") or "0.75")
    dynamic_ttl_s = float(_env(environ, "MATERIAL_COSTMAP_DYNAMIC_TTL_S", "1.0") or "1.0")
    rl_timeout_ms = float(_env(environ, "MATERIAL_RL_TIMEOUT_MS", "50") or "50")
    if rl_timeout_ms <= 0.0:
        raise ValueError("MATERIAL_RL_TIMEOUT_MS must be positive")
    rl_quarantine_after = int(
        _env(environ, "MATERIAL_RL_QUARANTINE_AFTER_TIMEOUTS", "3") or "3"
    )
    if rl_quarantine_after <= 0:
        raise ValueError("MATERIAL_RL_QUARANTINE_AFTER_TIMEOUTS must be positive")

    candidate_provider = ProjectCandidateProvider(
        costmap=WorldCostmap(dynamic_ttl_s=dynamic_ttl_s)
    )
    rl_policy = None
    isolated_policy_supervisor = None
    load_error = None
    if isolated_shadow:
        isolated_policy_supervisor, load_error = _build_isolated_shadow_supervisor(
            environ=environ
        )
        if isolated_policy_supervisor is None:
            effective = "heuristic"
    elif effective != "heuristic":
        rl_policy, load_error = _load_optional_rl_policy(
            environ=environ, effective_policy=effective
        )
        if rl_policy is None:
            effective = "heuristic"

    decision_service = SchedulerDecisionService(
        config=DecisionConfig(
            policy_mode=effective,
            minimum_action_hold_s=minimum_hold_s,
            switch_utility_margin=switch_margin,
        ),
        rl_policy=rl_policy,
        isolated_policy_supervisor=isolated_policy_supervisor,
        policy_guard=PolicyGuard(
            PolicyGuardConfig(
                inference_timeout_s=rl_timeout_ms / 1000.0,
                quarantine_after_timeout=effective == "rl_guarded",
                consecutive_timeouts_before_quarantine=rl_quarantine_after,
            )
        ),
        event_log=event_log,
    )
    return SchedulerStack(
        scheduler_policy=effective,
        rl_enabled=rl_enabled and effective != "heuristic",
        decision_service=decision_service,
        candidate_provider=candidate_provider,
        rl_policy=rl_policy,
        isolated_policy_supervisor=isolated_policy_supervisor,
        rl_load_error=load_error,
    )


__all__ = [
    "ALLOWED_POLICIES",
    "SchedulerStack",
    "build_v2_scheduler_stack",
    "parse_rl_enabled",
    "requested_policy",
    "resolve_effective_policy",
]
