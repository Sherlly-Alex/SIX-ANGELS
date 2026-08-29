from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _release_values() -> dict[str, str]:
    values: dict[str, str] = {}
    path = ROOT / "config" / "competition_release.env"
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def test_competition_defaults_match_accepted_runtime() -> None:
    values = _release_values()
    assert values["ROS_DOMAIN_ID"] == "102"
    assert values["MATERIAL_EXECUTION_MODE"] == "task123_full"
    assert values["MATERIAL_SCHEDULER_ENGINE"] == "v2"
    assert values["MATERIAL_SCHEDULER_POLICY"] == "rl_guarded"
    assert values["MATERIAL_MEASURED_CARRY_GUARD"] == "0"
    assert values["MATERIAL_TASK3_LATERAL_GUARD"] == "1"
    assert values["MATERIAL_ACCEPTANCE_BASE_COMMIT"] == "e3f5284"
    # Keep the hard fail-safe deadline above the measured 25 ms promotion
    # gate so a single host scheduling hiccup does not quarantine an otherwise
    # healthy policy for the rest of an official run.
    assert values["MATERIAL_RL_TIMEOUT_MS"] == "50"
    assert values["MATERIAL_RL_QUARANTINE_AFTER_TIMEOUTS"] == "3"
    assert len(values["MATERIAL_RL_MODEL_SHA256"]) == 64
    assert len(values["MATERIAL_RL_APPROVAL_SHA256"]) == 64


def test_control_script_has_safe_explicit_rollback() -> None:
    script = (ROOT / "scripts" / "competitionctl.sh").read_text(encoding="utf-8")
    assert "eval " not in script
    assert "rollback)" in script
    assert "run_client \"$2\" heuristic" in script
    assert "MATERIAL_SCHEDULER_ENGINE=\"$engine\"" in script
    assert "MATERIAL_SCHEDULER_RL_ENABLED=\"$rl_enabled\"" in script
    assert "OMP_NUM_THREADS=1" in script
    assert "verify_guarded_assets" in script
    assert "policy=rl_shadow" in script
    assert 'local mode="${2:-${MATERIAL_DEFAULT_CLIENT_MODE:-heuristic}}"' in script
    release = (ROOT / "config" / "competition_release.env").read_text(encoding="utf-8")
    assert "MATERIAL_DEFAULT_CLIENT_MODE=heuristic" in release
    assert "MATERIAL_ARTIFACT_ROOT" in script
    assert "server-detached" in script
    assert "PYTHONUNBUFFERED=1" in script
    assert "EXTERNAL_RL_MODEL_RELATIVE_PATH" in script
    assert "EXTERNAL_RL_APPROVAL_SHA256" in script


def test_deploy_requires_frozen_prefix_and_empty_target() -> None:
    script = (ROOT / "scripts" / "deploy_competition_release.sh").read_text(
        encoding="utf-8"
    )
    assert 'path.parts[0] != "SIX-ANGELS"' in script
    assert "target must be empty" in script
    assert "--strip-components=1" in script
    assert "RELEASE_ASSETS.sha256" in script
    assert 'sha256sum -c "$(basename -- "$archive").sha256"' in script


def test_freeze_ignores_checkout_only_cross_platform_differences() -> None:
    script = (ROOT / "scripts" / "freeze_competition_release.sh").read_text(
        encoding="utf-8"
    )
    assert "core.fileMode=false" in script
    assert "core.autocrlf=true" in script
    assert "MATERIAL_FREEZE_MODEL_SOURCE" in script
    assert "guarded_policy_acceptance.json" in script
