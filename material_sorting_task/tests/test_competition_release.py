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
    assert values["MATERIAL_SCHEDULER_POLICY"] == "heuristic"
    assert values["MATERIAL_MEASURED_CARRY_GUARD"] == "0"
    assert values["MATERIAL_ACCEPTANCE_BASE_COMMIT"] == "1549df9"


def test_control_script_has_safe_explicit_rollback() -> None:
    script = (ROOT / "scripts" / "competitionctl.sh").read_text(encoding="utf-8")
    assert "eval " not in script
    assert "rollback)" in script
    assert "run_client \"$2\" legacy" in script
    assert "MATERIAL_SCHEDULER_ENGINE=\"$engine\"" in script


def test_deploy_requires_frozen_prefix_and_empty_target() -> None:
    script = (ROOT / "scripts" / "deploy_competition_release.sh").read_text(
        encoding="utf-8"
    )
    assert 'path.parts[0] != "SIX-ANGELS"' in script
    assert "target must be empty" in script
    assert "--strip-components=1" in script


def test_freeze_ignores_checkout_only_cross_platform_differences() -> None:
    script = (ROOT / "scripts" / "freeze_competition_release.sh").read_text(
        encoding="utf-8"
    )
    assert "core.fileMode=false" in script
    assert "core.autocrlf=true" in script
