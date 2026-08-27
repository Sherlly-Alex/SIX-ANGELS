#!/usr/bin/env python3
"""Fail fast when required Client source files are missing."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "examples" / "material_sorting"

REQUIRED = (
    ROOT / "config" / "competition_release.env",
    TASK / "client_task.py",
    TASK / "competition_controller.py",
    TASK / "control_types.py",
    TASK / "instruction_parser.py",
    TASK / "runtime_health.py",
    TASK / "task_orchestration.py",
    TASK / "scheduler" / "models.py",
    TASK / "scheduler" / "events.py",
    TASK / "scheduler" / "decision.py",
    TASK / "scheduler" / "engine.py",
    TASK / "learning" / "event_replay.py",
    TASK / "learning" / "benchmark.py",
    TASK / "learning" / "model_package.py",
    TASK / "learning" / "promotion.py",
    TASK / "learning" / "replay_env.py",
    TASK / "learning" / "configs" / "replay_training_v1.json",
    TASK / "learning" / "configs" / "replay_training_contextual_success.json",
    TASK / "learning" / "configs" / "replay_training_contextual_costaware.json",
    TASK / "learning" / "simulation_backend.py",
    TASK / "learning" / "sim_collect.py",
    TASK / "learning" / "coverage_audit.py",
    TASK / "learning" / "rl2_pipeline.py",
    TASK / "learning" / "configs" / "project_simulation_v1.json",
    TASK / "learning" / "configs" / "project_simulation_v2.json",
    TASK / "learning" / "configs" / "project_simulation_v3.json",
    TASK / "learning" / "shadow_gate.py",
    TASK / "executors" / "__init__.py",
    TASK / "executors" / "base.py",
    TASK / "executors" / "dry_run.py",
    TASK / "executors" / "task1.py",
    TASK / "executors" / "task2.py",
    TASK / "executors" / "task3.py",
    TASK / "material_competition_layout.json",
    TASK / "mjcf" / "material_competition.xml",
    TASK / "navigation" / "navigation_controller.py",
    TASK / "navigation" / "competition_adapter.py",
    TASK / "navigation" / "dynamic_overlay.py",
    TASK / "navigation" / "footprint_checker.py",
    TASK / "navigation" / "path_smoother.py",
    TASK / "navigation" / "robot_geometry.py",
    TASK / "perception" / "box_detect.py",
    TASK / "perception" / "checkpoints" / "best.pt",
    TASK / "desktop_grasp" / "manual_dual_arm_pregrasp.py",
    TASK / "desktop_grasp" / "manual_dual_arm_to_shelf.py",
    TASK / "desktop_grasp" / "semantic_target_locator.py",
    TASK / "desktop_grasp" / "target_metadata.py",
    TASK / "desktop_grasp" / "pregrasp_core.py",
    TASK / "shelf" / "placement_feedback.py",
    ROOT / "scripts" / "run_client.sh",
    ROOT / "scripts" / "competitionctl.sh",
    ROOT / "scripts" / "freeze_competition_release.sh",
    ROOT / "scripts" / "deploy_competition_release.sh",
    ROOT / "scripts" / "run_desktop_grasp.sh",
    ROOT / "scripts" / "replay_scheduler_events.py",
    ROOT / "scripts" / "benchmark_scheduler_policy.py",
    ROOT / "scripts" / "train_scheduler_policy.py",
    ROOT / "scripts" / "validate_rl_shadow.py",
    ROOT / "scripts" / "validate_guarded_lineage.py",
    ROOT / "scripts" / "validate_scheduler_model.py",
    ROOT / "scripts" / "approve_guarded_policy.py",
    ROOT / "scripts" / "validate_guarded_release.py",
    ROOT / "scripts" / "validate_rl_guarded.py",
    ROOT / "scripts" / "split_replay_dataset.py",
    ROOT / "scripts" / "validate_rl1_heldout.py",
    ROOT / "scripts" / "run_rl1_pipeline.py",
    ROOT / "scripts" / "rl2_cli.py",
    ROOT / "scripts" / "rl2_official_runner.py",
    ROOT / "scripts" / "rl2ctl.sh",
    ROOT / "scripts" / "audit_rl2_coverage.py",
    ROOT / "examples" / "material_sorting" / "learning" / "replay_split.py",
    ROOT / "examples" / "material_sorting" / "learning" / "heldout_gate.py",
    ROOT / "scripts" / "validate_runtime_health_run.py",
    ROOT / "docs" / "PROJECT_SIMULATION_BENCHMARK.md",
    ROOT / "docs" / "GUARDED_POLICY_PROMOTION.md",
    ROOT / "docs" / "FINAL_SCHEDULER_REMOTE_ACCEPTANCE.md",
    ROOT / "docs" / "RL_PHASE1.md",
    ROOT / "docs" / "RL2_SUCCESS_FIRST.md",
    ROOT / "docs" / "RL2_REMOTE_EXECUTION.md",
    ROOT / "docs" / "RL2_1_CONTEXTUAL_OUTCOME.md",
)


def main() -> int:
    missing = [path.relative_to(ROOT) for path in REQUIRED if not path.is_file()]
    if missing:
        for path in missing:
            print(f"MISSING {path}")
        return 1

    checkpoint = TASK / "perception" / "checkpoints" / "best.pt"
    if checkpoint.stat().st_size < 1_000_000:
        print(f"INVALID {checkpoint.relative_to(ROOT)}: file is unexpectedly small")
        return 1

    syntax_errors: list[tuple[Path, SyntaxError]] = []
    for path in ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
        except SyntaxError as exc:
            syntax_errors.append((path.relative_to(ROOT), exc))

    if syntax_errors:
        for path, exc in syntax_errors:
            print(f"SYNTAX {path}:{exc.lineno}: {exc.msg}")
        return 1

    print(
        f"workspace OK ({len(REQUIRED)} required files present; "
        f"Python syntax valid)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
