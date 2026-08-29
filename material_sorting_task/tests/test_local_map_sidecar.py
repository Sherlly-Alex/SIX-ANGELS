"""Tests for optional local-map sidecar and shelf standoff gating."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from executors.base import ExecutionContext
from executors.local_map_motion import (
    MAP_STANDOFF_APPLY_MAX_DIST_M,
    map_standoff_m,
)
from perception.local_map_sidecar import (
    LOCAL_MAP_CLEAR_LINEAR_BOOST,
    LOCAL_MAP_NEAR_HIT_LINEAR_SCALE,
    LOCAL_MAP_MID_HIT_LINEAR_SCALE,
    LocalMapAdvice,
    LocalMapSidecar,
    applied_standoff_m,
    local_map_linear_scale,
)


def test_applied_standoff_respects_flags():
    fb = 0.42
    assert applied_standoff_m(None, fb) == fb
    advice = LocalMapAdvice(
        enabled=True,
        apply=False,
        fresh=True,
        clear=False,
        distance_m=0.5,
        suggested_standoff_m=0.55,
    )
    assert applied_standoff_m(advice, fb) == fb
    advice_on = LocalMapAdvice(
        enabled=True,
        apply=True,
        fresh=True,
        clear=False,
        distance_m=0.5,
        suggested_standoff_m=0.55,
    )
    assert applied_standoff_m(advice_on, fb) == pytest.approx(0.55)
    stale = LocalMapAdvice(
        enabled=True,
        apply=True,
        fresh=False,
        clear=False,
        distance_m=0.5,
        suggested_standoff_m=0.55,
    )
    assert applied_standoff_m(stale, fb) == fb


def test_sidecar_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MATERIAL_LOCAL_MAP", raising=False)
    monkeypatch.delenv("MATERIAL_LOCAL_MAP_APPLY", raising=False)
    side = LocalMapSidecar()
    assert side.enabled is False
    advice = side.on_tick(now_s=1.0, odometry=None, t_cam_world=None)
    assert advice.enabled is False
    assert advice.as_dict()["reason"] == "disabled"


def test_shelf_clearance_helper_noop_without_advice():
    ctx = ExecutionContext(
        now_s=0.0,
        instruction={},
        task_index=0,
        attempt=1,
        local_map_advice=None,
    )
    assert map_standoff_m(ctx, 0.42, max_standoff_m=0.88) == pytest.approx(0.42)


def _near_hit_advice(**overrides):
    base = {
        "enabled": True,
        "apply": True,
        "fresh": True,
        "clear": False,
        "distance_m": 0.70,
        "suggested_standoff_m": 0.60,
    }
    base.update(overrides)
    return base


def test_shelf_clearance_applies_on_near_obstacle_hit():
    ctx = ExecutionContext(
        now_s=0.0,
        instruction={},
        task_index=0,
        attempt=1,
        local_map_advice=_near_hit_advice(),
    )
    # max(0.42, 0.60, 0.70 + 0.10) capped at 0.88
    assert map_standoff_m(ctx, 0.42, max_standoff_m=0.88) == pytest.approx(0.80)


def test_shelf_clearance_keeps_fallback_when_corridor_clear():
    ctx = ExecutionContext(
        now_s=0.0,
        instruction={},
        task_index=0,
        attempt=1,
        local_map_advice=_near_hit_advice(clear=True, suggested_standoff_m=0.90),
    )
    assert map_standoff_m(ctx, 0.75, max_standoff_m=0.88) == pytest.approx(0.75)


def test_shelf_clearance_keeps_fallback_when_hit_is_far():
    ctx = ExecutionContext(
        now_s=0.0,
        instruction={},
        task_index=0,
        attempt=1,
        local_map_advice=_near_hit_advice(
            distance_m=MAP_STANDOFF_APPLY_MAX_DIST_M + 0.05,
        ),
    )
    assert map_standoff_m(ctx, 0.75, max_standoff_m=0.88) == pytest.approx(0.75)


def test_shelf_clearance_never_stands_closer_than_fallback():
    ctx = ExecutionContext(
        now_s=0.0,
        instruction={},
        task_index=0,
        attempt=1,
        local_map_advice=_near_hit_advice(
            distance_m=0.55,
            suggested_standoff_m=0.40,
        ),
    )
    assert map_standoff_m(ctx, 0.75, max_standoff_m=0.88) == pytest.approx(0.75)


def test_shelf_clearance_respects_max_cap():
    ctx = ExecutionContext(
        now_s=0.0,
        instruction={},
        task_index=0,
        attempt=1,
        local_map_advice=_near_hit_advice(distance_m=0.95, suggested_standoff_m=0.90),
    )
    assert map_standoff_m(ctx, 0.75, max_standoff_m=0.88) == pytest.approx(0.88)


def test_shelf_clearance_custom_max_cap():
    ctx = ExecutionContext(
        now_s=0.0,
        instruction={},
        task_index=0,
        attempt=1,
        local_map_advice=_near_hit_advice(distance_m=0.95, suggested_standoff_m=0.90),
    )
    assert map_standoff_m(ctx, 0.75, max_standoff_m=1.08) == pytest.approx(1.05)


def test_task2_shelf_pick_approach_x_defaults_without_map():
    from executors.task2 import Task2IntegratedExecutor

    executor = Task2IntegratedExecutor.__new__(Task2IntegratedExecutor)
    ctx = ExecutionContext(
        now_s=0.0,
        instruction={},
        task_index=1,
        attempt=1,
        local_map_advice=None,
    )
    assert executor._shelf_pick_approach_x(ctx) == pytest.approx(-1.50)


def test_task2_shelf_pick_approach_x_retreats_on_near_hit():
    from executors.task2 import Task2IntegratedExecutor

    executor = Task2IntegratedExecutor.__new__(Task2IntegratedExecutor)
    ctx = ExecutionContext(
        now_s=0.0,
        instruction={},
        task_index=1,
        attempt=1,
        local_map_advice=_near_hit_advice(distance_m=0.99, suggested_standoff_m=0.90),
    )
    # shelf_front (-2.465) + clearance capped at 1.08 -> -1.385
    assert executor._shelf_pick_approach_x(ctx) == pytest.approx(-1.385)


def test_task2_table_entry_margin_defaults_without_map():
    from executors.task2 import Task2IntegratedExecutor

    executor = Task2IntegratedExecutor.__new__(Task2IntegratedExecutor)
    ctx = ExecutionContext(
        now_s=0.0,
        instruction={},
        task_index=1,
        attempt=1,
        local_map_advice=None,
    )
    assert executor._table_entry_margin_m(ctx) == pytest.approx(0.25)


def test_local_map_linear_scale_boosts_when_clear():
    advice = _near_hit_advice(clear=True)
    assert local_map_linear_scale(advice) == pytest.approx(LOCAL_MAP_CLEAR_LINEAR_BOOST)


def test_local_map_linear_scale_slows_on_near_hit(monkeypatch):
    monkeypatch.setenv("MATERIAL_LOCAL_MAP_SPEED_MODE", "full")
    assert local_map_linear_scale(_near_hit_advice(distance_m=0.30)) == pytest.approx(
        LOCAL_MAP_NEAR_HIT_LINEAR_SCALE
    )


def test_local_map_linear_scale_mild_slow_on_mid_hit(monkeypatch):
    monkeypatch.setenv("MATERIAL_LOCAL_MAP_SPEED_MODE", "full")
    assert local_map_linear_scale(_near_hit_advice(distance_m=0.70)) == pytest.approx(
        LOCAL_MAP_MID_HIT_LINEAR_SCALE
    )


def test_local_map_linear_scale_boost_only_skips_hit_slowdown(monkeypatch):
    monkeypatch.setenv("MATERIAL_LOCAL_MAP_SPEED_MODE", "boost_only")
    assert local_map_linear_scale(_near_hit_advice(distance_m=0.30)) == pytest.approx(1.0)


def test_local_map_clear_boost_env(monkeypatch):
    monkeypatch.setenv("MATERIAL_LOCAL_MAP_CLEAR_BOOST", "1.55")
    from perception.local_map_sidecar import local_map_clear_boost

    assert local_map_clear_boost() == pytest.approx(1.55)


def test_local_map_linear_scale_noop_without_apply():
    from executors.local_map_motion import map_linear_scale

    advice = dict(_near_hit_advice(clear=True))
    advice["apply"] = False
    ctx = ExecutionContext(
        now_s=0.0,
        instruction={},
        task_index=0,
        attempt=1,
        local_map_advice=advice,
    )
    assert map_linear_scale(ctx) == pytest.approx(1.0)


def test_map_standoff_requires_apply_flag():
    advice = _near_hit_advice(distance_m=0.60)
    advice["apply"] = False
    ctx = ExecutionContext(now_s=0.0, instruction={}, task_index=0, attempt=1,
                           local_map_advice=advice)
    assert map_standoff_m(ctx, 0.42, max_standoff_m=0.88) == pytest.approx(0.42)


def test_sidecar_tick_fail_open_with_pose(monkeypatch):
    monkeypatch.setenv("MATERIAL_LOCAL_MAP", "1")
    monkeypatch.delenv("MATERIAL_LOCAL_MAP_APPLY", raising=False)
    side = LocalMapSidecar()
    assert side.enabled is True
    assert side.apply is False
    odom = SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )
    )
    advice = side.on_tick(now_s=1.0, odometry=odom, t_cam_world=None)
    assert advice.enabled is True
    assert advice.apply is False
    assert advice.fresh is False
    ctx = ExecutionContext(
        now_s=1.0,
        instruction={},
        task_index=0,
        attempt=1,
        local_map_advice=advice.as_dict(),
    )
    assert map_standoff_m(ctx, 0.50, max_standoff_m=0.88) == pytest.approx(0.50)


def test_sidecar_throttles_integrate(monkeypatch):
    monkeypatch.setenv("MATERIAL_LOCAL_MAP", "1")
    monkeypatch.setenv("MATERIAL_LOCAL_MAP_HZ", "1")
    monkeypatch.delenv("MATERIAL_LOCAL_MAP_APPLY", raising=False)
    side = LocalMapSidecar()
    assert "1Hz" in side.describe()
    odom = SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )
    )
    advice1 = side.on_tick(now_s=1.0, odometry=odom, t_cam_world=None)
    advice2 = side.on_tick(now_s=1.1, odometry=odom, t_cam_world=None)
    assert advice2.reason.startswith("throttled")
