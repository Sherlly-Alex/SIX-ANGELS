from __future__ import annotations

import math
import statistics
import time
from types import SimpleNamespace
import unittest

from control_types import ArmCommand
from executors.base import ExecutionContext, StageStatus, TaskStage
from executors.task1_full import Task1IntegratedExecutor
from executors.task2 import Task2IntegratedExecutor
from executors.task3 import Task3IntegratedExecutor
from executors.transfer_support import TransferMotion, stand_from_held_center
from navigation.carried_envelope import CarriedEnvelopeChecker, EnvelopeCheck, HeldObjectGeometry
from navigation.navigation_types import NavigationStatus
from navigation.task3_lateral_safety import (
    SafeLateralTarget,
    Task3LateralGuardParams,
    compute_safe_lateral_target,
    evaluate_left_placement_feasibility,
    guard_task3_lateral_cmd,
    is_lateral_motion_safe,
    left_placement_critical_white_y,
    left_placement_required_width_m,
)
from scheduler.models import FailureCode
from shelf.state_tracker import ShelfState
from shelf.task_memory import CompetitionTaskMemory


def _odom(x: float, y: float, yaw: float):
    orientation = SimpleNamespace(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0),
    )
    position = SimpleNamespace(x=x, y=y)
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(position=position, orientation=orientation)
        )
    )


def _hold_command() -> ArmCommand:
    return ArmCommand(
        spine_position=0.18,
        head_positions=(0.0, 0.45),
        left_arm_positions=(0.1,) * 6,
        left_gripper_position=1.0,
        right_arm_positions=(-0.1,) * 6,
        right_gripper_position=1.0,
    )


def _task3_memory() -> CompetitionTaskMemory:
    state = ShelfState(
        empty_layer=3,
        colored_layer=2,
        colored_class_id="brown",
        white_obstacle_layer=1,
        layer_contents=("packaging_box", "brown", "EMPTY"),
        layer_centers_world=(
            (-2.54, 0.778, 0.530),
            (-2.55, 0.810, 0.837),
            (-2.63, 0.778, 1.166),
        ),
        confidence=0.95,
        frames_used=7,
    )
    memory = CompetitionTaskMemory()
    memory.record_shelf_state(state)
    return memory


class LeftWallChecker(CarriedEnvelopeChecker):
    """Reject carried poses whose Y is south of a synthetic left wall."""

    def __init__(self, wall_y: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.wall_y = float(wall_y)

    def check_pose(self, pose, held_center_base, half_width_m) -> EnvelopeCheck:
        check = super().check_pose(pose, held_center_base, half_width_m)
        if float(pose[1]) < self.wall_y:
            return EnvelopeCheck(
                False,
                float(pose[1]) - self.wall_y,
                "carried_box clearance to left_wall is negative",
            )
        if check.safe:
            extra = float(pose[1]) - self.wall_y
            if extra < check.clearance_m:
                return EnvelopeCheck(
                    True,
                    extra,
                    f"minimum carried-envelope clearance={extra:.3f} m "
                    "(carried_box to left_wall)",
                )
        return check


class MidGapChecker(CarriedEnvelopeChecker):
    """Reject only a middle Y band so start and end can both be safe."""

    def __init__(self, low: float, high: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.low = float(low)
        self.high = float(high)

    def check_pose(self, pose, held_center_base, half_width_m) -> EnvelopeCheck:
        y = float(pose[1])
        if self.low < y < self.high:
            return EnvelopeCheck(
                False,
                -0.05,
                "carried_box clearance to mid_obstacle is negative",
            )
        return super().check_pose(pose, held_center_base, half_width_m)


class MidYawChecker(CarriedEnvelopeChecker):
    """Reject only a mid-sweep yaw so both rotation endpoints can be safe."""

    def __init__(self, blocked_yaw: float, half_width_rad: float = 0.20, **kwargs) -> None:
        super().__init__(**kwargs)
        self.blocked_yaw = float(blocked_yaw)
        self.half_width_rad = float(half_width_rad)

    def check_pose(self, pose, held_center_base, half_width_m) -> EnvelopeCheck:
        yaw = float(pose[2])
        delta = (yaw - self.blocked_yaw + math.pi) % (2.0 * math.pi) - math.pi
        if abs(delta) <= self.half_width_rad:
            return EnvelopeCheck(
                False,
                -0.02,
                "carried_box clearance to mid_yaw_wall is negative",
            )
        return EnvelopeCheck(True, 0.20, "mid_yaw_wall endpoints clear")


class NorthWallChecker(CarriedEnvelopeChecker):
    """Reject carried poses whose base Y is north of a synthetic wall."""

    def __init__(self, wall_y: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.wall_y = float(wall_y)

    def check_pose(self, pose, held_center_base, half_width_m) -> EnvelopeCheck:
        check = super().check_pose(pose, held_center_base, half_width_m)
        if float(pose[1]) > self.wall_y:
            return EnvelopeCheck(
                False,
                self.wall_y - float(pose[1]),
                "carried_box clearance to north_wall is negative",
            )
        if check.safe:
            extra = self.wall_y - float(pose[1])
            if extra < check.clearance_m:
                return EnvelopeCheck(
                    True,
                    extra,
                    f"minimum carried-envelope clearance={extra:.3f} m "
                    "(carried_box to north_wall)",
                )
        return check


class NorthPayloadWallChecker(CarriedEnvelopeChecker):
    """Reject when the carried-box centre, not just the base, crosses north."""

    def __init__(self, wall_y: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.wall_y = float(wall_y)

    def check_pose(self, pose, held_center_base, half_width_m) -> EnvelopeCheck:
        x, y, yaw = (float(value) for value in pose)
        hx, hy = float(held_center_base[0]), float(held_center_base[1])
        box_y = y + hx * math.sin(yaw) + hy * math.cos(yaw)
        if box_y > self.wall_y:
            return EnvelopeCheck(
                False,
                self.wall_y - box_y,
                "carried_box clearance to north_payload_wall is negative",
            )
        return EnvelopeCheck(
            True,
            self.wall_y - box_y,
            "north_payload_wall clear",
        )


HELD = HeldObjectGeometry((0.70, 0.0, 0.90), 0.08, source="test")
YAW = math.pi
N_REF = -0.92
CENTER_S = 0.78
PARAMS = Task3LateralGuardParams()


class Task3LateralSafetyUnitTests(unittest.TestCase):
    def test_case_a_centered_legacy_target_is_unchanged(self) -> None:
        result = compute_safe_lateral_target(
            (N_REF, CENTER_S, YAW),
            (N_REF, CENTER_S),
            YAW,
            HELD,
            params=PARAMS,
            white_object_y=0.778,
            place_radius_m=0.24,
            scoring_target_xy=(-2.68, 0.540),
        )
        self.assertTrue(result.feasible)
        self.assertFalse(result.was_clipped)
        self.assertAlmostEqual(result.safe_target_s, CENTER_S, places=6)
        self.assertEqual(result.reason, "already_at_legacy_target")
        self.assertTrue(result.skip_motion)

    def test_scoring_radius_does_not_keep_observation_y_in_front_of_white(self) -> None:
        """Official layout: white Y is 0.778 on every layer; left stand is ~0.58.

        Observation Y sits inside the 0.24 m scoring circle, but releasing
        there puts the carried box into the packaging cuboid.  Stop Y must
        remain the qzhRL left stand, not the current pose.
        """

        legacy = 0.58
        result = compute_safe_lateral_target(
            (N_REF, 0.778, YAW),
            (N_REF, legacy),
            YAW,
            HELD,
            params=PARAMS,
            white_object_y=0.778,
            place_radius_m=0.24,
            scoring_target_xy=(-2.68, 0.540),
        )
        self.assertTrue(result.feasible)
        self.assertFalse(result.skip_motion)
        self.assertAlmostEqual(result.safe_target_s, legacy, places=6)
        self.assertEqual(result.reason, "legacy_target_safe")

    def test_case_b_unsafe_envelope_still_parks_at_qzhrl_y(self) -> None:
        checker = LeftWallChecker(0.66)
        legacy = 0.58
        result = compute_safe_lateral_target(
            (N_REF, 0.90, YAW),
            (N_REF, legacy),
            YAW,
            HELD,
            checker=checker,
            params=PARAMS,
            white_object_y=0.778,
            place_radius_m=0.24,
            scoring_target_xy=(-2.68, 0.540),
        )
        self.assertTrue(result.feasible)
        self.assertFalse(result.was_clipped)
        self.assertAlmostEqual(result.safe_target_s, legacy, places=6)
        self.assertEqual(result.legacy_s, legacy)
        self.assertFalse(result.skip_motion)

    def test_case_c_extreme_left_is_infeasible(self) -> None:
        result = compute_safe_lateral_target(
            (N_REF, CENTER_S, YAW),
            (N_REF, 0.40),
            YAW,
            HELD,
            params=PARAMS,
            white_object_y=0.48,
            place_radius_m=0.24,
            scoring_target_xy=(-2.68, 0.40),
        )
        self.assertFalse(result.feasible)
        self.assertFalse(result.placement_feasible)
        self.assertEqual(result.reason, "placement_left_infeasible")
        self.assertIsNone(result.safe_target_s)

    def test_case_d_segment_detects_middle_obstacle(self) -> None:
        checker = MidGapChecker(0.70, 0.76)
        start = (N_REF, 0.64, YAW)
        end = (N_REF, 0.82, YAW)
        start_ok = checker.check_pose(start, HELD.center_base, HELD.half_width_m)
        end_ok = checker.check_pose(end, HELD.center_base, HELD.half_width_m)
        self.assertTrue(start_ok.safe)
        self.assertTrue(end_ok.safe)
        segment = is_lateral_motion_safe(
            start, end, HELD, checker=checker, params=PARAMS
        )
        self.assertFalse(segment.safe)
        self.assertIn("mid_obstacle", segment.detail)

    def test_case_i_rotation_sweep_rejects_mid_yaw(self) -> None:
        checker = MidYawChecker(0.75 * math.pi)
        start_yaw = math.pi / 2.0
        end_yaw = math.pi
        start = (N_REF, 0.58, start_yaw)
        end = (N_REF, 0.58, end_yaw)
        start_ok = checker.check_pose(start, HELD.center_base, HELD.half_width_m)
        end_ok = checker.check_pose(end, HELD.center_base, HELD.half_width_m)
        self.assertTrue(start_ok.safe)
        self.assertTrue(end_ok.safe)
        rotation = checker.check_rotation(
            start, end_yaw, HELD.center_base, HELD.half_width_m
        )
        self.assertFalse(rotation.safe)
        self.assertIn("mid_yaw_wall", rotation.detail)

    def test_case_i_restore_north_to_west_at_left_target(self) -> None:
        checker = MidYawChecker(0.75 * math.pi)
        start = (N_REF, 0.58, math.pi / 2.0)
        end = (N_REF, 0.58, math.pi)
        segment = is_lateral_motion_safe(
            start,
            end,
            HELD,
            checker=checker,
            params=PARAMS,
            travel_face_yaw=math.pi / 2.0,
        )
        self.assertFalse(segment.safe)
        self.assertIn("mid_yaw_wall", segment.detail)

    def test_case_i_west_to_north_face_at_start(self) -> None:
        checker = MidYawChecker(0.75 * math.pi)
        start = (N_REF, 0.58, math.pi)
        end = (N_REF, 0.58, math.pi)
        west_ok = checker.check_pose(start, HELD.center_base, HELD.half_width_m)
        self.assertTrue(west_ok.safe)
        segment = is_lateral_motion_safe(
            start,
            end,
            HELD,
            checker=checker,
            params=PARAMS,
            travel_face_yaw=math.pi / 2.0,
        )
        self.assertFalse(segment.safe)
        self.assertIn("mid_yaw_wall", segment.detail)

    def test_case_e_yaw_enters_envelope_check(self) -> None:
        checker = CarriedEnvelopeChecker()
        pose_west = (N_REF, 0.58, YAW)
        pose_south = (N_REF, 0.58, -math.pi / 2.0)
        west = checker.check_pose(pose_west, HELD.center_base, HELD.half_width_m)
        south = checker.check_pose(pose_south, HELD.center_base, HELD.half_width_m)
        self.assertTrue(west.safe)
        self.assertFalse(south.safe)
        self.assertNotAlmostEqual(west.clearance_m, south.clearance_m, places=3)

    def test_case_f_large_yaw_does_not_spin_at_shelf(self) -> None:
        result = compute_safe_lateral_target(
            (N_REF, CENTER_S, 0.0),
            (N_REF, CENTER_S),
            YAW,
            HELD,
            params=PARAMS,
            white_object_y=0.778,
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.reason, "yaw_error_too_large")
        motion = TransferMotion()
        started = motion.begin_lateral_alignment(
            (N_REF, CENTER_S),
            YAW,
            _odom(N_REF, CENTER_S, 0.0),
            0.0,
            held_geometry=HELD,
            travel_face_yaw=math.pi / 2.0,
            predictive_guard=True,
        )
        self.assertFalse(started)

    def test_case_h_disabled_guard_returns_legacy(self) -> None:
        checker = LeftWallChecker(0.66)
        legacy = 0.58
        result = compute_safe_lateral_target(
            (N_REF, CENTER_S, YAW),
            (N_REF, legacy),
            YAW,
            HELD,
            checker=checker,
            params=Task3LateralGuardParams(enabled=False),
            white_object_y=0.48,
        )
        self.assertTrue(result.feasible)
        self.assertFalse(result.was_clipped)
        self.assertAlmostEqual(result.safe_target_s, legacy, places=6)
        self.assertEqual(result.reason, "guard_disabled")
        self.assertFalse(result.skip_motion)
        parked = compute_safe_lateral_target(
            (N_REF, legacy, YAW),
            (N_REF, legacy),
            YAW,
            HELD,
            checker=checker,
            params=Task3LateralGuardParams(enabled=False),
            white_object_y=0.48,
        )
        self.assertEqual(parked.reason, "guard_disabled")
        self.assertFalse(parked.skip_motion)
        self.assertAlmostEqual(parked.safe_target_s, legacy, places=6)

    def test_north_face_southbound_reverses(self) -> None:
        motion = TransferMotion()
        started = motion.begin_lateral_alignment(
            (N_REF, 0.58),
            YAW,
            _odom(N_REF, 0.78, YAW),
            0.0,
            held_geometry=HELD,
            travel_face_yaw=math.pi / 2.0,
            predictive_guard=True,
        )
        self.assertTrue(started)
        self.assertAlmostEqual(motion._lateral_heading, math.pi / 2.0, places=6)
        self.assertAlmostEqual(motion._lateral_drive_sign, -1.0, places=6)
        self.assertEqual(motion._lateral_phase, "rotate_lateral")

    def test_north_face_northbound_drives_forward(self) -> None:
        motion = TransferMotion()
        started = motion.begin_lateral_alignment(
            (N_REF, 0.90),
            YAW,
            _odom(N_REF, 0.70, YAW),
            0.0,
            held_geometry=HELD,
            travel_face_yaw=math.pi / 2.0,
            predictive_guard=True,
        )
        self.assertTrue(started)
        self.assertAlmostEqual(motion._lateral_heading, math.pi / 2.0, places=6)
        self.assertAlmostEqual(motion._lateral_drive_sign, 1.0, places=6)

    def test_predictive_guard_stops_collision_command(self) -> None:
        guarded = guard_task3_lateral_cmd(
            (N_REF, 0.58, -math.pi / 2.0),
            (0.09, 0.0),
            HELD,
        )
        self.assertTrue(guarded.blocked)
        self.assertEqual((guarded.linear_x, guarded.angular_z), (0.0, 0.0))

    def test_predictive_guard_keeps_safe_command(self) -> None:
        guarded = guard_task3_lateral_cmd(
            (N_REF, CENTER_S, YAW),
            (0.0, 0.0),
            HELD,
        )
        self.assertTrue(guarded.allowed)
        self.assertFalse(guarded.blocked)

    def test_missing_geometry_fails_closed(self) -> None:
        result = compute_safe_lateral_target(
            (N_REF, CENTER_S, YAW),
            (N_REF, CENTER_S),
            YAW,
            None,
            params=PARAMS,
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.reason, "carried_envelope_unavailable")

    def test_left_placement_formula_has_no_double_counted_margin(self) -> None:
        checker = CarriedEnvelopeChecker()
        shelf = checker.obstacle_bounds("shelf")
        self.assertIsNotNone(shelf)
        assert shelf is not None
        shelf_ymin = float(shelf[2])
        required = left_placement_required_width_m(PARAMS)
        self.assertAlmostEqual(
            required,
            2.0 * PARAMS.place_object_half_y_m + 2.0 * PARAMS.carry_clearance_margin_m,
            places=9,
        )
        self.assertAlmostEqual(required, 0.20, places=6)
        critical = left_placement_critical_white_y(shelf_ymin, PARAMS)
        self.assertAlmostEqual(
            critical,
            shelf_ymin + PARAMS.white_object_half_y_m + required,
            places=9,
        )
        just_below = evaluate_left_placement_feasibility(
            white_object_y=critical - 0.001,
            checker=checker,
            params=PARAMS,
        )
        just_above = evaluate_left_placement_feasibility(
            white_object_y=critical + 0.001,
            checker=checker,
            params=PARAMS,
        )
        self.assertFalse(just_below.feasible)
        self.assertTrue(just_above.feasible)
        self.assertAlmostEqual(just_above.required_place_width_m, required, places=9)
        self.assertAlmostEqual(
            just_above.left_free_space_m,
            (critical + 0.001 - PARAMS.white_object_half_y_m) - shelf_ymin,
            places=9,
        )

    def test_left_placement_white_y_scan_matches_closed_form(self) -> None:
        checker = CarriedEnvelopeChecker()
        shelf = checker.obstacle_bounds("shelf")
        self.assertIsNotNone(shelf)
        assert shelf is not None
        shelf_ymin = float(shelf[2])
        critical = left_placement_critical_white_y(shelf_ymin, PARAMS)
        print(
            "\nTASK3_LEFT_PLACEMENT_SCAN "
            f"shelf_ymin={shelf_ymin:.4f} critical_white_y={critical:.4f} "
            f"required={left_placement_required_width_m(PARAMS):.4f}"
        )
        print(
            "  white_y left_free required I_place I_safe I_final "
            "feasible reason"
        )
        seen_infeasible = False
        seen_feasible = False
        for index in range(16):
            white_y = 0.55 + 0.01 * index
            placement = evaluate_left_placement_feasibility(
                white_object_y=white_y,
                checker=checker,
                params=PARAMS,
            )
            result = compute_safe_lateral_target(
                (N_REF, CENTER_S, YAW),
                (N_REF, max(0.58, min(0.98, white_y - 0.20))),
                YAW,
                HELD,
                params=PARAMS,
                white_object_y=white_y,
                place_radius_m=0.24,
                scoring_target_xy=(-2.68, white_y - 0.238),
            )
            print(
                f"  {white_y:.2f} {placement.left_free_space_m:.4f} "
                f"{placement.required_place_width_m:.4f} "
                f"[{result.place_min_s},{result.place_max_s}] "
                f"[{result.safe_min_s},{result.safe_max_s}] "
                f"{result.safe_target_s} {int(result.feasible)} {result.reason}"
            )
            if white_y + 1e-9 < critical:
                self.assertFalse(placement.feasible)
                self.assertEqual(result.reason, "placement_left_infeasible")
                seen_infeasible = True
            else:
                self.assertTrue(placement.feasible)
                seen_feasible = True
        self.assertTrue(seen_infeasible)
        self.assertTrue(seen_feasible)

    def test_guard_latency_is_lightweight(self) -> None:
        samples: list[float] = []
        for _ in range(40):
            started = time.perf_counter()
            compute_safe_lateral_target(
                (N_REF, CENTER_S, YAW),
                (N_REF, 0.70),
                YAW,
                HELD,
                params=PARAMS,
                white_object_y=0.778,
                place_radius_m=0.24,
                scoring_target_xy=(-2.68, 0.540),
            )
            samples.append(time.perf_counter() - started)
        average_ms = statistics.mean(samples) * 1000.0
        p95_ms = statistics.quantiles(samples, n=20)[18] * 1000.0
        self.assertLess(average_ms, 20.0)
        self.assertLess(p95_ms, 40.0)
        # Keep the numbers in the unittest output for the review report.
        print(
            f"\nTASK3_LATERAL_GUARD_LATENCY_MS average={average_ms:.3f} "
            f"p95={p95_ms:.3f}"
        )


class Task3LateralScenarioMatrixTests(unittest.TestCase):
    def _plan(self, white_y: float, legacy_s: float, current_s: float = CENTER_S):
        return compute_safe_lateral_target(
            (N_REF, current_s, YAW),
            (N_REF, legacy_s),
            YAW,
            HELD,
            params=PARAMS,
            white_object_y=white_y,
            place_radius_m=0.24,
            scoring_target_xy=(-2.68, white_y - 0.238),
        )

    def test_white_object_positions(self) -> None:
        rows = [
            ("near_left", 0.48, 0.40),
            ("left", 0.70, 0.58),
            ("center", 0.778, 0.78),
            ("right", 0.95, 0.90),
            ("near_right", 1.10, 0.98),
        ]
        print("\nTASK3_LATERAL_SCENARIO_MATRIX")
        for name, white_y, legacy_s in rows:
            result = self._plan(white_y, legacy_s)
            print(
                f"  {name}: white_y={white_y:.3f} legacy={legacy_s:.3f} "
                f"feasible={int(result.feasible)} clipped={int(result.was_clipped)} "
                f"final={result.safe_target_s} reason={result.reason} "
                f"clearance={result.min_clearance_m} skip={int(result.skip_motion)} "
                f"safe=[{result.safe_min_s},{result.safe_max_s}]"
            )
            if name == "near_left":
                self.assertFalse(result.feasible)
            elif name == "left":
                self.assertTrue(result.feasible)
                self.assertFalse(result.was_clipped)
                self.assertFalse(result.skip_motion)
                self.assertAlmostEqual(result.safe_target_s, legacy_s, places=6)
            elif name == "center":
                self.assertTrue(result.feasible)
                self.assertFalse(result.was_clipped)
                self.assertAlmostEqual(result.safe_target_s, legacy_s, places=6)
            elif name == "right":
                self.assertTrue(result.feasible)
                self.assertFalse(result.skip_motion)
                self.assertAlmostEqual(result.safe_target_s, legacy_s, places=6)
            elif name == "near_right":
                self.assertTrue(result.feasible)
                self.assertFalse(result.skip_motion)
                self.assertAlmostEqual(result.safe_target_s, legacy_s, places=6)

    def test_near_right_keeps_qzhrl_stop_even_with_north_wall(self) -> None:
        checker = NorthWallChecker(0.92)
        legacy = 0.98
        result = compute_safe_lateral_target(
            (N_REF, 0.70, YAW),
            (N_REF, legacy),
            YAW,
            HELD,
            checker=checker,
            params=PARAMS,
            white_object_y=1.10,
            place_radius_m=0.24,
            scoring_target_xy=(-2.68, 0.86),
        )
        self.assertTrue(result.feasible)
        self.assertFalse(result.was_clipped)
        self.assertAlmostEqual(result.safe_target_s, legacy, places=6)

    def test_northbound_travel_payload_leading_is_rejected(self) -> None:
        checker = NorthPayloadWallChecker(1.05)
        start = (N_REF, 0.70, math.pi)
        end = (N_REF, 0.90, math.pi)
        start_ok = checker.check_pose(start, HELD.center_base, HELD.half_width_m)
        end_ok = checker.check_pose(end, HELD.center_base, HELD.half_width_m)
        self.assertTrue(start_ok.safe)
        self.assertTrue(end_ok.safe)
        frozen = is_lateral_motion_safe(
            start, end, HELD, checker=checker, params=PARAMS
        )
        self.assertTrue(frozen.safe)
        north_travel = is_lateral_motion_safe(
            start,
            end,
            HELD,
            checker=checker,
            params=PARAMS,
            travel_face_yaw=math.pi / 2.0,
        )
        self.assertFalse(north_travel.safe)
        self.assertIn("north_payload_wall", north_travel.detail)
        guarded = guard_task3_lateral_cmd(
            (N_REF, 0.70, math.pi / 2.0),
            (0.09, 0.0),
            HELD,
            checker=checker,
        )
        self.assertTrue(guarded.blocked)


class Task3ExecutorLateralGuardTests(unittest.TestCase):
    def _executor(self) -> Task3IntegratedExecutor:
        executor = Task3IntegratedExecutor(_task3_memory())
        executor._held_arm_command = _hold_command()
        executor._held_center_base = (0.70, 0.0, 0.90)
        executor._held_grasp_half_width = 0.08
        executor._held_grasp_orientation = "yaw90"
        executor._place_world = (-2.59, 0.575, 0.498)
        executor._shelf_scan_stand = (N_REF, CENTER_S)
        executor._shelf_state = executor._memory.require_shelf_state()
        return executor

    def _expected_parking_y(self, executor: Task3IntegratedExecutor) -> tuple[float, float]:
        qzhRL_y = max(
            0.58,
            min(
                0.98,
                stand_from_held_center(
                    executor._place_world,
                    executor._held_center_base,
                    executor.SHELF_YAW,
                )[1],
            ),
        )
        parking_y = max(
            executor.TASK3_LATERAL_Y_MIN_M,
            qzhRL_y - executor.TASK3_LATERAL_LEFT_BIAS_M,
        )
        return qzhRL_y, parking_y

    def test_already_aligned_skips_chassis_move(self) -> None:
        executor = self._executor()
        executor.enter_stage(
            TaskStage.ALIGN_FOR_PLACE,
            ExecutionContext(now_s=1.0, instruction={}, task_index=2, attempt=1),
        )
        executor._phase = "task3_lateral"
        executor._shelf_scan_stand = (N_REF, CENTER_S)
        executor._place_world = (-2.59, CENTER_S, 0.498)
        executor._task3_release_place = (-2.59, CENTER_S, 0.498)
        executor._task3_scoring_place = (-2.68, 0.540, 0.498)
        executor._task3_place_radius_m = 0.24
        _, parking_y = self._expected_parking_y(executor)
        context = ExecutionContext(
            now_s=2.0,
            instruction={
                "task": 3,
                "target_color": "pink",
                "place_type": "shelf_prop_side",
                "direction": "left",
                "place_world": [-2.68, 0.540, 0.498],
                "place_radius": 0.24,
            },
            task_index=2,
            attempt=1,
            odometry=_odom(N_REF, parking_y, YAW),
        )
        result = executor.tick(TaskStage.ALIGN_FOR_PLACE, context)
        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertFalse(result.controls_base)
        self.assertIn("TASK3_LATERAL", result.message)
        self.assertEqual(executor._phase, "task3_advance")

    def test_left_of_white_starts_lateral_to_legacy_y(self) -> None:
        executor = self._executor()
        executor.enter_stage(
            TaskStage.ALIGN_FOR_PLACE,
            ExecutionContext(now_s=1.0, instruction={}, task_index=2, attempt=1),
        )
        executor._phase = "task3_lateral"
        executor._shelf_scan_stand = (N_REF, CENTER_S)
        executor._task3_release_place = (-2.59, 0.575, 0.498)
        executor._task3_scoring_place = (-2.68, 0.540, 0.498)
        executor._task3_place_radius_m = 0.24
        result = executor.tick(
            TaskStage.ALIGN_FOR_PLACE,
            ExecutionContext(
                now_s=2.0,
                instruction={
                    "task": 3,
                    "target_color": "pink",
                    "place_type": "shelf_prop_side",
                    "direction": "left",
                    "place_world": [-2.68, 0.540, 0.498],
                    "place_radius": 0.24,
                },
                task_index=2,
                attempt=1,
                odometry=_odom(N_REF, CENTER_S, YAW),
            ),
        )
        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertTrue(result.controls_base)
        self.assertEqual(executor._phase, "task3_lateral")
        self.assertFalse(executor._task3_lateral_plan.skip_motion)
        qzhRL_y, expected_y = self._expected_parking_y(executor)
        self.assertAlmostEqual(executor._task3_qzhrl_lateral_y, qzhRL_y, places=6)
        self.assertAlmostEqual(
            executor._task3_shallow_place_stand[1], expected_y, places=6
        )
        self.assertAlmostEqual(
            executor._transfer._lateral_target[1], expected_y, places=6
        )
        self.assertLess(executor._task3_shallow_place_stand[1], CENTER_S - 0.05)
        self.assertAlmostEqual(
            executor._transfer._lateral_heading, math.pi / 2.0, places=6
        )
        self.assertAlmostEqual(executor._transfer._lateral_drive_sign, -1.0, places=6)
        self.assertTrue(executor._transfer._lateral_predictive_guard)
        self.assertEqual(executor._transfer._lateral_phase, "rotate_lateral")

    def test_safe_guard_parks_left_of_the_qzhrl_clamp(self) -> None:
        executor = self._executor()
        executor.enter_stage(
            TaskStage.ALIGN_FOR_PLACE,
            ExecutionContext(now_s=1.0, instruction={}, task_index=2, attempt=1),
        )
        executor._phase = "task3_lateral"
        executor._shelf_scan_stand = (N_REF, CENTER_S)
        executor._task3_release_place = (-2.59, 0.575, 0.498)
        executor._task3_scoring_place = (-2.68, 0.540, 0.498)
        executor._task3_place_radius_m = 0.24
        result = executor.tick(
            TaskStage.ALIGN_FOR_PLACE,
            self._lateral_context(CENTER_S, YAW),
        )

        qzhRL_y, expected_y = self._expected_parking_y(executor)
        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertIsNotNone(executor._task3_lateral_plan)
        self.assertFalse(executor._task3_lateral_plan.was_clipped)
        self.assertFalse(executor._task3_lateral_plan.skip_motion)
        self.assertEqual(executor._task3_lateral_plan.reason, "legacy_target_safe")
        self.assertAlmostEqual(executor._task3_qzhrl_lateral_y, qzhRL_y, places=6)
        self.assertAlmostEqual(executor._task3_legacy_lateral_y, expected_y, places=6)
        self.assertAlmostEqual(
            executor._task3_lateral_plan.safe_target_s, expected_y, places=6
        )
        self.assertAlmostEqual(executor._task3_shallow_place_stand[1], expected_y, places=6)
        self.assertAlmostEqual(executor._transfer._lateral_target[1], expected_y, places=6)
        self.assertAlmostEqual(expected_y, 0.54, places=2)
        self.assertLess(expected_y, qzhRL_y)
        self.assertIn("parking_s=", result.message)
        self.assertIn("current_s=", result.message)

    def test_github_parking_stand_shifts_left_of_qzhrl_clamp(self) -> None:
        executor = self._executor()
        executor._freeze_task3_github_parking_stand()
        qzhRL_y, parking_y = self._expected_parking_y(executor)
        self.assertAlmostEqual(qzhRL_y, 0.58, places=6)
        self.assertAlmostEqual(parking_y, 0.54, places=6)
        self.assertAlmostEqual(executor._task3_parking_y(), 0.54, places=6)
        self.assertAlmostEqual(executor._task3_qzhrl_lateral_y, 0.58, places=6)
        self.assertAlmostEqual(executor._final_place_stand[1], 0.54, places=6)
        self.assertAlmostEqual(executor._task3_shallow_place_stand[1], 0.54, places=6)

    def test_disabled_guard_keeps_legacy_y(self) -> None:
        executor = self._executor()
        executor.set_task3_lateral_guard(False)
        executor.enter_stage(
            TaskStage.ALIGN_FOR_PLACE,
            ExecutionContext(now_s=1.0, instruction={}, task_index=2, attempt=1),
        )
        executor._phase = "task3_lateral"
        executor._shelf_scan_stand = (N_REF, CENTER_S)
        context = ExecutionContext(
            now_s=2.0,
            instruction={
                "task": 3,
                "target_color": "pink",
                "place_type": "shelf_prop_side",
                "direction": "left",
                "place_world": [-2.68, 0.540, 0.498],
                "place_radius": 0.24,
            },
            task_index=2,
            attempt=1,
            odometry=_odom(N_REF, CENTER_S, YAW),
        )
        result = executor.tick(TaskStage.ALIGN_FOR_PLACE, context)
        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertIsNotNone(executor._task3_shallow_place_stand)
        self.assertAlmostEqual(
            executor._task3_lateral_plan.legacy_s,
            executor._task3_shallow_place_stand[1],
            places=6,
        )
        self.assertFalse(executor._task3_lateral_plan.was_clipped)
        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertTrue(result.controls_base)
        self.assertAlmostEqual(result.base_linear_x, 0.0, places=6)
        self.assertGreater(result.base_angular_z, 0.0)
        self.assertAlmostEqual(
            executor._transfer._lateral_heading, -math.pi / 2.0, places=6
        )
        self.assertFalse(executor._transfer._lateral_predictive_guard)
        self.assertIsNone(executor._transfer._lateral_held_geometry)
        self.assertEqual(executor._transfer._lateral_phase, "rotate_lateral")
        self.assertNotIn("predictive_guard", result.message)
        self.assertEqual(executor._phase, "task3_lateral")

    def test_disabled_guard_restores_yaw_when_y_already_aligned(self) -> None:
        executor = self._executor()
        executor.set_task3_lateral_guard(False)
        executor.enter_stage(
            TaskStage.ALIGN_FOR_PLACE,
            ExecutionContext(now_s=1.0, instruction={}, task_index=2, attempt=1),
        )
        executor._phase = "task3_lateral"
        executor._shelf_scan_stand = (N_REF, CENTER_S)
        executor._final_place_stand = (N_REF, CENTER_S)
        executor._task3_shallow_place_stand = (N_REF, CENTER_S)
        yaw_off = YAW + 0.15
        result = executor.tick(
            TaskStage.ALIGN_FOR_PLACE, self._lateral_context(CENTER_S, yaw_off)
        )
        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertEqual(executor._phase, "task3_lateral")
        self.assertFalse(executor._task3_lateral_plan.skip_motion)
        self.assertEqual(executor._transfer._lateral_phase, "rotate_final")
        self.assertTrue(result.controls_base)
        self.assertNotAlmostEqual(result.base_angular_z, 0.0, places=3)
        self.assertFalse(executor._transfer._lateral_predictive_guard)

    def test_enabled_guard_restores_yaw_when_y_already_aligned(self) -> None:
        executor = self._executor()
        executor.enter_stage(
            TaskStage.ALIGN_FOR_PLACE,
            ExecutionContext(now_s=1.0, instruction={}, task_index=2, attempt=1),
        )
        executor._phase = "task3_lateral"
        executor._shelf_scan_stand = (N_REF, CENTER_S)
        executor._final_place_stand = (N_REF, CENTER_S)
        executor._task3_shallow_place_stand = (N_REF, CENTER_S)
        executor._task3_release_place = (-2.59, CENTER_S, 0.498)
        executor._task3_scoring_place = (-2.68, 0.540, 0.498)
        executor._task3_place_radius_m = 0.24
        yaw_off = YAW + 0.15
        result = executor.tick(
            TaskStage.ALIGN_FOR_PLACE, self._lateral_context(CENTER_S, yaw_off)
        )
        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertEqual(executor._phase, "task3_lateral")
        self.assertEqual(executor._transfer._lateral_phase, "rotate_final")
        self.assertTrue(result.controls_base)
        self.assertNotAlmostEqual(result.base_angular_z, 0.0, places=3)
        self.assertTrue(executor._transfer._lateral_predictive_guard)

    def _lateral_context(self, y: float, yaw: float, now_s: float = 2.0):
        return ExecutionContext(
            now_s=now_s,
            instruction={
                "task": 3,
                "target_color": "pink",
                "place_type": "shelf_prop_side",
                "direction": "left",
                "place_world": [-2.68, 0.540, 0.498],
                "place_radius": 0.24,
            },
            task_index=2,
            attempt=1,
            odometry=_odom(N_REF, y, yaw),
        )

    def test_infeasible_left_gap_blocks_before_lateral_motion(self) -> None:
        executor = self._executor()
        executor._memory.task3_packaging_box_center_world = (-2.54, 0.55, 0.530)
        executor.enter_stage(
            TaskStage.ALIGN_FOR_PLACE,
            ExecutionContext(now_s=1.0, instruction={}, task_index=2, attempt=1),
        )
        executor._phase = "task3_lateral"
        executor._shelf_scan_stand = (N_REF, CENTER_S)
        context = self._lateral_context(CENTER_S, YAW)
        first = executor.tick(TaskStage.ALIGN_FOR_PLACE, context)
        self.assertEqual(first.status, StageStatus.BLOCKED)
        self.assertIs(first.failure_code, FailureCode.UNSAFE_COLLISION)
        self.assertIn("placement_left_infeasible", first.message)
        self.assertEqual(executor._phase, "task3_lateral")
        self.assertIsNone(executor._transfer._lateral_target)

    def test_yaw_error_too_large_blocks_and_does_not_rebegin(self) -> None:
        executor = self._executor()
        executor.enter_stage(
            TaskStage.ALIGN_FOR_PLACE,
            ExecutionContext(now_s=1.0, instruction={}, task_index=2, attempt=1),
        )
        executor._phase = "task3_lateral"
        executor._shelf_scan_stand = (N_REF, CENTER_S)
        first = executor.tick(
            TaskStage.ALIGN_FOR_PLACE, self._lateral_context(CENTER_S, 0.0)
        )
        self.assertEqual(first.status, StageStatus.BLOCKED)
        self.assertIs(first.failure_code, FailureCode.UNSAFE_COLLISION)
        self.assertIn("yaw_error_too_large", first.message)
        second = executor.tick(
            TaskStage.ALIGN_FOR_PLACE, self._lateral_context(CENTER_S, 0.0, now_s=2.1)
        )
        self.assertIs(second, first)
        self.assertIsNone(executor._transfer._lateral_target)

    def test_no_feasible_plan_blocks_before_lateral_motion(self) -> None:
        executor = self._executor()
        executor.enter_stage(
            TaskStage.ALIGN_FOR_PLACE,
            ExecutionContext(now_s=1.0, instruction={}, task_index=2, attempt=1),
        )
        executor._phase = "task3_lateral"
        executor._shelf_scan_stand = (N_REF, CENTER_S)
        executor._final_place_stand = (N_REF, 0.58)
        executor._task3_shallow_place_stand = (N_REF, 0.58)
        executor._task3_lateral_plan = SafeLateralTarget(
            feasible=False,
            current_s=CENTER_S,
            legacy_s=0.58,
            safe_target_s=None,
            safe_min_s=0.70,
            safe_max_s=1.20,
            place_min_s=0.38,
            place_max_s=0.42,
            was_clipped=False,
            placement_feasible=True,
            motion_safe=False,
            skip_motion=False,
            yaw_error_rad=0.0,
            min_clearance_m=-0.01,
            reason="no_feasible_lateral_target",
            latency_s=0.0,
        )
        first = executor.tick(
            TaskStage.ALIGN_FOR_PLACE, self._lateral_context(CENTER_S, YAW)
        )
        self.assertEqual(first.status, StageStatus.BLOCKED)
        self.assertIs(first.failure_code, FailureCode.UNSAFE_COLLISION)
        self.assertIn("no_feasible_lateral_target", first.message)
        self.assertIsNone(executor._transfer._lateral_target)

    def test_clipped_plan_does_not_rewrite_the_qzhrl_stand(self) -> None:
        executor = self._executor()
        executor._final_place_stand = (N_REF, 0.58)
        executor._task3_shallow_place_stand = (N_REF, 0.58)
        executor._task3_legacy_lateral_y = 0.58
        plan = SafeLateralTarget(
            feasible=True,
            current_s=CENTER_S,
            legacy_s=0.58,
            safe_target_s=0.66,
            safe_min_s=0.66,
            safe_max_s=0.80,
            place_min_s=0.40,
            place_max_s=0.80,
            was_clipped=True,
            placement_feasible=True,
            motion_safe=True,
            skip_motion=False,
            yaw_error_rad=0.0,
            min_clearance_m=0.02,
            reason="clipped_to_safe_interval",
            latency_s=0.0,
        )

        result = executor._apply_task3_safe_lateral_target(plan)

        self.assertIsNone(result)
        self.assertAlmostEqual(executor._task3_shallow_place_stand[0], N_REF)
        self.assertAlmostEqual(executor._task3_shallow_place_stand[1], 0.58)
        self.assertAlmostEqual(executor._final_place_stand[1], 0.58)
        self.assertAlmostEqual(executor._task3_parking_y(), 0.58)

    def test_unclipped_plan_keeps_qzhrl_stop_even_if_reported_target_differs(self) -> None:
        executor = self._executor()
        executor._final_place_stand = (N_REF, 0.58)
        executor._task3_shallow_place_stand = (N_REF, 0.58)
        executor._task3_legacy_lateral_y = 0.58
        plan = SafeLateralTarget(
            feasible=True,
            current_s=CENTER_S,
            legacy_s=0.58,
            safe_target_s=0.778,
            safe_min_s=0.58,
            safe_max_s=1.10,
            place_min_s=0.38,
            place_max_s=0.78,
            was_clipped=False,
            placement_feasible=True,
            motion_safe=True,
            skip_motion=False,
            yaw_error_rad=0.0,
            min_clearance_m=0.05,
            reason="legacy_target_safe",
            latency_s=0.0,
        )

        result = executor._apply_task3_safe_lateral_target(plan)

        self.assertIsNone(result)
        self.assertAlmostEqual(executor._task3_shallow_place_stand[1], 0.58)
        self.assertAlmostEqual(executor._final_place_stand[1], 0.58)
        self.assertAlmostEqual(executor._task3_parking_y(), 0.58)

    def test_motion_unsafe_plan_still_parks_at_qzhrl_y(self) -> None:
        executor = self._executor()
        executor._final_place_stand = (N_REF, 0.58)
        executor._task3_shallow_place_stand = (N_REF, 0.58)
        executor._task3_legacy_lateral_y = 0.58
        plan = SafeLateralTarget(
            feasible=True,
            current_s=CENTER_S,
            legacy_s=0.58,
            safe_target_s=0.58,
            safe_min_s=0.58,
            safe_max_s=1.10,
            place_min_s=0.38,
            place_max_s=0.78,
            was_clipped=False,
            placement_feasible=True,
            motion_safe=False,
            skip_motion=False,
            yaw_error_rad=0.0,
            min_clearance_m=-0.02,
            reason="legacy_stop_kept",
            latency_s=0.0,
        )

        result = executor._apply_task3_safe_lateral_target(plan)

        self.assertIsNone(result)
        self.assertAlmostEqual(executor._task3_shallow_place_stand[1], 0.58)
        self.assertAlmostEqual(executor._task3_parking_y(), 0.58)

    def test_emergency_stop_maps_to_unsafe_collision_and_latches(self) -> None:
        executor = self._executor()
        executor.enter_stage(
            TaskStage.ALIGN_FOR_PLACE,
            ExecutionContext(now_s=1.0, instruction={}, task_index=2, attempt=1),
        )
        executor._phase = "task3_lateral"
        executor._shelf_scan_stand = (N_REF, CENTER_S)
        executor._final_place_stand = (N_REF, 0.58)
        executor._task3_shallow_place_stand = (N_REF, 0.58)
        executor._task3_lateral_plan = SafeLateralTarget(
            feasible=True,
            current_s=CENTER_S,
            legacy_s=0.58,
            safe_target_s=0.58,
            safe_min_s=0.58,
            safe_max_s=1.10,
            place_min_s=0.40,
            place_max_s=0.80,
            was_clipped=False,
            placement_feasible=True,
            motion_safe=True,
            skip_motion=False,
            yaw_error_rad=0.0,
            min_clearance_m=0.05,
            reason="legacy_target_safe",
            latency_s=0.0,
        )
        executor._motion_started = True
        first = executor.tick(
            TaskStage.ALIGN_FOR_PLACE, self._lateral_context(CENTER_S, YAW)
        )
        self.assertEqual(first.status, StageStatus.BLOCKED)
        self.assertIs(first.failure_code, FailureCode.UNSAFE_COLLISION)
        self.assertIn("TASK3_LATERAL_BLOCKED", first.message)
        second = executor.tick(
            TaskStage.ALIGN_FOR_PLACE, self._lateral_context(CENTER_S, YAW, now_s=2.1)
        )
        self.assertIs(second, first)

    def test_predictive_guard_consecutive_blocks_become_emergency_stop(self) -> None:
        motion = TransferMotion()
        started = motion.begin_lateral_alignment(
            (N_REF, 0.40),
            YAW,
            _odom(N_REF, 0.58, YAW),
            0.0,
            held_geometry=HELD,
            travel_face_yaw=math.pi / 2.0,
            predictive_guard=True,
        )
        self.assertTrue(started)
        colliding_pose = (N_REF, 0.58, -math.pi / 2.0)
        colliding_command = (0.09, 0.0)
        status = NavigationStatus.NAVIGATING
        detail = ""
        navigating_holds = 0
        for _ in range(5):
            status, command, detail = motion._emit_lateral_command(
                colliding_pose, colliding_command, "southbound drive"
            )
            self.assertEqual(command, (0.0, 0.0))
            if status is NavigationStatus.EMERGENCY_STOP:
                break
            self.assertEqual(status, NavigationStatus.NAVIGATING)
            navigating_holds += 1
        self.assertEqual(navigating_holds, 2)
        self.assertEqual(status, NavigationStatus.EMERGENCY_STOP)
        self.assertIn("TASK3_LATERAL_BLOCKED", detail)
        self.assertEqual(motion._lateral_phase, "failed")

    def test_extra_base_advance_times_out_instead_of_crawling(self) -> None:
        executor = self._executor()
        executor.enter_stage(
            TaskStage.ALIGN_FOR_PLACE,
            ExecutionContext(now_s=1.0, instruction={}, task_index=2, attempt=1),
        )
        executor._phase = "task3_extra_base_advance"
        executor._phase_started_s = 2.0
        executor._motion_started = False
        executor._final_place_stand = (N_REF, 0.54)
        executor._task3_shallow_place_stand = (N_REF, 0.54)
        first = executor.tick(
            TaskStage.ALIGN_FOR_PLACE, self._lateral_context(0.54, YAW, now_s=2.0)
        )
        self.assertEqual(first.status, StageStatus.RUNNING)
        self.assertEqual(executor._phase, "task3_extra_base_advance")
        timed_out = executor.tick(
            TaskStage.ALIGN_FOR_PLACE, self._lateral_context(0.54, YAW, now_s=10.5)
        )
        self.assertEqual(timed_out.status, StageStatus.SUCCEEDED)
        self.assertIn("extra_advance_timeout", timed_out.message)


class Task12NoRegressionTests(unittest.TestCase):
    def test_case_g_task1_and_task2_still_use_unguarded_lateral_api(self) -> None:
        motion = TransferMotion()
        self.assertTrue(
            motion.begin_lateral_alignment(
                (-1.30, 1.00),
                math.pi,
                _odom(-1.30, 0.85, math.pi),
                0.0,
            )
        )
        status, command, detail = motion.tick_lateral_alignment(
            _odom(-1.30, 0.85, math.pi), 0.05
        )
        self.assertEqual(status, NavigationStatus.NAVIGATING)
        self.assertEqual(command[0], 0.0)
        self.assertIn("rotating toward shelf-front", detail)
        self.assertIsInstance(Task1IntegratedExecutor(_task3_memory()), Task1IntegratedExecutor)
        self.assertIsInstance(Task2IntegratedExecutor(_task3_memory()), Task2IntegratedExecutor)
        self.assertFalse(hasattr(Task1IntegratedExecutor, "set_task3_lateral_guard"))
        self.assertFalse(hasattr(Task2IntegratedExecutor, "set_task3_lateral_guard"))


if __name__ == "__main__":
    unittest.main()
