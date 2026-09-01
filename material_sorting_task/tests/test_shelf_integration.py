from __future__ import annotations

from dataclasses import replace
import math
import unittest
from types import SimpleNamespace

from control_types import ArmCommand
from competition_controller import CompetitionController
from desktop_grasp.pregrasp_core import PregraspPlanningError, SPINE_MIN
from executors import build_task_executors
from executors.base import ExecutionContext, StageStatus, TargetObservation, TaskStage
from executors.task1_full import (
    Task1IntegratedExecutor,
    shelf_observation_stand,
    target_delta_in_heading,
)
from executors.task2 import Task2IntegratedExecutor
from executors.transfer_support import TransferMotion, stand_from_held_center
from navigation.carried_envelope import CarriedEnvelopeChecker, HeldObjectGeometry
from navigation.navigation_types import NavigationGoal, NavigationSegment
from navigation.navigation_types import NavigationStatus
from shelf.manipulation import (
    ArmRetractController,
    HeldTransportController,
    ReleaseSpreadController,
    ShelfOpenPregraspController,
    SlideHoldController,
)
from shelf.placement_feedback import CompliantSlideLoweringController
from shelf.state_tracker import ShelfStateTracker
from shelf.target_center import StableTargetCenterTracker
from shelf.task_memory import CompetitionTaskMemory


def observation(label: str, xyz, stamp: float, score: float = 0.9) -> TargetObservation:
    return TargetObservation(
        color=label,
        position_world=tuple(float(value) for value in xyz),
        received_at_s=float(stamp),
        score=float(score),
    )


class ShelfStateTrackerTests(unittest.TestCase):
    def test_infers_unique_empty_layer_and_filters_carried_color(self) -> None:
        tracker = ShelfStateTracker(required_votes=3)
        result = None
        for index in range(3):
            stamp = 10.0 + index
            result = tracker.update(
                {
                    "brown": observation("brown", (-2.55, 0.81, 0.837), stamp),
                    "packaging_box": observation(
                        "packaging_box", (-2.54, 0.78, 0.530), stamp
                    ),
                    # A carried task-1 box can be visible in front of the
                    # camera and must not become shelf occupancy evidence.
                    "pink": observation("pink", (-2.55, 0.78, 1.166), stamp),
                },
                now_s=stamp,
                carried_class_id="pink",
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.colored_class_id, "brown")
        self.assertEqual(result.colored_layer, 2)
        self.assertEqual(result.white_obstacle_layer, 1)
        self.assertEqual(result.empty_layer, 3)
        self.assertEqual(result.layer_contents, ("packaging_box", "brown", "EMPTY"))
        # Occupied-object centers come from their own RGB-D observations;
        # only the empty slot uses calibrated shelf geometry.
        self.assertEqual(result.task2_target_center_world, (-2.55, 0.81, 0.837))
        self.assertEqual(
            result.task3_packaging_box_center_world,
            (-2.54, 0.78, 0.530),
        )
        self.assertAlmostEqual(result.empty_place_world[2], 1.166, places=3)
        self.assertAlmostEqual(result.empty_shelf_center_world[0], -2.63, places=3)
        self.assertAlmostEqual(result.empty_shelf_center_world[1], 0.778, places=3)

    def test_desktop_material_box_is_not_part_of_shelf_state(self) -> None:
        tracker = ShelfStateTracker(required_votes=3)
        result = None
        for index in range(3):
            stamp = 30.0 + index
            result = tracker.update(
                {
                    "brown": observation("brown", (-2.55, 0.81, 0.837), stamp),
                    "packaging_box": observation(
                        "packaging_box", (-2.54, 0.78, 0.530), stamp
                    ),
                    # This is a desktop obstacle and must not create another
                    # shelf layer or enter the shared shelf-state cache.
                    "material_box": observation(
                        "material_box", (-0.54, 2.30, 0.833), stamp
                    ),
                },
                now_s=stamp,
                carried_class_id="pink",
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.layer_contents, ("packaging_box", "brown", "EMPTY"))

    def test_shelf_local_material_box_label_is_packaging_alias(self) -> None:
        tracker = ShelfStateTracker(required_votes=3)
        result = None
        for stamp in (1.0, 2.0, 3.0):
            result = tracker.update(
                {
                    "pink": observation("pink", (-2.63, 0.778, 0.837), stamp),
                    # The official YOLO checkpoint can confuse the two white
                    # props.  Shelf coordinates disambiguate this safely.
                    "material_box": observation(
                        "material_box", (-2.636, 0.690, 0.563), stamp
                    ),
                },
                now_s=stamp,
                carried_class_id="yellow",
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.colored_layer, 2)
        self.assertEqual(result.white_obstacle_layer, 1)
        self.assertEqual(result.empty_layer, 3)
        self.assertEqual(result.layer_contents, ("packaging_box", "pink", "EMPTY"))

    def test_depth_occupancy_recovers_missed_l1_packaging_label(self) -> None:
        tracker = ShelfStateTracker(
            required_votes=3,
            require_empty_confirmation=True,
        )
        result = None
        for stamp in (1.0, 2.0, 3.0):
            result = tracker.update(
                {
                    "yellow": observation(
                        "yellow", (-2.63, 0.778, 0.837), stamp
                    ),
                    "shelf_empty": observation(
                        "shelf_empty", (-2.63, 0.778, 1.166), stamp
                    ),
                    "shelf_obstacle": observation(
                        "shelf_obstacle", (-2.61, 0.742, 0.557), stamp, 0.75
                    ),
                },
                now_s=stamp,
                carried_class_id="pink",
                expected_colored_class_id="yellow",
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.layer_contents, ("packaging_box", "yellow", "EMPTY"))
        self.assertEqual(result.white_obstacle_layer, 1)
        self.assertEqual(
            result.task3_packaging_box_center_world,
            (-2.61, 0.742, 0.557),
        )
        self.assertIn("occupancy=locked:L1", tracker.diagnostic_summary)

    def test_official_l1_layout_fallback_requires_confirmed_empty_layer(self) -> None:
        tracker = ShelfStateTracker(
            required_votes=3,
            require_empty_confirmation=True,
            allow_l1_layout_packaging_fallback=True,
        )
        result = None
        for stamp in (1.0, 2.0, 3.0):
            result = tracker.update(
                {
                    "yellow": observation(
                        "yellow", (-2.63, 0.778, 0.837), stamp
                    ),
                    "shelf_empty": observation(
                        "shelf_empty", (-2.63, 0.778, 1.166), stamp
                    ),
                },
                now_s=stamp,
                carried_class_id="pink",
                expected_colored_class_id="yellow",
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.layer_contents, ("packaging_box", "yellow", "EMPTY"))
        self.assertEqual(result.white_obstacle_layer, 1)
        self.assertIn("layout-inferred:L1", tracker.diagnostic_summary)

    def test_l1_layout_fallback_is_disabled_by_default(self) -> None:
        tracker = ShelfStateTracker(
            required_votes=3,
            require_empty_confirmation=True,
        )
        result = None
        for stamp in (1.0, 2.0, 3.0):
            result = tracker.update(
                {
                    "yellow": observation(
                        "yellow", (-2.63, 0.778, 0.837), stamp
                    ),
                    "shelf_empty": observation(
                        "shelf_empty", (-2.63, 0.778, 1.166), stamp
                    ),
                },
                now_s=stamp,
                carried_class_id="pink",
                expected_colored_class_id="yellow",
            )

        self.assertIsNone(result)

    def test_depth_occupancy_cannot_replace_missing_semantic_evidence(self) -> None:
        tracker = ShelfStateTracker(
            required_votes=3,
            require_empty_confirmation=True,
        )
        result = None
        for stamp in (1.0, 2.0, 3.0):
            result = tracker.update(
                {
                    "yellow": observation(
                        "yellow", (-2.63, 0.778, 0.837), stamp
                    ),
                    "shelf_obstacle": observation(
                        "shelf_obstacle", (-2.61, 0.742, 0.557), stamp, 0.75
                    ),
                },
                now_s=stamp,
                carried_class_id="pink",
            )

        self.assertIsNone(result)

    def test_depth_occupancy_must_match_unique_remaining_layer(self) -> None:
        tracker = ShelfStateTracker(
            required_votes=3,
            require_empty_confirmation=True,
        )
        result = None
        for stamp in (1.0, 2.0, 3.0):
            result = tracker.update(
                {
                    "yellow": observation(
                        "yellow", (-2.63, 0.778, 0.530), stamp
                    ),
                    "shelf_empty": observation(
                        "shelf_empty", (-2.63, 0.778, 1.166), stamp
                    ),
                    # The current depth-only probe is physically restricted
                    # to L1, while the unique remaining layer here is L2.
                    "shelf_obstacle": observation(
                        "shelf_obstacle", (-2.61, 0.742, 0.557), stamp, 0.75
                    ),
                },
                now_s=stamp,
                carried_class_id="pink",
            )

        self.assertIsNone(result)

    def test_uses_recent_static_packaging_during_fresh_colored_scan(self) -> None:
        tracker = ShelfStateTracker(required_votes=3, max_observation_age_s=2.0)
        result = None
        # The fixed packaging prop was seen on arrival.  Fresh colour frames
        # arrive during the head scan, while the held object occludes the prop.
        for stamp in (30.0, 35.0, 40.0):
            result = tracker.update(
                {
                    "brown": observation("brown", (-2.63, 0.778, 0.837), stamp),
                    "packaging_box": observation(
                        "packaging_box", (-2.627, 0.692, 0.528), 15.0
                    ),
                },
                now_s=stamp,
                carried_class_id="yellow",
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.layer_contents, ("packaging_box", "brown", "EMPTY"))

    def test_task2_instruction_colour_rejects_early_wrong_colour_votes(self) -> None:
        tracker = ShelfStateTracker(required_votes=3)
        packaging = observation(
            "packaging_box", (-2.634, 0.721, 0.569), 10.0
        )

        # Reproduce the remote run: the one physical shelf box is initially
        # misclassified as pink.  Shelf ROI/layer evidence identifies the
        # instance, while task 2's instruction supplies its canonical colour.
        result = None
        for stamp in (10.0, 11.0, 12.0):
            result = tracker.update(
                {
                    "pink": observation("pink", (-2.63, 0.778, 1.168), stamp),
                    "packaging_box": packaging,
                },
                now_s=stamp,
                carried_class_id="yellow",
                expected_colored_class_id="brown",
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.colored_class_id, "brown")
        self.assertEqual(result.colored_center_world, (-2.63, 0.778, 1.168))

    def test_fails_closed_when_two_semantics_vote_for_same_layer(self) -> None:
        tracker = ShelfStateTracker(required_votes=3)
        result = None
        for index in range(4):
            stamp = 20.0 + index
            result = tracker.update(
                {
                    "yellow": observation("yellow", (-2.55, 0.78, 0.508), stamp),
                    "packaging_box": observation(
                        "packaging_box", (-2.54, 0.78, 0.530), stamp
                    ),
                },
                now_s=stamp,
                carried_class_id="pink",
            )
        self.assertIsNone(result)


    def test_strict_l1_candidate_requires_fresh_empty_confirmation(self) -> None:
        tracker = ShelfStateTracker(
            required_votes=3,
            require_empty_confirmation=True,
        )

        result = None
        for stamp in (1.0, 2.0, 3.0):
            result = tracker.update(
                {
                    "brown": observation(
                        "brown", (-2.63, 0.778, 1.166), stamp
                    ),
                    "packaging_box": observation(
                        "packaging_box", (-2.63, 0.778, 0.837), stamp
                    ),
                },
                now_s=stamp,
                carried_class_id="yellow",
            )

        self.assertIsNone(result)
        self.assertEqual(tracker.semantic_empty_candidate, 1)

        for stamp in (4.0, 5.0, 6.0):
            result = tracker.update(
                {
                    "brown": observation(
                        "brown", (-2.63, 0.778, 1.166), stamp
                    ),
                    "packaging_box": observation(
                        "packaging_box", (-2.63, 0.778, 0.837), stamp
                    ),
                    "shelf_empty": observation(
                        "shelf_empty", (-2.63, 0.778, 0.530), stamp
                    ),
                },
                now_s=stamp,
                carried_class_id="yellow",
            )

        self.assertIsNotNone(result)
        tracker.reset_empty_confirmation()
        self.assertIsNone(tracker.result())
        self.assertEqual(tracker.semantic_empty_candidate, 1)
        self.assertIn("empty=none", tracker.diagnostic_summary)

    def test_missing_colored_l1_is_observation_hint_not_shelf_state(self) -> None:
        tracker = ShelfStateTracker(
            required_votes=3,
            require_empty_confirmation=True,
        )

        result = None
        for stamp in (1.0, 2.0, 3.0):
            result = tracker.update(
                {
                    "packaging_box": observation(
                        "packaging_box", (-2.63, 0.778, 0.837), stamp
                    ),
                    "shelf_empty": observation(
                        "shelf_empty", (-2.63, 0.778, 1.166), stamp
                    ),
                },
                now_s=stamp,
                carried_class_id="yellow",
                expected_colored_class_id="brown",
            )

        self.assertIsNone(result)
        self.assertIsNone(tracker.result())
        self.assertEqual(tracker.missing_colored_layer_candidate, 1)

        for stamp in (4.0, 5.0, 6.0):
            result = tracker.update(
                {
                    "brown": observation(
                        "brown", (-2.63, 0.778, 0.530), stamp
                    ),
                    "packaging_box": observation(
                        "packaging_box", (-2.63, 0.778, 0.837), stamp
                    ),
                    "shelf_empty": observation(
                        "shelf_empty", (-2.63, 0.778, 1.166), stamp
                    ),
                },
                now_s=stamp,
                carried_class_id="yellow",
                expected_colored_class_id="brown",
            )

        self.assertIsNotNone(result)
        self.assertIsNone(tracker.missing_colored_layer_candidate)


class StableTargetCenterTrackerTests(unittest.TestCase):
    def test_task2_tracker_rejects_visible_surface_fallbacks(self) -> None:
        tracker = StableTargetCenterTracker(require_quality="mask_cloud_cuboid")
        tracker.reset(accept_after_s=10.0)
        fallback = TargetObservation(
            color="brown",
            position_world=(-2.67, 0.81, 0.84),
            received_at_s=10.2,
            quality="bbox_depth_center",
        )
        self.assertIsNone(
            tracker.update(fallback, now_s=10.2, reference_layer_z=0.84)
        )
        self.assertEqual(tracker.sample_count, 0)
        self.assertIn("quality=mask_cloud_cuboid", tracker.status())

    def test_locks_full_object_center_and_rejects_one_spatial_outlier(self) -> None:
        tracker = StableTargetCenterTracker()
        tracker.reset(accept_after_s=10.5)
        samples = [
            # This pre-settle frame must not enter the fresh observation set.
            (10.4, (-2.74, 0.90, 0.84)),
            (10.6, (-2.671, 0.814, 0.842)),
            (10.8, (-2.669, 0.816, 0.844)),
            (11.0, (-2.672, 0.815, 0.843)),
            # A colour/depth edge fit in the same broad shelf ROI.
            (11.2, (-2.742, 0.875, 0.842)),
            (11.4, (-2.670, 0.813, 0.841)),
            (11.6, (-2.673, 0.817, 0.845)),
            (11.8, (-2.668, 0.815, 0.843)),
        ]
        result = None
        for stamp, center in samples:
            result = tracker.update(
                observation("brown", center, stamp),
                now_s=stamp,
                reference_layer_z=0.837,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.sample_count, 6)
        self.assertAlmostEqual(result.center_world[0], -2.6705, places=3)
        self.assertAlmostEqual(result.center_world[1], 0.815, places=3)
        self.assertAlmostEqual(result.center_world[2], 0.843, places=3)
        self.assertLess(result.max_axis_deviation[1], 0.003)

    def test_ignores_stale_fast_duplicate_and_wrong_layer_frames(self) -> None:
        tracker = StableTargetCenterTracker()
        tracker.reset(accept_after_s=20.0)

        tracker.update(
            observation("yellow", (-2.67, 0.81, 0.84), 19.9),
            now_s=20.0,
            reference_layer_z=0.84,
        )
        tracker.update(
            observation("yellow", (-2.67, 0.81, 0.84), 20.0),
            now_s=21.0,
            reference_layer_z=0.84,
        )
        tracker.update(
            observation("yellow", (-2.67, 0.81, 0.84), 20.1),
            now_s=20.1,
            reference_layer_z=0.84,
        )
        tracker.update(
            observation("yellow", (-2.67, 0.81, 1.16), 20.3),
            now_s=20.3,
            reference_layer_z=0.84,
        )

        self.assertEqual(tracker.sample_count, 1)


class SlideHoldControllerTests(unittest.TestCase):
    def test_arm_velocity_transients_do_not_block_slide_settle(self) -> None:
        controller = SlideHoldController()
        hold = ArmCommand(
            spine_position=0.20,
            head_positions=(0.0, 0.3),
            left_arm_positions=(0.1,) * 6,
            left_gripper_position=0.0,
            right_arm_positions=(-0.1,) * 6,
            right_gripper_position=0.0,
        )
        feedback = _arm_joint_state(
            slide=0.20,
            head=hold.head_positions,
            left=hold.left_arm_positions,
            right=hold.right_arm_positions,
        )
        # The slide is stationary, but the simulator reports arm-joint
        # transients larger than FEEDBACK_VEL_TOL while the held pose settles.
        feedback.velocity = [0.0] * len(feedback.name)
        for index, name in enumerate(feedback.name):
            if name.startswith(("left_arm_joint", "right_arm_joint")):
                feedback.velocity[index] = 0.03

        controller.plan(hold, 0.20, feedback)
        _command, reached, detail = controller.update(0.0, feedback)
        self.assertFalse(reached)
        self.assertIn("slide_vel=0.000", detail)
        _command, reached, _detail = controller.update(0.10, feedback)
        self.assertFalse(reached)
        _command, reached, _detail = controller.update(0.70, feedback)
        self.assertTrue(reached)


class IntegratedExecutorWiringTests(unittest.TestCase):
    def test_task1_release_uses_grasp_latched_width_after_contact_reset(self) -> None:
        executor = Task1IntegratedExecutor(CompetitionTaskMemory())
        executor._contact._half_width = 0.109
        executor._contact._orientation = "yaw0"
        executor._capture_held_grasp_snapshot()

        # Reproduce the remote failure: later controller lifecycle work clears
        # the contact planner although the arms still hold the same box.
        executor._contact.reset()

        self.assertAlmostEqual(executor._held_release_half_width(), 0.109)
        self.assertEqual(executor._require_held_grasp_orientation(), "yaw0")

    def test_task2_transport_uses_grasp_latched_width_after_contact_reset(self) -> None:
        executor = Task2IntegratedExecutor(CompetitionTaskMemory())
        executor._contact._half_width = 0.081
        executor._contact._orientation = "yaw90"
        executor._capture_held_grasp_snapshot()

        # Transport/resource lifecycle work may reset the contact transaction;
        # the held payload geometry must remain available until release.
        executor._contact.reset()

        self.assertAlmostEqual(executor._held_half_width(), 0.081)
        self.assertEqual(executor._require_held_grasp_orientation(), "yaw90")

    def test_all_integrated_tasks_use_independent_compliant_place_lowering(self) -> None:
        memory = CompetitionTaskMemory()
        task1 = Task1IntegratedExecutor(memory)
        task2 = Task2IntegratedExecutor(memory)

        self.assertIsInstance(task1._place_lowering, CompliantSlideLoweringController)
        self.assertIsInstance(task2._place_lowering, CompliantSlideLoweringController)
        self.assertIsNot(task1._place_lowering, task2._place_lowering)
        self.assertEqual(task1.RELEASE_SUPPORT_SETTLE_S, 0.40)
        self.assertEqual(task2.RELEASE_SUPPORT_SETTLE_S, 0.40)

    def test_task1_uses_slow_staged_shelf_release(self) -> None:
        executor = Task1IntegratedExecutor(CompetitionTaskMemory())

        self.assertAlmostEqual(executor.RELEASE_SPREAD_M, 0.040)
        self.assertLess(
            executor._release.command_rate_per_s,
            1.20,
        )

    def test_task2_does_not_lift_without_bilateral_alignment(self) -> None:
        executor = Task2IntegratedExecutor(CompetitionTaskMemory())

        self.assertFalse(executor.ALLOW_SETTLED_MAX_SEARCH)

    def test_task1_records_stable_shelf_state_during_transport_updates(self) -> None:
        memory = CompetitionTaskMemory()
        memory.record_task1_origin((-0.22, 2.20, 0.84), "yellow")
        executor = Task1IntegratedExecutor(memory)
        executor.configure_instructions(
            (
                {"task": 1, "target_color": "yellow"},
                {"task": 2, "target_color": "brown"},
            )
        )

        state = None
        for stamp in (1.0, 2.0, 3.0):
            state = executor._update_shelf_state(
                ExecutionContext(
                    now_s=stamp,
                    instruction={"task": 1, "target_color": "yellow"},
                    task_index=1,
                    attempt=1,
                    target_observations={
                        "brown": observation(
                            "brown", (-2.55, 0.81, 0.837), stamp
                        ),
                        "packaging_box": observation(
                            "packaging_box", (-2.54, 0.78, 0.530), stamp
                        ),
                        "shelf_empty": observation(
                            "shelf_empty", (-2.63, 0.778, 1.166), stamp
                        ),
                    },
                )
            )

        self.assertIsNotNone(state)
        self.assertIs(memory.shelf_state, state)
        self.assertEqual(
            memory.require_task2_target_center(),
            (-2.55, 0.81, 0.837),
        )
        self.assertEqual(
            memory.require_task3_packaging_box_center(),
            (-2.54, 0.78, 0.530),
        )

    def test_task1_lowers_carried_box_only_for_unconfirmed_l1(self) -> None:
        class RecordingSlideHold:
            def __init__(self) -> None:
                self.target_slide = None

            def plan(self, hold_command, target_slide, _joint_states):
                self.target_slide = float(target_slide)
                return hold_command

        memory = CompetitionTaskMemory()
        memory.record_task1_origin((-0.22, 2.20, 0.84), "yellow")
        executor = Task1IntegratedExecutor(memory)
        executor.configure_instructions(
            (
                {"task": 1, "target_color": "yellow"},
                {"task": 2, "target_color": "brown"},
            )
        )
        executor._held_center_base = (0.70, 0.0, 0.96)
        executor._held_arm_command = ArmCommand(
            spine_position=0.30,
            head_positions=(0.0, 0.0),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=0.20,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=0.20,
        )
        executor._phase = "scan_shelf"
        executor._stage_started_s = 0.0
        recording_slide = RecordingSlideHold()
        executor._slide_hold = recording_slide

        result = None
        for stamp in (1.0, 2.0, 3.0):
            result = executor._tick_align_for_place(
                ExecutionContext(
                    now_s=stamp,
                    instruction={"task": 1, "target_color": "yellow"},
                    task_index=1,
                    attempt=1,
                    target_observations={
                        "brown": observation(
                            "brown", (-2.63, 0.778, 1.166), stamp
                        ),
                        "packaging_box": observation(
                            "packaging_box", (-2.63, 0.778, 0.837), stamp
                        ),
                    },
                )
            )

        assert result is not None
        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertEqual(executor._phase, "l1_visibility_clearance")
        self.assertAlmostEqual(recording_slide.target_slide, 0.58, places=6)
        self.assertIsNone(executor._shelf_state)
        self.assertIsNone(memory.shelf_state)
        self.assertIn("explicit empty-layer confirmation", result.message)

    def test_l1_visibility_posture_does_not_run_for_task3(self) -> None:
        class RejectingSlideHold:
            def plan(self, _hold_command, _target_slide, _joint_states):
                raise AssertionError("task 3 must not use task-1 L1 visibility")

        executor = Task1IntegratedExecutor(CompetitionTaskMemory())
        executor.task_id = 3
        executor._expected_shelf_color = "brown"
        executor._held_center_base = (0.70, 0.0, 0.96)
        executor._held_arm_command = ArmCommand(
            spine_position=0.30,
            head_positions=(0.0, 0.0),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=0.20,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=0.20,
        )
        executor._phase = "scan_shelf"
        executor._stage_started_s = 0.0
        executor._slide_hold = RejectingSlideHold()

        result = None
        for stamp in (1.0, 2.0, 3.0):
            result = executor._tick_align_for_place(
                ExecutionContext(
                    now_s=stamp,
                    instruction={"task": 3, "target_color": "yellow"},
                    task_index=3,
                    attempt=1,
                    target_observations={
                        "brown": observation(
                            "brown", (-2.63, 0.778, 1.166), stamp
                        ),
                        "packaging_box": observation(
                            "packaging_box", (-2.63, 0.778, 0.837), stamp
                        ),
                    },
                )
            )

        assert result is not None
        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertEqual(executor._phase, "scan_shelf")

    def test_task1_l1_recovery_also_handles_missing_colored_target(self) -> None:
        class RecordingSlideHold:
            def __init__(self) -> None:
                self.calls = 0

            def plan(self, hold_command, _target_slide, _joint_states):
                self.calls += 1
                return hold_command

        memory = CompetitionTaskMemory()
        memory.record_task1_origin((-0.22, 2.20, 0.84), "yellow")
        executor = Task1IntegratedExecutor(memory)
        executor.configure_instructions(
            (
                {"task": 1, "target_color": "yellow"},
                {"task": 2, "target_color": "brown"},
            )
        )
        executor._held_center_base = (0.70, 0.0, 0.96)
        executor._held_arm_command = ArmCommand(
            spine_position=0.30,
            head_positions=(0.0, 0.0),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=0.20,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=0.20,
        )
        executor._phase = "scan_shelf"
        executor._stage_started_s = 0.0
        slide = RecordingSlideHold()
        executor._slide_hold = slide

        result = None
        for stamp in (1.0, 2.0, 3.0):
            result = executor._tick_align_for_place(
                ExecutionContext(
                    now_s=stamp,
                    instruction={"task": 1, "target_color": "yellow"},
                    task_index=1,
                    attempt=1,
                    target_observations={
                        "packaging_box": observation(
                            "packaging_box", (-2.63, 0.778, 0.837), stamp
                        ),
                        "shelf_empty": observation(
                            "shelf_empty", (-2.63, 0.778, 1.166), stamp
                        ),
                    },
                )
            )

        assert result is not None
        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertEqual(executor._phase, "l1_visibility_clearance")
        self.assertEqual(slide.calls, 1)
        self.assertIn("missing colored target", result.message)
        self.assertIsNone(executor._shelf_state)
        self.assertIsNone(memory.shelf_state)

    def test_task1_direct_route_uses_project_geometry_not_legacy_turn_point(self) -> None:
        memory = CompetitionTaskMemory()
        memory.record_task1_origin((-0.22, 2.20, 0.84), "yellow")
        executor = Task1IntegratedExecutor(memory)
        executor._held_center_base = (0.70, 0.03, 1.10)
        executor._held_arm_command = ArmCommand(
            spine_position=0.30,
            head_positions=(0.0, 0.0),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=0.20,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=0.20,
        )
        executor._phase = "navigate_shelf_direct"
        fake = _DirectRouteTransfer()
        executor._direct_shelf_transfer = fake

        result = executor._tick_transport(
            ExecutionContext(
                now_s=10.0,
                instruction={"task": 1, "target_color": "yellow"},
                task_index=1,
                attempt=1,
                odometry=_odom(-0.20, 1.20, math.pi / 2.0),
            )
        )

        self.assertEqual(result.status, StageStatus.SUCCEEDED)
        assert fake.goal is not None
        self.assertNotAlmostEqual(fake.goal.x, executor.SHELF_TURN_X, places=3)
        self.assertAlmostEqual(
            executor._shelf_observation_target_y(),
            executor._shelf_tracker.geometry.shelf_xy[1],
            places=6,
        )
        self.assertEqual(
            fake.goal.source_tag,
            "integrated_task1_direct_shelf_preplace",
        )

    def test_task1_direct_plan_failure_uses_legacy_route_once(self) -> None:
        executor = Task1IntegratedExecutor(CompetitionTaskMemory())
        executor._held_center_base = (0.70, 0.03, 1.10)
        executor._held_arm_command = ArmCommand(
            spine_position=0.30,
            head_positions=(0.0, 0.0),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=0.20,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=0.20,
        )
        executor._phase = "navigate_shelf_direct"
        executor._direct_shelf_transfer = _FailingDirectRouteTransfer()
        legacy = _RunningLegacyRouteTransfer()
        executor._transfer = legacy

        result = executor._tick_transport(
            ExecutionContext(
                now_s=10.0,
                instruction={"task": 1, "target_color": "yellow"},
                task_index=1,
                attempt=1,
                odometry=_odom(-0.20, 1.20, math.pi / 2.0),
            )
        )

        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertTrue(executor._legacy_shelf_route_used)
        self.assertEqual(executor._phase, "navigate_shelf_turn_fallback")
        assert legacy.goal is not None
        self.assertEqual(
            legacy.goal.source_tag,
            "integrated_task1_legacy_shelf_turn_fallback",
        )

    def test_task1_direct_arrival_skips_three_step_lateral_alignment(self) -> None:
        executor = Task1IntegratedExecutor(CompetitionTaskMemory())
        executor._held_center_base = (0.70, 0.03, 1.10)
        executor._held_arm_command = ArmCommand(
            spine_position=0.30,
            head_positions=(0.0, 0.0),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=0.20,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=0.20,
        )
        executor._place_world = (-2.63, 0.778, 0.84)
        executor._shelf_scan_stand = shelf_observation_stand(
            executor._held_center_base,
            shelf_front_x=executor.SHELF_FRONT_X,
            shelf_y=0.778,
            center_clearance_m=executor.SHELF_SCAN_CENTER_CLEARANCE_M,
            shelf_yaw=executor.SHELF_YAW,
        )
        executor._phase = "check_place_alignment"

        result = executor._tick_align_for_place(
            ExecutionContext(
                now_s=10.0,
                instruction={"task": 1, "target_color": "yellow"},
                task_index=1,
                attempt=1,
                odometry=_odom(
                    executor._shelf_scan_stand[0],
                    executor._shelf_scan_stand[1],
                    executor.SHELF_YAW,
                ),
            )
        )

        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertEqual(executor._phase, "approach_place_final")
        self.assertIn("entering the recognized empty shelf layer", result.message)

    def test_task1_corrects_three_centimeter_lateral_place_offset(self) -> None:
        executor = Task1IntegratedExecutor(CompetitionTaskMemory())
        executor._held_center_base = (0.70, 0.03, 1.10)
        executor._held_arm_command = ArmCommand(
            spine_position=0.30,
            head_positions=(0.0, 0.0),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=0.20,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=0.20,
        )
        executor._place_world = (-2.63, 0.778, 0.84)
        executor._shelf_scan_stand = shelf_observation_stand(
            executor._held_center_base,
            shelf_front_x=executor.SHELF_FRONT_X,
            shelf_y=0.778,
            center_clearance_m=executor.SHELF_SCAN_CENTER_CLEARANCE_M,
            shelf_yaw=executor.SHELF_YAW,
        )
        executor._phase = "check_place_alignment"

        result = executor._tick_align_for_place(
            ExecutionContext(
                now_s=10.0,
                instruction={"task": 1, "target_color": "yellow"},
                task_index=1,
                attempt=1,
                odometry=_odom(
                    executor._shelf_scan_stand[0],
                    executor._shelf_scan_stand[1] - 0.03,
                    executor.SHELF_YAW,
                ),
            )
        )

        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertEqual(executor._phase, "navigate_place_lateral")
        self.assertAlmostEqual(
            executor._transfer._lateral_position_tolerance_m,
            executor.FINAL_PLACE_LATERAL_TOLERANCE_M,
            places=6,
        )
        self.assertIn("aligning laterally", result.message)

    def test_task1_uses_guided_curve_for_small_latched_place_error(self) -> None:
        executor = Task1IntegratedExecutor(CompetitionTaskMemory())
        executor._held_center_base = (0.70, 0.03, 1.10)
        executor._held_grasp_half_width = 0.08
        executor._held_arm_command = ArmCommand(
            spine_position=0.30,
            head_positions=(0.0, 0.0),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=0.20,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=0.20,
        )
        executor._place_world = (-2.63, 0.778, 0.84)
        executor._shelf_scan_stand = shelf_observation_stand(
            executor._held_center_base,
            shelf_front_x=executor.SHELF_FRONT_X,
            shelf_y=0.778,
            center_clearance_m=executor.SHELF_SCAN_CENTER_CLEARANCE_M,
            shelf_yaw=executor.SHELF_YAW,
        )
        executor._phase = "check_place_alignment"
        pose = _odom(
            executor._shelf_scan_stand[0],
            executor._shelf_scan_stand[1] - 0.03,
            executor.SHELF_YAW,
        )

        result = executor._tick_align_for_place(
            ExecutionContext(
                now_s=10.0,
                instruction={"task": 1, "target_color": "yellow"},
                task_index=1,
                attempt=1,
                odometry=pose,
            )
        )

        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertEqual(executor._phase, "approach_place_guided")
        self.assertTrue(result.controls_base)
        self.assertGreater(result.base_linear_x, 0.0)
        self.assertIn("correcting the final approach", result.message)
        gate = executor._guided_gate_stand()
        assert gate is not None
        gate_center_x = gate[0] - executor._held_center_base[0]
        self.assertAlmostEqual(
            gate_center_x,
            executor.SHELF_FRONT_X
            + executor._transfer.guided_object_center_clearance_m,
            places=6,
        )

    def test_task1_guided_switch_zero_restores_lateral_controller(self) -> None:
        executor = Task1IntegratedExecutor(CompetitionTaskMemory())
        executor.set_guided_place_approach(False)
        executor._held_center_base = (0.70, 0.03, 1.10)
        executor._held_grasp_half_width = 0.08
        executor._held_arm_command = ArmCommand(
            spine_position=0.30,
            head_positions=(0.0, 0.0),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=0.20,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=0.20,
        )
        executor._place_world = (-2.63, 0.778, 0.84)
        executor._shelf_scan_stand = shelf_observation_stand(
            executor._held_center_base,
            shelf_front_x=executor.SHELF_FRONT_X,
            shelf_y=0.778,
            center_clearance_m=executor.SHELF_SCAN_CENTER_CLEARANCE_M,
            shelf_yaw=executor.SHELF_YAW,
        )
        executor._phase = "check_place_alignment"

        result = executor._tick_align_for_place(
            ExecutionContext(
                now_s=10.0,
                instruction={"task": 1, "target_color": "yellow"},
                task_index=1,
                attempt=1,
                odometry=_odom(
                    executor._shelf_scan_stand[0],
                    executor._shelf_scan_stand[1] - 0.03,
                    executor.SHELF_YAW,
                ),
            )
        )

        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertEqual(executor._phase, "navigate_place_lateral")
        self.assertIn("aligning laterally", result.message)

    def test_task1_transport_keeps_task3_approach_phase_compatibility(self) -> None:
        executor = Task1IntegratedExecutor(CompetitionTaskMemory())
        executor._held_center_base = (0.70, 0.03, 1.10)
        executor._held_arm_command = ArmCommand(
            spine_position=0.30,
            head_positions=(0.0, 0.0),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=0.20,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=0.20,
        )
        executor._shelf_scan_stand = (-1.015, 0.808)
        executor._phase = "approach_shelf_scan"

        result = executor._tick_transport(
            ExecutionContext(
                now_s=10.0,
                instruction={"task": 3, "target_color": "pink"},
                task_index=3,
                attempt=1,
                odometry=_odom(-1.015, 0.808, executor.SHELF_YAW),
            )
        )

        self.assertEqual(result.status, StageStatus.SUCCEEDED)
        self.assertNotIn("invalid transport phase", result.message)

    def test_task1_alignment_preserves_transport_shelf_state(self) -> None:
        memory = CompetitionTaskMemory()
        executor = Task1IntegratedExecutor(memory)

        # Model a complete result collected while travelling.  The observation
        # stand must continue this same epoch instead of discarding valid
        # frames that may become occluded by the carried box after arrival.
        tracker = ShelfStateTracker(required_votes=2)
        transport_state = None
        for stamp in (1.0, 2.0):
            transport_state = tracker.update(
                {
                    "pink": observation("pink", (-2.55, 0.81, 0.837), stamp),
                    "packaging_box": observation(
                        "packaging_box", (-2.54, 0.78, 0.530), stamp
                    ),
                },
                now_s=stamp,
                carried_class_id="brown",
            )
        self.assertIsNotNone(transport_state)
        assert transport_state is not None
        executor._shelf_tracker = tracker
        executor._shelf_state = transport_state
        memory.record_shelf_state(transport_state)

        context = ExecutionContext(
            now_s=2.0,
            instruction={},
            task_index=1,
            attempt=1,
        )
        executor.enter_stage(TaskStage.ALIGN_FOR_PLACE, context)

        self.assertIs(executor._shelf_state, transport_state)
        self.assertEqual(executor._shelf_tracker.frames_used, 2)
        self.assertIs(memory.shelf_state, transport_state)
        self.assertIsNotNone(memory.shelf_empty_center_world)

    def test_task1_transport_starts_one_fresh_shelf_epoch(self) -> None:
        memory = CompetitionTaskMemory()
        executor = Task1IntegratedExecutor(memory)
        tracker = ShelfStateTracker(required_votes=2)
        state = None
        for stamp in (1.0, 2.0):
            state = tracker.update(
                {
                    "pink": observation("pink", (-2.55, 0.81, 0.837), stamp),
                    "packaging_box": observation(
                        "packaging_box", (-2.54, 0.78, 0.530), stamp
                    ),
                },
                now_s=stamp,
                carried_class_id="brown",
            )
        assert state is not None
        executor._shelf_tracker = tracker
        executor._shelf_state = state
        memory.record_shelf_state(state)

        executor.enter_stage(
            TaskStage.TRANSPORT,
            ExecutionContext(
                now_s=3.0,
                instruction={},
                task_index=1,
                attempt=1,
            ),
        )

        self.assertIsNone(executor._shelf_state)
        self.assertEqual(executor._shelf_tracker.frames_used, 0)
        self.assertIsNone(memory.shelf_state)
        self.assertIsNone(memory.shelf_empty_center_world)

    def test_task12_mode_uses_one_shared_memory_instance(self) -> None:
        executors = build_task_executors("task12_full")
        self.assertIsInstance(executors[1], Task1IntegratedExecutor)
        self.assertIsInstance(executors[2], Task2IntegratedExecutor)
        self.assertIs(executors[1]._memory, executors[2]._memory)

    def test_controller_configures_task1_from_task2_instruction(self) -> None:
        executors = build_task_executors("task123_full")
        controller = CompetitionController(executors)
        controller.configure(
            [
                {"task": 1, "target_color": "yellow"},
                {"task": 2, "target_color": "brown"},
                {"task": 3, "target_color": "pink"},
            ]
        )
        self.assertEqual(executors[1]._expected_shelf_color, "brown")

    def test_wrong_detector_colour_flows_to_task2_as_instructed_identity(self) -> None:
        memory = CompetitionTaskMemory()
        tracker = ShelfStateTracker(required_votes=3)
        state = None
        for stamp in (10.0, 11.0, 12.0):
            state = tracker.update(
                {
                    # Remote failure: the brown shelf box was initially
                    # reported as pink at the far observation stand.
                    "pink": observation("pink", (-2.63, 0.778, 1.168), stamp),
                    "packaging_box": observation(
                        "packaging_box", (-2.634, 0.721, 0.569), 10.0
                    ),
                },
                now_s=stamp,
                carried_class_id="yellow",
                expected_colored_class_id="brown",
            )
        assert state is not None
        memory.record_shelf_state(state)

        task2 = Task2IntegratedExecutor(memory)
        color, center = task2._task2_target(
            ExecutionContext(
                now_s=20.0,
                instruction={"task": 2, "target_color": "brown"},
                task_index=2,
                attempt=1,
            )
        )
        self.assertEqual(color, "brown")
        self.assertEqual(center, (-2.63, 0.778, 1.168))

    def test_shared_memory_keeps_only_three_shelf_centers(self) -> None:
        tracker = ShelfStateTracker(required_votes=3)
        state = None
        for index in range(3):
            stamp = 40.0 + index
            state = tracker.update(
                {
                    "brown": observation("brown", (-2.55, 0.81, 0.837), stamp),
                    "packaging_box": observation(
                        "packaging_box", (-2.54, 0.78, 0.530), stamp
                    ),
                },
                now_s=stamp,
                carried_class_id="pink",
            )
        assert state is not None
        memory = CompetitionTaskMemory()
        memory.record_shelf_state(state)
        self.assertEqual(memory.require_empty_shelf_center(), state.empty_shelf_center_world)
        self.assertEqual(memory.require_task2_target_center(), state.task2_target_center_world)
        self.assertEqual(
            memory.require_task3_packaging_box_center(),
            state.task3_packaging_box_center_world,
        )
        self.assertIsNone(getattr(memory, "material_box_center_world", None))

    def test_task2_uses_narrow_shelf_pregrasp_controller(self) -> None:
        executor = Task2IntegratedExecutor(CompetitionTaskMemory())
        self.assertIsInstance(executor._pregrasp, ShelfOpenPregraspController)
        self.assertAlmostEqual(executor._pregrasp.half_width, 0.20, places=6)
        self.assertEqual(executor.SHELF_LATERAL_ALIGNMENT_TIMEOUT_S, 40.0)

    def test_task2_corrects_yaw0_shelf_fit_along_shelf_normal(self) -> None:
        executor = Task2IntegratedExecutor(CompetitionTaskMemory())
        corrected, offset = executor._correct_shelf_center_orientation(
            (-2.719, 0.826, 1.139),
            "yaw0",
        )
        self.assertAlmostEqual(offset, 0.04, places=6)
        self.assertAlmostEqual(corrected[0], -2.679, places=6)
        self.assertAlmostEqual(corrected[1], 0.826, places=6)
        self.assertAlmostEqual(corrected[2], 1.139, places=6)
        unchanged, offset = executor._correct_shelf_center_orientation(
            (-2.679, 0.826, 1.139),
            "yaw90",
        )
        self.assertEqual(offset, 0.0)
        self.assertEqual(unchanged, (-2.679, 0.826, 1.139))

    def test_task2_keeps_arms_retracted_until_final_grab_stand(self) -> None:
        executor = Task2IntegratedExecutor(CompetitionTaskMemory())
        context = ExecutionContext(
            now_s=12.0,
            instruction={"task": 2, "target_color": "yellow"},
            task_index=2,
            attempt=1,
        )
        executor.enter_stage(TaskStage.ALIGN_FOR_PICK, context)
        self.assertEqual(executor._phase, "stage_camera")
        self.assertFalse(executor._pregrasp.planned)

    def test_place_stand_preserves_held_object_transform(self) -> None:
        stand = stand_from_held_center(
            (-2.63, 0.778, 0.837),
            (0.65, 0.02, 0.95),
            3.141592653589793,
        )
        self.assertAlmostEqual(stand[0], -1.98, places=6)
        self.assertAlmostEqual(stand[1], 0.798, places=6)

    def test_shelf_scan_stand_keeps_carried_center_outside_front(self) -> None:
        held = (0.70, -0.025, 0.984)
        stand = shelf_observation_stand(
            held,
            shelf_front_x=-2.465,
            shelf_y=0.85,
            center_clearance_m=0.18,
        )
        # Facing west, the held center is base + R(pi) * held.
        carried_center_x = stand[0] - held[0]
        carried_center_y = stand[1] - held[1]
        self.assertAlmostEqual(carried_center_x, -2.285, places=6)
        self.assertAlmostEqual(carried_center_y, 0.85, places=6)
        self.assertGreater(carried_center_x, -2.465)

    def test_straight_shelf_delta_is_forward_when_target_is_west(self) -> None:
        forward, lateral = target_delta_in_heading(
            (-1.30, 0.825, 3.141592653589793),
            (-1.585, 0.825),
            3.141592653589793,
        )
        self.assertAlmostEqual(forward, 0.285, places=6)
        self.assertAlmostEqual(lateral, 0.0, places=6)

    def test_guided_advance_converges_with_weak_plant_response(self) -> None:
        motion = TransferMotion()
        held = HeldObjectGeometry((0.70, 0.02, 0.90), 0.08, source="test")
        start = (-0.95, 0.85, math.pi + 0.04)
        gate = (-1.535, 0.805)
        self.assertTrue(
            motion.begin_guided_advance(
                gate, math.pi, _odom(*start), 0.0, held_geometry=held
            )
        )

        x, y, yaw = start
        status = NavigationStatus.NAVIGATING
        maximum_yaw_offset = 0.0
        for step in range(700):
            now_s = step * 0.05
            status, command, _detail = motion.tick_guided_advance(
                _odom(x, y, yaw), now_s
            )
            if status is NavigationStatus.GOAL_REACHED:
                break
            self.assertIs(status, NavigationStatus.NAVIGATING)
            # Stress the same weak response envelope used by the offline audit.
            linear = 0.30 * command[0]
            angular = 0.25 * command[1] + 0.003
            yaw += angular * 0.05
            x += linear * math.cos(yaw) * 0.05
            y += linear * math.sin(yaw) * 0.05
            maximum_yaw_offset = max(
                maximum_yaw_offset, abs(_wrap_test_angle(yaw - math.pi))
            )

        self.assertIs(status, NavigationStatus.GOAL_REACHED)
        self.assertLess(maximum_yaw_offset, 0.20)
        self.assertLess(abs(y - gate[1]), motion.GUIDED_POSITION_TOLERANCE_M)

    def test_guided_advance_rejects_out_of_domain_or_shelf_overlap(self) -> None:
        motion = TransferMotion()
        held = HeldObjectGeometry((0.70, 0.0, 0.90), 0.08, source="test")
        start = _odom(-0.95, 0.85, math.pi)
        self.assertFalse(
            motion.begin_guided_advance(
                (-1.50, 0.79), math.pi, start, 0.0, held_geometry=held
            )
        )
        self.assertFalse(
            motion.begin_guided_advance(
                (-1.75, 0.82), math.pi, start, 0.0, held_geometry=held
            )
        )

    def test_transfer_advance_holds_heading_and_finishes_by_odometry(self) -> None:
        motion = TransferMotion()
        self.assertTrue(motion.begin_advance(_odom(-1.30, 0.85, 3.141592653589793), 0.28))
        done, command, _detail = motion.tick_advance(
            _odom(-1.40, 0.85, 3.141592653589793)
        )
        self.assertFalse(done)
        self.assertGreater(command[0], 0.0)
        self.assertAlmostEqual(command[1], 0.0, places=6)
        done, command, _detail = motion.tick_advance(
            _odom(-1.58, 0.85, 3.141592653589793)
        )
        self.assertTrue(done)
        self.assertEqual(command, (0.0, 0.0))

    def test_transfer_retreat_corrects_explicit_heading_before_translation(self) -> None:
        motion = TransferMotion()
        start_yaw = math.pi - 0.12
        self.assertTrue(
            motion.begin_retreat(
                _odom(-1.28, 0.78, start_yaw),
                0.30,
                heading_yaw=math.pi,
            )
        )
        done, command, _detail = motion.tick_retreat(
            _odom(-1.28, 0.78, start_yaw)
        )
        self.assertFalse(done)
        self.assertEqual(command[0], 0.0)
        self.assertGreater(command[1], 0.0)

        done, command, _detail = motion.tick_retreat(
            _odom(-1.28, 0.78, math.pi)
        )
        self.assertFalse(done)
        self.assertLess(command[0], 0.0)
        done, command, _detail = motion.tick_retreat(
            _odom(-0.98, 0.78, math.pi)
        )
        self.assertTrue(done)
        self.assertEqual(command, (0.0, 0.0))

    def test_transfer_lateral_alignment_rotates_drives_then_restores_yaw(self) -> None:
        motion = TransferMotion()
        final_yaw = math.pi
        self.assertTrue(
            motion.begin_lateral_alignment(
                (-1.30, 1.00),
                final_yaw,
                _odom(-1.30, 0.85, final_yaw),
                0.0,
            )
        )

        status, command, detail = motion.tick_lateral_alignment(
            _odom(-1.30, 0.85, final_yaw), 0.05
        )
        self.assertEqual(status.value, "navigating")
        self.assertEqual(command[0], 0.0)
        self.assertIn("rotating toward shelf-front", detail)

        status, command, _detail = motion.tick_lateral_alignment(
            _odom(-1.30, 0.85, math.pi / 2.0), 0.20
        )
        self.assertEqual(status.value, "navigating")
        self.assertGreater(command[0], 0.0)

        status, command, detail = motion.tick_lateral_alignment(
            _odom(-1.30, 1.00, math.pi / 2.0), 2.0
        )
        self.assertEqual(status.value, "navigating")
        self.assertEqual(command[0], 0.0)
        self.assertIn("restoring shelf-facing yaw", detail)

        status, command, _detail = motion.tick_lateral_alignment(
            _odom(-1.30, 1.00, final_yaw), 3.0
        )
        self.assertEqual(status, NavigationStatus.GOAL_REACHED)
        self.assertEqual(command, (0.0, 0.0))

    def test_transfer_lateral_alignment_accepts_task_specific_yaw_tolerance(self) -> None:
        motion = TransferMotion()
        self.assertTrue(
            motion.begin_lateral_alignment(
                (-1.30, 0.85),
                math.pi,
                _odom(-1.30, 0.85, math.pi - 0.02),
                0.0,
                position_tolerance_m=0.008,
                yaw_tolerance_rad=0.015,
            )
        )
        status, command, detail = motion.tick_lateral_alignment(
            _odom(-1.30, 0.85, math.pi - 0.02), 0.05
        )
        self.assertEqual(status, NavigationStatus.NAVIGATING)
        self.assertEqual(command[0], 0.0)
        self.assertIn("restoring shelf-facing yaw", detail)

    def test_transfer_lateral_alignment_can_reverse_away_from_payload_wall(self) -> None:
        motion = TransferMotion()
        self.assertTrue(
            motion.begin_lateral_alignment(
                (-1.30, 0.85),
                math.pi,
                _odom(-1.30, 1.20, math.pi),
                0.0,
                drive_in_reverse=True,
            )
        )

        status, command, detail = motion.tick_lateral_alignment(
            _odom(-1.30, 1.20, math.pi), 0.05
        )
        self.assertEqual(status, NavigationStatus.NAVIGATING)
        self.assertEqual(command[0], 0.0)
        self.assertIn("rotating toward shelf-front", detail)

        status, command, _detail = motion.tick_lateral_alignment(
            _odom(-1.30, 1.20, math.pi / 2.0), 0.20
        )
        self.assertEqual(status, NavigationStatus.NAVIGATING)
        self.assertLess(command[0], 0.0)

    def test_transfer_lateral_alignment_keeps_default_timeout(self) -> None:
        motion = TransferMotion()
        self.assertTrue(
            motion.begin_lateral_alignment(
                (-1.30, 1.00),
                math.pi,
                _odom(-1.30, 0.85, math.pi),
                0.0,
            )
        )
        status, _command, detail = motion.tick_lateral_alignment(
            _odom(-1.30, 0.85, math.pi), 30.1
        )
        self.assertEqual(status, NavigationStatus.FAILED)
        self.assertIn("limit=30.0s", detail)

    def test_transfer_lateral_alignment_accepts_task_specific_timeout(self) -> None:
        motion = TransferMotion()
        self.assertTrue(
            motion.begin_lateral_alignment(
                (-1.30, 1.00),
                math.pi,
                _odom(-1.30, 0.85, math.pi),
                0.0,
                timeout_s=35.0,
            )
        )
        status, command, detail = motion.tick_lateral_alignment(
            _odom(-1.30, 0.85, math.pi), 20.1
        )
        self.assertEqual(status, NavigationStatus.NAVIGATING)
        self.assertEqual(command[0], 0.0)
        self.assertIn("rotating toward shelf-front", detail)

    def test_arm_retract_targets_neutral_posture_and_waits_for_stability(self) -> None:
        controller = ArmRetractController()
        hold = ArmCommand(
            spine_position=0.14,
            head_positions=(0.12, 0.20),
            left_arm_positions=(0.4, 0.3, 0.2, 0.1, -0.1, -0.2),
            left_gripper_position=1.0,
            right_arm_positions=(-0.4, -0.3, -0.2, -0.1, 0.1, 0.2),
            right_gripper_position=1.0,
        )
        feedback = _arm_joint_state(
            slide=0.14,
            head=(0.12, 0.20),
            left=hold.left_arm_positions,
            right=hold.right_arm_positions,
        )
        command = controller.plan(hold, feedback)
        self.assertAlmostEqual(command.spine_position, 0.14, places=6)
        self.assertEqual(command.left_arm_positions, hold.left_arm_positions)

        command, reached, detail = controller.update(0.0, feedback)
        self.assertFalse(reached)
        self.assertIn("transport retract", detail)
        self.assertLess(max(abs(value) for value in command.left_arm_positions), 0.4)

        neutral = _arm_joint_state(
            slide=0.10,
            head=(0.12, 0.20),
            left=(0.0,) * 6,
            right=(0.0,) * 6,
        )
        _command, reached, _detail = controller.update(0.10, neutral)
        self.assertFalse(reached)
        _command, reached, _detail = controller.update(0.70, neutral)
        self.assertFalse(reached)
        _command, reached, _detail = controller.update(1.30, neutral)
        self.assertTrue(reached)

    def test_task2_raises_to_physical_top_before_table_navigation(self) -> None:
        class RecordingSlideHold:
            def __init__(self) -> None:
                self.target_slide = None
                self.command = None

            @property
            def planned(self) -> bool:
                return self.target_slide is not None

            def plan(self, hold_command, target_slide, _joint_states):
                self.target_slide = float(target_slide)
                self.command = hold_command
                return hold_command

            def update(self, _now_s, _joint_states):
                return self.command, False, "waiting for physical top"

        memory = CompetitionTaskMemory(
            task1_origin_world=(-0.20, 2.30, 0.84),
            task1_color="brown",
        )
        executor = Task2IntegratedExecutor(memory)
        hold = ArmCommand(
            spine_position=0.714,
            head_positions=(0.0, 0.45),
            left_arm_positions=(0.0,) * 6,
            left_gripper_position=1.0,
            right_arm_positions=(0.0,) * 6,
            right_gripper_position=1.0,
        )
        context = ExecutionContext(
            now_s=10.0,
            instruction={},
            task_index=2,
            attempt=1,
            odometry=_odom(-2.10, 0.85, math.pi),
            joint_states=_arm_joint_state(
                slide=hold.spine_position,
                head=hold.head_positions,
                left=hold.left_arm_positions,
                right=hold.right_arm_positions,
            ),
        )
        executor.enter_stage(TaskStage.TRANSPORT, context)
        executor._phase = "lift_for_table_transport"
        # The shelf retreat may already have consumed more than the old
        # stage-wide 25 s deadline.  The maximum-height slide must use its own
        # freshly started phase timer instead of failing immediately.
        executor._stage_started_s = -100.0
        executor._phase_started_s = context.now_s
        executor._held_arm_command = hold
        executor._held_center_base = (0.82, 0.0, 0.58)
        recording_slide = RecordingSlideHold()
        executor._slide_hold = recording_slide

        result = executor.tick(TaskStage.TRANSPORT, context)

        self.assertEqual(recording_slide.target_slide, SPINE_MIN)
        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertFalse(result.controls_base)
        self.assertIn("maximum transport height", result.message)

    def test_task2_scales_long_table_column_reverse_timeout(self) -> None:
        class NeverFinishedTransfer:
            def begin_retreat(self, _odometry, _distance, *, heading_yaw=None):
                self.heading_yaw = heading_yaw
                return True

            def tick_retreat(self, _odometry):
                return False, (-0.04, 0.0), "retreating straight; remaining=0.156 m"

        class AlwaysSafeEnvelope:
            def check_rotation(self, *_args):
                return SimpleNamespace(safe=True, detail="safe")

            def check_fixed_heading_translation(self, *_args):
                return SimpleNamespace(safe=True, detail="safe")

        memory = CompetitionTaskMemory(
            task1_origin_world=(-0.174, 2.20, 0.84),
            task1_color="brown",
        )
        executor = Task2IntegratedExecutor(memory)
        hold = ArmCommand(
            spine_position=-0.04,
            head_positions=(0.0, 0.16),
            left_arm_positions=(0.4,) * 6,
            left_gripper_position=0.20,
            right_arm_positions=(-0.4,) * 6,
            right_gripper_position=0.20,
        )
        context = ExecutionContext(
            now_s=0.0,
            instruction={},
            task_index=2,
            attempt=1,
            odometry=_odom(-2.05, 0.77, math.pi),
        )
        executor.enter_stage(TaskStage.TRANSPORT, context)
        executor._phase = "reverse_to_table_column"
        executor._held_arm_command = hold
        executor._held_center_base = (0.82, 0.0, 1.30)
        executor._held_half_width = lambda: 0.10
        executor._carried_envelope = AlwaysSafeEnvelope()
        executor._transfer = NeverFinishedTransfer()
        executor._guard_carried_command = lambda _context, _command: (True, "safe")

        result = executor.tick(TaskStage.TRANSPORT, context)

        self.assertEqual(result.status, StageStatus.RUNNING, result.message)
        self.assertGreater(executor._table_column_reverse_timeout_s, 30.0)
        context = replace(context, now_s=31.0)
        result = executor.tick(TaskStage.TRANSPORT, context)
        self.assertEqual(result.status, StageStatus.RUNNING)
        context = replace(
            context,
            now_s=executor._table_column_reverse_timeout_s + 0.1,
        )
        result = executor.tick(TaskStage.TRANSPORT, context)
        self.assertEqual(result.status, StageStatus.BLOCKED)
        self.assertIn("limit=60.0s", result.message)

    def test_task1_arm_retract_timeout_starts_after_shelf_retreat(self) -> None:
        class NeverReachedArmRetract:
            def __init__(self) -> None:
                self.planned = False
                self.command = None

            def plan(self, hold_command, _joint_states):
                self.planned = True
                self.command = hold_command
                return hold_command

            def update(self, _now_s, _joint_states):
                return self.command, False, "still retracting"

        executor = Task1IntegratedExecutor(CompetitionTaskMemory())
        hold = ArmCommand(
            spine_position=0.70,
            head_positions=(0.0, 0.16),
            left_arm_positions=(0.4,) * 6,
            left_gripper_position=0.20,
            right_arm_positions=(-0.4,) * 6,
            right_gripper_position=0.20,
        )
        context = ExecutionContext(
            now_s=100.0,
            instruction={"task": 1, "target_color": "brown"},
            task_index=1,
            attempt=1,
        )
        executor.enter_stage(TaskStage.RETURN_TO_END, context)
        executor._stage_started_s = 0.0
        executor._phase = "retract_arms"
        executor._held_arm_command = hold
        executor._arm_retract = NeverReachedArmRetract()

        result = executor.tick(TaskStage.RETURN_TO_END, context)

        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertIn("retracting arms", result.message)
        self.assertAlmostEqual(executor._phase_started_s, 100.0, places=6)

    def test_task2_retracts_arms_after_table_retreat_before_end_navigation(self) -> None:
        memory = CompetitionTaskMemory(
            task1_origin_world=(-0.20, 2.30, 0.84),
            task1_color="brown",
        )
        executor = Task2IntegratedExecutor(memory)
        hold = ArmCommand(
            spine_position=0.14,
            head_positions=(0.12, 0.20),
            left_arm_positions=(0.4, 0.3, 0.2, 0.1, -0.1, -0.2),
            left_gripper_position=0.0,
            right_arm_positions=(-0.4, -0.3, -0.2, -0.1, 0.1, 0.2),
            right_gripper_position=0.0,
        )
        context = ExecutionContext(
            now_s=10.0,
            instruction={},
            task_index=2,
            attempt=1,
            odometry=_odom(-0.70, 0.55, math.pi / 2.0),
            joint_states=_arm_joint_state(
                slide=hold.spine_position,
                head=hold.head_positions,
                left=hold.left_arm_positions,
                right=hold.right_arm_positions,
            ),
        )
        executor.enter_stage(TaskStage.RETURN_TO_END, context)
        executor._held_arm_command = hold
        # Simulate that the base retreat has already completed.  The first
        # return tick must command the retract controller and must not start
        # end-zone navigation yet.
        executor._phase = "retract_arms"
        result = executor.tick(TaskStage.RETURN_TO_END, context)
        self.assertEqual(result.status, StageStatus.RUNNING)
        self.assertIn("retracting arms", result.message)
        self.assertFalse(result.controls_base)
        self.assertIsNotNone(result.arm_command)
        assert result.arm_command is not None
        self.assertLess(
            max(abs(value) for value in result.arm_command.left_arm_positions),
            max(abs(value) for value in hold.left_arm_positions),
        )

    def test_held_transport_starts_from_preloaded_command_and_keeps_grippers(self) -> None:
        class RecordingKdl:
            def __init__(self) -> None:
                self.calls = []

            def inverse_kinematics(
                self, *, T_left, T_right, ref_pos, target_height
            ):
                self.calls.append(
                    (T_left.copy(), T_right.copy(), tuple(ref_pos), target_height)
                )
                return [list(ref_pos)]

        kdl = RecordingKdl()
        controller = HeldTransportController(kdl=kdl)
        hold = ArmCommand(
            spine_position=SPINE_MIN,
            head_positions=(0.11, 0.42),
            left_arm_positions=(0.3, -0.2, 0.4, 0.1, -0.3, 0.2),
            left_gripper_position=0.83,
            right_arm_positions=(-0.3, 0.2, -0.4, -0.1, 0.3, -0.2),
            right_gripper_position=0.79,
        )

        first = controller.plan(hold, (0.76, 0.03, 1.34), 0.118)

        self.assertEqual(first, hold)
        self.assertEqual(controller.waypoint_count, 4)
        self.assertEqual(len(kdl.calls), controller.waypoint_count)
        self.assertAlmostEqual(kdl.calls[0][3], SPINE_MIN, places=6)
        self.assertAlmostEqual(first.left_gripper_position, 0.83, places=6)
        self.assertAlmostEqual(first.right_gripper_position, 0.79, places=6)
        self.assertEqual(controller.target_center_base, (0.5, 0.0, 1.34))

    def test_held_transport_supports_bounded_forward_insertion(self) -> None:
        class RecordingKdl:
            def __init__(self) -> None:
                self.calls = []

            def inverse_kinematics(
                self, *, T_left, T_right, ref_pos, target_height
            ):
                self.calls.append(
                    (T_left.copy(), T_right.copy(), tuple(ref_pos), target_height)
                )
                return [list(ref_pos)]

        kdl = RecordingKdl()
        controller = HeldTransportController(
            allow_extension=True,
            max_translation_m=0.22,
            kdl=kdl,
        )
        hold = ArmCommand(
            spine_position=SPINE_MIN,
            head_positions=(0.11, 0.42),
            left_arm_positions=(0.3, -0.2, 0.4, 0.1, -0.3, 0.2),
            left_gripper_position=0.83,
            right_arm_positions=(-0.3, 0.2, -0.4, -0.1, 0.3, -0.2),
            right_gripper_position=0.79,
        )

        first = controller.plan(
            hold,
            (0.72, 0.02, 1.10),
            0.118,
            target_center_base=(0.88, 0.0, 1.10),
        )

        self.assertEqual(first, hold)
        self.assertEqual(controller.waypoint_count, 4)
        self.assertEqual(len(kdl.calls), 4)
        self.assertEqual(controller.target_center_base, (0.88, 0.0, 1.10))
        self.assertAlmostEqual(first.left_gripper_position, 0.83, places=6)
        self.assertAlmostEqual(first.right_gripper_position, 0.79, places=6)

    def test_held_transport_rejects_insertion_beyond_bound(self) -> None:
        class IdentityKdl:
            def inverse_kinematics(
                self, *, T_left, T_right, ref_pos, target_height
            ):
                return [list(ref_pos)]

        controller = HeldTransportController(
            allow_extension=True,
            max_translation_m=0.22,
            kdl=IdentityKdl(),
        )
        hold = ArmCommand(
            spine_position=SPINE_MIN,
            head_positions=(0.11, 0.42),
            left_arm_positions=(0.3, -0.2, 0.4, 0.1, -0.3, 0.2),
            left_gripper_position=0.83,
            right_arm_positions=(-0.3, 0.2, -0.4, -0.1, 0.3, -0.2),
            right_gripper_position=0.79,
        )

        with self.assertRaises(PregraspPlanningError):
            controller.plan(
                hold,
                (0.72, 0.0, 1.10),
                0.118,
                target_center_base=(0.96, 0.0, 1.10),
            )

    def test_relative_release_keeps_achieved_center_and_spine(self) -> None:
        class RecordingKdl:
            def __init__(self) -> None:
                self.calls = []

            def inverse_kinematics(
                self, *, T_left, T_right, ref_pos, target_height
            ):
                self.calls.append(
                    (T_left.copy(), T_right.copy(), tuple(ref_pos), target_height)
                )
                return [list(ref_pos)]

        kdl = RecordingKdl()
        controller = ReleaseSpreadController(kdl=kdl)
        hold = ArmCommand(
            spine_position=0.804,
            head_positions=(0.0, 0.45),
            left_arm_positions=(0.1,) * 6,
            left_gripper_position=1.0,
            right_arm_positions=(-0.1,) * 6,
            right_gripper_position=1.0,
        )
        feedback = _arm_joint_state(
            slide=0.804,
            head=hold.head_positions,
            left=hold.left_arm_positions,
            right=hold.right_arm_positions,
        )

        command = controller.plan_from_held(
            hold,
            (0.897, -0.036, 0.498),
            feedback,
            half_width=0.114,
        )

        self.assertEqual(controller.target_base, (0.897, -0.036, 0.498))
        self.assertAlmostEqual(command.spine_position, 0.804, places=6)
        self.assertEqual(len(kdl.calls), 1)
        left_target, right_target, _reference, target_height = kdl.calls[0]
        self.assertAlmostEqual(target_height, 0.804, places=6)
        self.assertAlmostEqual(left_target[0, 3], 0.917, places=6)
        self.assertAlmostEqual(
            left_target[1, 3] - right_target[1, 3],
            2.0 * 0.114,
            places=6,
        )

    def test_relative_release_preserves_task_specific_shallow_fore_aft_center(
        self,
    ) -> None:
        class RecordingKdl:
            def __init__(self) -> None:
                self.left_target = None

            def inverse_kinematics(
                self, *, T_left, T_right, ref_pos, target_height
            ):
                del T_right, target_height
                self.left_target = T_left.copy()
                return [list(ref_pos)]

        kdl = RecordingKdl()
        controller = ReleaseSpreadController(
            kdl=kdl,
            center_backoff_x=-0.010,
        )
        hold = ArmCommand(
            spine_position=0.804,
            head_positions=(0.0, 0.45),
            left_arm_positions=(0.1,) * 6,
            left_gripper_position=1.0,
            right_arm_positions=(-0.1,) * 6,
            right_gripper_position=1.0,
        )
        feedback = _arm_joint_state(
            slide=0.804,
            head=hold.head_positions,
            left=hold.left_arm_positions,
            right=hold.right_arm_positions,
        )

        controller.plan_from_held(
            hold,
            (0.897, -0.036, 0.498),
            feedback,
            half_width=0.114,
        )

        self.assertAlmostEqual(controller.center_backoff_x, -0.010, places=6)
        self.assertAlmostEqual(kdl.left_target[0, 3], 0.907, places=6)

    def test_carried_envelope_rejects_extended_direct_turn(self) -> None:
        checker = CarriedEnvelopeChecker()
        motion = TransferMotion()
        start = (-1.28, 0.78, math.pi)
        old_goal = NavigationGoal(
            -0.18,
            1.09,
            math.pi / 2.0,
            0.08,
            0.07,
            0.0,
            NavigationSegment.NAV_TABLE,
            "old_direct_table_mid",
        )
        self.assertTrue(motion.begin_navigation(old_goal, _odom(*start)))
        check = checker.check_path(
            start,
            motion.navigation_path,
            old_goal.yaw,
            (0.75, 0.0, 1.34),
            0.118,
        )
        self.assertFalse(check.safe)

    def test_extended_hold_segmented_task2_routes_pass_carried_envelope(self) -> None:
        checker = CarriedEnvelopeChecker()
        held = (0.631, -0.014, 1.34)
        half_width = 0.118
        start = (-1.28, 0.78, math.pi)
        # Both randomized task-1 source slots.  The base first reverses east
        # while still facing west, turns only west -> north at the destination
        # column, then advances north.  The arms and box remain fully extended.
        for place in ((-1.0, 2.20, 0.84), (-0.18, 2.20, 0.84)):
            stand_x, stand_y = stand_from_held_center(
                place, held, math.pi / 2.0
            )
            reverse_end = (stand_x, start[1])
            reverse = checker.check_fixed_heading_translation(
                start,
                reverse_end,
                held,
                half_width,
            )
            self.assertTrue(reverse.safe, msg=f"reverse: {reverse.detail}")

            rotation_pose = (stand_x, start[1], math.pi)
            rotation = checker.check_rotation(
                rotation_pose,
                math.pi / 2.0,
                held,
                half_width,
            )
            self.assertTrue(rotation.safe, msg=f"rotation: {rotation.detail}")

            entry_y = min(1.35, stand_y - 0.25)
            advance = checker.check_fixed_heading_translation(
                (stand_x, start[1], math.pi / 2.0),
                (stand_x, entry_y),
                held,
                half_width,
            )
            self.assertTrue(advance.safe, msg=f"advance: {advance.detail}")

            entry_pose = (stand_x, entry_y, math.pi / 2.0)
            final_check = checker.check_fixed_heading_translation(
                entry_pose,
                (stand_x, stand_y),
                held,
                half_width,
            )
            self.assertTrue(
                final_check.safe,
                msg=f"final placement: {final_check.detail}",
            )


class _DirectRouteTransfer:
    def __init__(self) -> None:
        self.goal = None

    def reset(self) -> None:
        return None

    def begin_navigation(self, goal, _odometry, **_kwargs) -> bool:
        self.goal = goal
        return True

    def tick_navigation(self, _odometry, _now_s):
        return NavigationStatus.GOAL_REACHED, (0.0, 0.0), "test goal reached"


class _FailingDirectRouteTransfer:
    def reset(self) -> None:
        return None

    def begin_navigation(self, _goal, _odometry, **_kwargs) -> bool:
        return False


class _RunningLegacyRouteTransfer(_DirectRouteTransfer):
    def tick_navigation(self, _odometry, _now_s):
        return NavigationStatus.NAVIGATING, (0.05, 0.10), "legacy running"


def _wrap_test_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _odom(x: float, y: float, yaw: float):
    half = 0.5 * yaw
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y),
                orientation=SimpleNamespace(
                    x=0.0,
                    y=0.0,
                    z=math.sin(half),
                    w=math.cos(half),
                ),
            )
        )
    )


def _arm_joint_state(*, slide: float, head, left, right):
    names = [
        "slide_joint",
        "head_yaw_joint",
        "head_pitch_joint",
        *(f"left_arm_joint{index}" for index in range(1, 7)),
        "left_arm_eef_gripper_joint",
        *(f"right_arm_joint{index}" for index in range(1, 7)),
        "right_arm_eef_gripper_joint",
    ]
    positions = [slide, *head, *left, 1.0, *right, 1.0]
    return SimpleNamespace(
        name=names,
        position=positions,
        velocity=[0.0] * len(names),
        effort=[0.0] * len(names),
    )


if __name__ == "__main__":
    unittest.main()
