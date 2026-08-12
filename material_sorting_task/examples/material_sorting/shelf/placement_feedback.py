"""Effort-aware incremental lowering using only official Server interfaces.

The official offline Server publishes ``sensor_msgs/msg/JointState`` on
``/joint_states``.  Its effort vector contains the slide, both six-joint arms,
and the remaining head/gripper actuators.  This module consumes the joint names
rather than relying on vector offsets and falls back to the existing geometric
lowering when effort is unavailable.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any

from control_types import ArmCommand
from desktop_grasp.pregrasp_core import (
    PregraspInputError,
    PregraspPlanningError,
    _effort_map,
)
from shelf.manipulation import SlideHoldController


class PlacementContactMonitor:
    BASELINE_TIME_S = 0.40
    BASELINE_MIN_SAMPLES = 8
    FILTER_ALPHA = 0.25
    NOISE_MULTIPLIER = 5.0
    MIN_EFFORT_DELTA = 0.35
    CONTACT_CONFIRM_S = 0.15

    LEFT_JOINTS = tuple(f"left_arm_joint{i}" for i in range(1, 7))
    RIGHT_JOINTS = tuple(f"right_arm_joint{i}" for i in range(1, 7))
    OBSERVED_JOINTS = ("slide_joint", *LEFT_JOINTS, *RIGHT_JOINTS)

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._baseline_started_s: float | None = None
        self._samples: dict[str, list[float]] = {
            name: [] for name in self.OBSERVED_JOINTS
        }
        self.baseline: dict[str, float] = {}
        self.threshold: dict[str, float] = {}
        self.filtered: dict[str, float] = {}
        self.latest_delta: dict[str, float] = {}
        self.available: bool | None = None
        self.ready = False
        self.contact_candidate_since_s: float | None = None
        self.contact_confirmed = False

    def prepare_baseline(self, now_s: float, joint_states: Any) -> tuple[bool, str]:
        if self.ready:
            return True, self.detail
        efforts = _effort_map(joint_states)
        if efforts is None or any(name not in efforts for name in self.OBSERVED_JOINTS):
            self.available = False
            self.ready = True
            return True, "placement_effort=unavailable; geometry_fallback=true"
        now = float(now_s)
        if not math.isfinite(now):
            raise PregraspInputError("placement effort baseline time is non-finite")
        if self._baseline_started_s is None:
            self._baseline_started_s = now
        for name in self.OBSERVED_JOINTS:
            self._samples[name].append(float(efforts[name]))
        elapsed = max(0.0, now - self._baseline_started_s)
        count = min(len(values) for values in self._samples.values())
        if elapsed < self.BASELINE_TIME_S or count < self.BASELINE_MIN_SAMPLES:
            self.available = True
            return False, (
                f"placement_effort=baseline; elapsed={elapsed:.2f}/"
                f"{self.BASELINE_TIME_S:.2f}s, samples={count}/"
                f"{self.BASELINE_MIN_SAMPLES}"
            )
        for name, values in self._samples.items():
            center = float(median(values))
            mad = float(median(abs(value - center) for value in values))
            self.baseline[name] = center
            self.threshold[name] = max(
                self.MIN_EFFORT_DELTA,
                self.NOISE_MULTIPLIER * 1.4826 * mad,
            )
            self.filtered[name] = center
            self.latest_delta[name] = 0.0
        self.available = True
        self.ready = True
        return True, self.detail

    def observe(
        self,
        now_s: float,
        joint_states: Any,
        *,
        motion_settled: bool,
    ) -> tuple[bool, str]:
        if not self.ready:
            raise PregraspPlanningError("placement effort observed before baseline")
        if not self.available:
            return False, self.detail
        efforts = _effort_map(joint_states)
        if efforts is None or any(name not in efforts for name in self.OBSERVED_JOINTS):
            self.available = False
            self.contact_candidate_since_s = None
            return False, "placement_effort=lost; geometry_fallback=true"
        now = float(now_s)
        for name in self.OBSERVED_JOINTS:
            current = float(efforts[name])
            previous = self.filtered[name]
            filtered = self.FILTER_ALPHA * current + (1.0 - self.FILTER_ALPHA) * previous
            self.filtered[name] = filtered
            self.latest_delta[name] = abs(filtered - self.baseline[name])

        def ratio(name: str) -> float:
            return self.latest_delta[name] / max(self.threshold[name], 1e-9)

        left_wrist = ratio("left_arm_joint6")
        right_wrist = ratio("right_arm_joint6")
        slide = ratio("slide_joint")
        left_arm = max(ratio(name) for name in self.LEFT_JOINTS)
        right_arm = max(ratio(name) for name in self.RIGHT_JOINTS)
        bilateral_wrist = left_wrist >= 1.0 and right_wrist >= 1.0
        supported_arm_load = slide >= 1.0 and left_arm >= 1.0 and right_arm >= 1.0
        evidence = bool(motion_settled and (bilateral_wrist or supported_arm_load))
        if evidence:
            if self.contact_candidate_since_s is None:
                self.contact_candidate_since_s = now
            elif now - self.contact_candidate_since_s >= self.CONTACT_CONFIRM_S:
                self.contact_confirmed = True
        else:
            self.contact_candidate_since_s = None
        return self.contact_confirmed, self.detail

    @property
    def contact_candidate(self) -> bool:
        return self.contact_candidate_since_s is not None and not self.contact_confirmed

    @property
    def detail(self) -> str:
        if self.available is False:
            return "placement_effort=unavailable; geometry_fallback=true"
        if not self.ready:
            return "placement_effort=baseline"
        if not self.latest_delta:
            return "placement_effort=ready"
        ratios = {
            name: self.latest_delta[name] / max(self.threshold[name], 1e-9)
            for name in self.OBSERVED_JOINTS
        }
        return (
            "placement_effort=active, "
            f"slide_ratio={ratios['slide_joint']:.2f}, "
            f"left_wrist_ratio={ratios['left_arm_joint6']:.2f}, "
            f"right_wrist_ratio={ratios['right_arm_joint6']:.2f}, "
            f"left_arm_max={max(ratios[name] for name in self.LEFT_JOINTS):.2f}, "
            f"right_arm_max={max(ratios[name] for name in self.RIGHT_JOINTS):.2f}, "
            f"candidate={self.contact_candidate}, confirmed={self.contact_confirmed}"
        )


class CompliantSlideLoweringController:
    """Approach quickly, then lower in 2 mm increments near the support."""

    DESCENT_STEP_M = 0.002
    FINE_APPROACH_MARGIN_M = 0.012
    STEP_OBSERVE_S = 0.20

    def __init__(self) -> None:
        self._monitor = PlacementContactMonitor()
        self._step = SlideHoldController()
        self.reset()

    def reset(self) -> None:
        self._monitor.reset()
        self._step.reset()
        self._hold_command: ArmCommand | None = None
        self._final_target: float | None = None
        self._step_observe_started_s: float | None = None
        self._fallback = False
        self._completed = False
        self.contact_detected = False
        self.completion_reason: str | None = None
        self.phase = "idle"

    @property
    def planned(self) -> bool:
        return self._hold_command is not None and self._final_target is not None

    @property
    def target_slide(self) -> float | None:
        return self._final_target

    def plan(
        self,
        hold_command: ArmCommand,
        target_slide: float,
        joint_states: Any,
    ) -> ArmCommand:
        # Validate through the existing official JointState-aware controller.
        validator = SlideHoldController()
        validator.plan(hold_command, hold_command.spine_position, joint_states)
        self.reset()
        self._hold_command = hold_command
        self._final_target = float(target_slide)
        self.phase = "baseline"
        return hold_command

    def _plan_next_step(self, joint_states: Any) -> None:
        assert self._hold_command is not None
        assert self._final_target is not None
        current = float(self._hold_command.spine_position)
        difference = self._final_target - current
        if abs(difference) <= 1e-9:
            self._completed = True
            self.completion_reason = "geometry_target"
            self.phase = "geometry_complete"
            return
        next_target = current + math.copysign(
            min(abs(difference), self.DESCENT_STEP_M), difference
        )
        self._step.reset()
        self._hold_command = self._step.plan(
            self._hold_command,
            next_target,
            joint_states,
        )
        self._step_observe_started_s = None
        self.phase = "descend"

    def _plan_approach(self, joint_states: Any) -> None:
        """Move directly to the start of the final force-controlled window."""

        assert self._hold_command is not None
        assert self._final_target is not None
        current = float(self._hold_command.spine_position)
        difference = self._final_target - current
        if abs(difference) <= self.FINE_APPROACH_MARGIN_M:
            self._plan_next_step(joint_states)
            return
        approach_target = self._final_target - math.copysign(
            self.FINE_APPROACH_MARGIN_M, difference
        )
        self._step.reset()
        self._hold_command = self._step.plan(
            self._hold_command,
            approach_target,
            joint_states,
        )
        self._step_observe_started_s = None
        self.phase = "fast_approach"

    def update(
        self,
        now_s: float,
        joint_states: Any,
    ) -> tuple[ArmCommand, bool, str]:
        if not self.planned or self._hold_command is None or self._final_target is None:
            raise PregraspPlanningError("compliant lowering updated before plan")
        if self._completed:
            return self._hold_command, True, self.detail

        if self.phase == "baseline":
            ready, baseline_detail = self._monitor.prepare_baseline(now_s, joint_states)
            if not ready:
                return self._hold_command, False, baseline_detail
            if not self._monitor.available:
                self._fallback = True
                self._step.reset()
                self._hold_command = self._step.plan(
                    self._hold_command,
                    self._final_target,
                    joint_states,
                )
                self.phase = "geometry_fallback"
            else:
                self._plan_approach(joint_states)
                if self._completed:
                    return self._hold_command, True, self.detail

        if self._fallback:
            command, reached, detail = self._step.update(now_s, joint_states)
            self._hold_command = command
            if reached:
                self._completed = True
                self.completion_reason = "geometry_fallback"
                self.phase = "geometry_complete"
            return command, self._completed, f"{detail}; {self.detail}"

        command, step_reached, step_detail = self._step.update(now_s, joint_states)
        self._hold_command = command

        if self.phase == "fast_approach":
            if not step_reached:
                return command, False, f"{step_detail}; {self.detail}"
            self._plan_next_step(joint_states)
            if self._completed:
                return self._hold_command, True, self.detail
            return self._hold_command, False, self.detail

        if not step_reached:
            self.phase = "descend"
            return command, False, f"{step_detail}; {self.detail}"

        now = float(now_s)
        if self._step_observe_started_s is None:
            self._step_observe_started_s = now
        contact, effort_detail = self._monitor.observe(
            now,
            joint_states,
            motion_settled=True,
        )
        self.phase = "contact_confirm" if self._monitor.contact_candidate else "descend"
        if contact:
            self._completed = True
            self.contact_detected = True
            self.completion_reason = "effort_contact"
            self.phase = "contact_complete"
            return command, True, f"{step_detail}; {effort_detail}"

        if now - self._step_observe_started_s >= self.STEP_OBSERVE_S:
            self._plan_next_step(joint_states)
            if self._completed:
                return self._hold_command, True, self.detail
        return self._hold_command, False, f"{step_detail}; {effort_detail}"

    @property
    def detail(self) -> str:
        return (
            f"compliant_place_phase={self.phase}, "
            f"step_mm={self.DESCENT_STEP_M * 1000.0:.1f}, "
            f"completion={self.completion_reason}; {self._monitor.detail}"
        )


__all__ = ["CompliantSlideLoweringController", "PlacementContactMonitor"]
