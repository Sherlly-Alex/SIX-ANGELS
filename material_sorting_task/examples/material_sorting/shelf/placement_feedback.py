"""Effort-aware vertical placement using the official JointState interface.

The Server exposes actuator generalized effort through ``/joint_states`` but
does not expose fingertip force or a six-axis end-effector wrench.  This module
therefore treats effort changes only as evidence that the held object has
become supported.  It keeps the existing geometric target as a hard fallback:
the controller approaches that target, checks for support in a bounded final
window, and never descends past it.
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
    """Detect settled bilateral support relative to a static effort baseline."""

    BASELINE_TIME_S = 0.40
    BASELINE_MIN_SAMPLES = 8
    FILTER_ALPHA = 0.25
    NOISE_MULTIPLIER = 5.0
    MIN_EFFORT_DELTA = 0.35
    CONTACT_CONFIRM_S = 0.15

    LEFT_JOINTS = tuple(f"left_arm_joint{index}" for index in range(1, 7))
    RIGHT_JOINTS = tuple(f"right_arm_joint{index}" for index in range(1, 7))
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

    def clear_candidate(self) -> None:
        """Do not carry stationary contact evidence through another motion."""

        if not self.contact_confirmed:
            self.contact_candidate_since_s = None

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
        contact_enabled: bool,
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

        slide_loaded = ratio("slide_joint") >= 1.0
        bilateral_wrist = (
            ratio("left_arm_joint6") >= 1.0
            and ratio("right_arm_joint6") >= 1.0
        )
        bilateral_arm = (
            max(ratio(name) for name in self.LEFT_JOINTS) >= 1.0
            and max(ratio(name) for name in self.RIGHT_JOINTS) >= 1.0
        )
        # Requiring the slide plus bilateral arm evidence rejects a single arm
        # touching a shelf wall and rejects the slide's own stopping transient.
        evidence = bool(
            contact_enabled
            and motion_settled
            and slide_loaded
            and (bilateral_wrist or bilateral_arm)
        )
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
    """Approach geometrically, then descend incrementally while sensing support."""

    DESCENT_STEP_M = 0.002
    FINE_APPROACH_MARGIN_M = 0.012
    CONTACT_ENABLE_MARGIN_M = 0.008
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
        # Validate the command and JointState through the existing controller.
        validator = SlideHoldController()
        validator.plan(hold_command, target_slide, joint_states)
        self.reset()
        self._hold_command = hold_command
        assert validator.target_slide is not None
        self._final_target = float(validator.target_slide)
        if not math.isfinite(self._final_target):
            raise PregraspInputError("placement slide target is non-finite")
        self.phase = "baseline"
        return hold_command

    def _remaining_m(self) -> float:
        assert self._hold_command is not None
        assert self._final_target is not None
        return abs(self._final_target - float(self._hold_command.spine_position))

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
        self._monitor.clear_candidate()
        self._step.reset()
        self._hold_command = self._step.plan(
            self._hold_command, next_target, joint_states
        )
        self._step_observe_started_s = None
        self.phase = "fine_descent"

    def _plan_approach(self, joint_states: Any) -> None:
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
            self._hold_command, approach_target, joint_states
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
                    self._hold_command, self._final_target, joint_states
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
            self.phase = "fine_descent"
            return command, False, f"{step_detail}; {self.detail}"

        now = float(now_s)
        if self._step_observe_started_s is None:
            self._step_observe_started_s = now
        contact_enabled = self._remaining_m() <= self.CONTACT_ENABLE_MARGIN_M + 1e-9
        contact, effort_detail = self._monitor.observe(
            now,
            joint_states,
            motion_settled=True,
            contact_enabled=contact_enabled,
        )
        self.phase = "contact_confirm" if self._monitor.contact_candidate else "observe"
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
            f"remaining_mm={self._remaining_m() * 1000.0:.1f}, "
            f"step_mm={self.DESCENT_STEP_M * 1000.0:.1f}, "
            f"completion={self.completion_reason}; {self._monitor.detail}"
        )


__all__ = ["CompliantSlideLoweringController", "PlacementContactMonitor"]
