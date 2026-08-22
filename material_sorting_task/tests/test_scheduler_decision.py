from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile

from competition_controller import CompetitionController
from executors import build_task_executors
from executors.base import ExecutionContext, StageResult, TargetObservation
from executors.scheduler_candidate import CandidateApplicationStatus
from scheduler.candidate_generator import CandidateAction
from scheduler.decision import DecisionConfig, SchedulerDecisionService
from scheduler.events import EventLog, MemoryEventSink
from scheduler.policies import RLPolicy
from scheduler.plans import build_executor_task_plans
from scheduler.project_candidates import ProjectCandidateProvider


def candidate(action_id: str, score: float, **constraints) -> CandidateAction:
    return CandidateAction(
        action_id=action_id,
        action_type="rescan",
        expected_score=score,
        success_probability=1.0,
        hard_constraints=constraints,
    )


class _FixedModel:
    def __init__(self, action_index: int) -> None:
        self.action_index = action_index

    def predict(self, observation, *, action_masks, deterministic=True):
        return self.action_index, None


class SchedulerDecisionTests(unittest.TestCase):
    def test_heuristic_returns_highest_safe_utility(self) -> None:
        service = SchedulerDecisionService()

        outcome = service.decide(
            (
                candidate("blocked", 100.0, collision_free=False),
                candidate("safe-low", 2.0, collision_free=True),
                candidate("safe-high", 5.0, collision_free=True),
            ),
            now_s=1.0,
        )

        self.assertEqual(outcome.action_id, "safe-high")
        self.assertEqual(outcome.source, "heuristic")
        invalid = next(item for item in outcome.evaluations if item.action_id == "blocked")
        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.utility, float("-inf"))
        service.close()

    def test_rl_shadow_records_suggestion_without_changing_selection(self) -> None:
        service = SchedulerDecisionService(
            config=DecisionConfig(
                policy_mode="rl_shadow",
                candidate_stability_frames=1,
            ),
            rl_policy=RLPolicy(model=_FixedModel(1)),
        )

        outcome = service.decide(
            (candidate("high", 10.0), candidate("low", 1.0)),
            now_s=1.0,
        )

        self.assertEqual(outcome.action_id, "high")
        self.assertEqual(outcome.policy_suggestion.action_id, "low")
        self.assertEqual(outcome.source, "heuristic")
        service.close()

    def test_rl_shadow_fallback_is_not_mislabeled_as_policy_suggestion(self) -> None:
        service = SchedulerDecisionService(
            config=DecisionConfig(
                policy_mode="rl_shadow",
                candidate_stability_frames=1,
            ),
            rl_policy=RLPolicy(),
        )
        outcome = service.decide(
            (candidate("high", 10.0), candidate("low", 1.0)),
            now_s=1.0,
        )

        self.assertEqual(outcome.action_id, "high")
        self.assertIsNone(outcome.policy_suggestion)
        self.assertEqual(outcome.policy_decision_reason, "model_missing")
        service.close()

    def test_rl_guarded_can_choose_only_an_unmasked_macro_action(self) -> None:
        service = SchedulerDecisionService(
            config=DecisionConfig(
                policy_mode="rl_guarded",
                candidate_stability_frames=1,
                rl_max_utility_regret=100.0,
            ),
            rl_policy=RLPolicy(model=_FixedModel(1)),
        )

        outcome = service.decide(
            (candidate("high", 10.0), candidate("low", 1.0)),
            now_s=1.0,
        )

        self.assertEqual(outcome.action_id, "low")
        self.assertEqual(outcome.source, "rl")
        service.close()

    def test_missing_model_falls_back_to_heuristic(self) -> None:
        service = SchedulerDecisionService(
            config=DecisionConfig(
                policy_mode="rl_guarded",
                candidate_stability_frames=1,
            ),
            rl_policy=RLPolicy(),
        )

        outcome = service.decide(
            (candidate("best", 3.0), candidate("other", 1.0)),
            now_s=1.0,
        )

        self.assertEqual(outcome.action_id, "best")
        self.assertEqual(outcome.source, "heuristic")
        self.assertEqual(outcome.reason, "model_missing")
        service.close()

    def test_selection_hysteresis_requires_stable_superior_candidate(self) -> None:
        service = SchedulerDecisionService(
            config=DecisionConfig(
                minimum_action_hold_s=0.0,
                switch_utility_margin=0.5,
                candidate_stability_frames=2,
            )
        )
        first = service.decide(
            (candidate("a", 10.0), candidate("b", 9.0)),
            now_s=0.0,
        )
        pending = service.decide(
            (candidate("a", 10.0), candidate("b", 12.0)),
            now_s=1.0,
        )
        switched = service.decide(
            (candidate("a", 10.0), candidate("b", 12.0)),
            now_s=2.0,
        )

        self.assertEqual(first.action_id, "a")
        self.assertEqual(pending.action_id, "a")
        self.assertEqual(pending.reason, "candidate_not_stable")
        self.assertEqual(switched.action_id, "b")
        self.assertTrue(switched.switched)
        service.close()

    def test_decisions_are_auditable(self) -> None:
        sink = MemoryEventSink()
        event_log = EventLog([sink], clock=lambda: 0.0)
        service = SchedulerDecisionService(event_log=event_log)

        service.decide((candidate("safe", 1.0),), now_s=5.0)

        self.assertEqual(
            [event.type for event in sink.events],
            ["candidates_evaluated", "action_selected"],
        )
        candidate_details = sink.events[0].details
        self.assertEqual(candidate_details["observation_schema_version"], "scheduler-observation-v1")
        self.assertEqual(len(candidate_details["action_mask"]), 8)
        self.assertEqual(candidate_details["action_mask"][:2], [True, False])
        self.assertEqual(len(candidate_details["observation"]), 138)
        self.assertEqual(sink.events[-1].action_id, "safe")
        service.close()

    def test_invalid_candidate_is_jsonl_auditable_without_infinity(self) -> None:
        from scheduler.events import JsonlEventSink

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scheduler.jsonl"
            event_log = EventLog([JsonlEventSink(path)], clock=lambda: 0.0)
            service = SchedulerDecisionService(event_log=event_log)

            service.decide(
                (
                    candidate("blocked", 10.0, collision_free=False),
                    candidate("safe", 1.0, collision_free=True),
                ),
                now_s=5.0,
            )

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn('"utility":null', lines[0])
            service.close()

    def test_project_provider_builds_live_pick_stands(self) -> None:
        provider = ProjectCandidateProvider()
        odometry = SimpleNamespace(
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=-0.5, y=0.5),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            )
        )
        observation = TargetObservation(
            color="pink",
            position_world=(-1.0, 2.2, 0.834),
            received_at_s=1.0,
            score=0.9,
        )
        context = ExecutionContext(
            now_s=1.0,
            instruction={"task": 1, "target_color": "pink"},
            task_index=0,
            attempt=1,
            odometry=odometry,
            target_observations={"pink": observation},
        )

        batch = provider.build(context, build_executor_task_plans()[1].stages[0])

        self.assertIsNotNone(batch)
        self.assertEqual(len(batch.candidates), 3)
        center = batch.candidates[0]
        self.assertAlmostEqual(center.x, -1.0)
        self.assertAlmostEqual(center.y, 1.55)
        self.assertTrue(all(item.metadata["costmap_sidecar"] for item in batch.candidates))

    def test_project_provider_rejects_stale_target_observation(self) -> None:
        provider = ProjectCandidateProvider()
        odometry = SimpleNamespace(
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=-0.5, y=0.5),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            )
        )
        stale = TargetObservation(
            color="pink",
            position_world=(-1.0, 2.2, 0.834),
            received_at_s=1.0,
            score=0.9,
        )
        context = ExecutionContext(
            now_s=100.0,
            instruction={"task": 1, "target_color": "pink"},
            task_index=0,
            attempt=1,
            odometry=odometry,
            target_observations={"pink": stale},
        )

        batch = provider.build(context, build_executor_task_plans()[1].stages[0])

        self.assertIsNone(batch)

    def test_v2_polls_decisions_nonblocking_and_uses_only_opt_in_hook(self) -> None:
        class HookExecutor:
            task_id = 1
            name = "hook"

            def __init__(self):
                self.selected = []
                self.call_order = []

            def reset(self):
                pass

            def enter_stage(self, stage, execution_context):
                pass

            def tick(self, stage, execution_context):
                self.call_order.append("tick")
                return StageResult.running("active")

            def cancel(self, reason):
                pass

            def apply_scheduler_candidate(self, selected, outcome, context):
                self.selected.append(selected.action_id)
                self.call_order.append("apply")
                return CandidateApplicationStatus.APPLIED

        class Provider:
            def build(self, context, spec):
                return SimpleNamespace(
                    candidates=(candidate("chosen", 1.0),),
                    start_pose=None,
                    costmap=None,
                    constraints=None,
                    footprint_mode="transit_stowed",
                    world_state={},
                )

        tasks = [
            {"task": task_id, "target_color": color, "place_world": [0, 0, 0]}
            for task_id, color in ((1, "pink"), (2, "yellow"), (3, "brown"))
        ]
        hook = HookExecutor()
        sink = MemoryEventSink()
        executors = build_task_executors("stub")
        executors[1] = hook
        service = SchedulerDecisionService()
        controller = CompetitionController(
            executors,
            referee_driven=False,
            scheduler_mode="v2",
            decision_service=service,
            candidate_provider=Provider(),
            event_sink=EventLog([sink]),
        )
        controller.configure(tasks)
        controller.set_inputs_ready(True)
        context = ExecutionContext(
            now_s=1.0,
            instruction=tasks[0],
            task_index=0,
            attempt=1,
        )
        controller.tick(context)
        controller.tick(context)
        controller.tick(context)
        controller._backend._decision_future.result(timeout=1.0)
        controller.tick(context)

        self.assertEqual(hook.selected, ["chosen"])
        self.assertEqual(hook.call_order[:2], ["apply", "tick"])
        self.assertEqual(controller._backend.last_candidate_application, "applied")
        applications = [
            event for event in sink.events if event.type == "candidate_application"
        ]
        self.assertEqual(len(applications), 1)
        self.assertEqual(applications[0].details["application_status"], "applied")

        # Periodic sidecar outcomes must not reapply an unchanged semantic
        # action in the same step; doing so can reset a live navigation goal.
        outcome = controller._backend.last_decision
        controller._backend._offer_candidate_to_executor(outcome, context)
        self.assertEqual(hook.selected, ["chosen"])
        self.assertEqual(
            len([event for event in sink.events if event.type == "candidate_application"]),
            1,
        )

        # Recovery/stage re-entry starts a fresh step run and intentionally
        # permits the current candidate to be installed again.
        controller._backend._start_step_run()
        controller._backend._offer_candidate_to_executor(outcome, context)
        self.assertEqual(hook.selected, ["chosen", "chosen"])
        controller.close()

    def test_candidate_offer_reapplies_when_goal_pose_changes(self) -> None:
        class HookExecutor:
            task_id = 1
            name = "goal-change"

            def __init__(self):
                self.goals = []

            def reset(self):
                pass

            def enter_stage(self, stage, execution_context):
                pass

            def tick(self, stage, execution_context):
                return StageResult.running("active")

            def cancel(self, reason):
                pass

            def apply_scheduler_candidate(self, selected, outcome, context):
                self.goals.append(selected.goal_pose)
                return CandidateApplicationStatus.APPLIED

        tasks = [
            {"task": task_id, "target_color": color, "place_world": [0, 0, 0]}
            for task_id, color in ((1, "pink"), (2, "yellow"), (3, "brown"))
        ]
        hook = HookExecutor()
        executors = build_task_executors("stub")
        executors[1] = hook
        controller = CompetitionController(
            executors,
            referee_driven=False,
            scheduler_mode="v2",
        )
        controller.configure(tasks)
        backend = controller._backend
        backend._start_task_run()
        backend._start_step_run()
        context = ExecutionContext(
            now_s=1.0,
            instruction=tasks[0],
            task_index=0,
            attempt=1,
        )

        def outcome(x):
            selected = SimpleNamespace(
                action_id="same-id",
                candidate=CandidateAction(
                    action_id="same-id",
                    action_type="navigate",
                    x=x,
                    y=1.0,
                    yaw=0.0,
                    metadata={"lateral_offset_m": 0.0},
                ),
            )
            return SimpleNamespace(selected=selected)

        backend._offer_candidate_to_executor(outcome(0.0), context)
        backend._offer_candidate_to_executor(outcome(0.0), context)
        backend._offer_candidate_to_executor(outcome(0.1), context)

        self.assertEqual(hook.goals, [(0.0, 1.0, 0.0), (0.1, 1.0, 0.0)])
        controller.close()

    def test_initial_candidate_wait_is_nonblocking_and_strictly_bounded(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class HookExecutor:
            task_id = 1
            name = "bounded-wait"

            def __init__(self):
                self.tick_count = 0

            def reset(self):
                pass

            def enter_stage(self, stage, execution_context):
                pass

            def tick(self, stage, execution_context):
                self.tick_count += 1
                return StageResult.running("active")

            def cancel(self, reason):
                pass

            def apply_scheduler_candidate(self, selected, outcome, context):
                return CandidateApplicationStatus.APPLIED

        class DelayedProvider:
            def build(self, context, spec, **kwargs):
                started.set()
                release.wait(timeout=1.0)
                return SimpleNamespace(
                    candidates=(candidate("chosen", 1.0),),
                    start_pose=None,
                    costmap=None,
                    constraints=None,
                    footprint_mode="transit_stowed",
                    world_state={},
                )

        tasks = [
            {"task": task_id, "target_color": color, "place_world": [0, 0, 0]}
            for task_id, color in ((1, "pink"), (2, "yellow"), (3, "brown"))
        ]
        hook = HookExecutor()
        executors = build_task_executors("stub")
        executors[1] = hook
        service = SchedulerDecisionService()
        controller = CompetitionController(
            executors,
            referee_driven=False,
            scheduler_mode="v2",
            decision_service=service,
            candidate_provider=DelayedProvider(),
            candidate_initial_wait_s=0.10,
        )
        controller.configure(tasks)
        controller.set_inputs_ready(True)

        def context(now_s):
            return ExecutionContext(
                now_s=now_s,
                instruction=tasks[0],
                task_index=0,
                attempt=1,
            )

        try:
            controller.tick(context(0.0))
            controller.tick(context(0.0))
            controller.tick(context(0.0))
            self.assertTrue(started.wait(timeout=1.0))

            waiting = controller.tick(context(0.05))
            self.assertEqual(hook.tick_count, 0)
            self.assertIn("waiting up to", waiting.message)

            controller.tick(context(0.11))
            self.assertEqual(hook.tick_count, 1)
        finally:
            release.set()
            controller.close()


if __name__ == "__main__":
    unittest.main()
