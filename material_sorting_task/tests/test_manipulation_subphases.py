from __future__ import annotations

import unittest
from types import SimpleNamespace

from competition_controller import CompetitionController, ControllerState
from executors.base import (
    ExecutionContext,
    StageResult,
    StageStatus,
    TaskStage,
    annotate_placement_result,
)
from executors.task1 import Task1ContactExecutor
from scheduler.events import EventLog, MemoryEventSink


TASKS = [
    {
        "task": task_id,
        "instruction": f"task {task_id}",
        "target_color": color,
        "place_world": [float(task_id), 0.0, 0.5],
    }
    for task_id, color in ((1, "pink"), (2, "yellow"), (3, "brown"))
]


class _ScriptedExecutor:
    name = "manipulation_subphase_script"

    def __init__(self, task_id: int, results=()) -> None:
        self.task_id = task_id
        self.results = list(results)

    def reset(self) -> None:
        pass

    def enter_stage(self, stage, context) -> None:
        pass

    def tick(self, stage, context) -> StageResult:
        if self.results:
            return self.results.pop(0)
        return StageResult.succeeded("scripted success")

    def cancel(self, reason: str) -> None:
        pass


def _context(controller: CompetitionController) -> ExecutionContext:
    index = min(controller.task_index, len(TASKS) - 1)
    return ExecutionContext(
        now_s=float(controller.snapshot().transition_serial),
        instruction=TASKS[index],
        task_index=index,
        attempt=controller.attempt,
    )


class ManipulationSubphaseTests(unittest.TestCase):
    def test_stage_result_normalises_and_freezes_manipulation_evidence(self) -> None:
        result = StageResult.running("contact").with_manipulation_subphase(
            " Grasp ",
            " Bilateral_Lock ",
            left_contact=True,
        )

        self.assertEqual(result.metadata["manipulation_kind"], "grasp")
        self.assertEqual(result.metadata["manipulation_subphase"], "bilateral_lock")
        self.assertTrue(result.metadata["manipulation_evidence"]["left_contact"])
        with self.assertRaises(TypeError):
            result.metadata["manipulation_evidence"]["left_contact"] = False
        with self.assertRaises(ValueError):
            result.with_manipulation_subphase("grasp", "unbounded_squeeze")

    def test_grasp_annotation_reports_lock_once_then_preload(self) -> None:
        executor = Task1ContactExecutor.__new__(Task1ContactExecutor)
        executor._contact = SimpleNamespace(
            planned=True,
            bilateral_aligned=True,
            any_contact=True,
            half_width=0.22,
        )
        executor._contact_search_used_m = 0.012
        executor._compliance_retry_count = 1
        executor._grasp_bilateral_lock_reported = False

        locked = executor._annotate_grasp_result(StageResult.running("locked"))
        preload = executor._annotate_grasp_result(StageResult.running("preload"))
        settled = executor._annotate_grasp_result(StageResult.succeeded("settled"))

        self.assertEqual(locked.metadata["manipulation_subphase"], "bilateral_lock")
        self.assertEqual(preload.metadata["manipulation_subphase"], "preload")
        self.assertEqual(settled.metadata["manipulation_subphase"], "settled")

    def test_placement_controller_phases_map_to_stable_schema(self) -> None:
        baseline = annotate_placement_result(
            StageResult.running(),
            "lower",
            SimpleNamespace(phase="baseline", contact_candidate=False),
        )
        candidate = annotate_placement_result(
            StageResult.running(),
            "lower",
            SimpleNamespace(phase="contact_confirm", contact_candidate=True),
        )
        confirmed = annotate_placement_result(
            StageResult.running(),
            "lower",
            SimpleNamespace(phase="contact_complete", contact_candidate=True),
        )
        released = annotate_placement_result(StageResult.succeeded(), "release")
        cleanup = annotate_placement_result(
            StageResult.running(), "cleanup", cleanup=True
        )

        self.assertEqual(baseline.metadata["manipulation_subphase"], "baseline")
        self.assertEqual(candidate.metadata["manipulation_subphase"], "contact_candidate")
        self.assertEqual(confirmed.metadata["manipulation_subphase"], "contact_confirm")
        self.assertEqual(released.metadata["manipulation_subphase"], "release")
        self.assertEqual(cleanup.metadata["manipulation_subphase"], "post_release_cleanup")

    def test_engine_emits_only_manipulation_subphase_transitions(self) -> None:
        results = [
            StageResult.succeeded("navigation"),
            StageResult.succeeded("target"),
            StageResult.succeeded("alignment"),
            StageResult.running("approach").with_manipulation_subphase(
                "grasp", "approach"
            ),
            StageResult.running("same approach").with_manipulation_subphase(
                "grasp", "approach"
            ),
            StageResult.running("locked").with_manipulation_subphase(
                "grasp", "bilateral_lock"
            ),
            StageResult.succeeded("settled").with_manipulation_subphase(
                "grasp", "settled"
            ),
        ]
        sink = MemoryEventSink()
        controller = CompetitionController(
            {
                1: _ScriptedExecutor(1, results),
                2: _ScriptedExecutor(2),
                3: _ScriptedExecutor(3),
            },
            referee_driven=False,
            scheduler_mode="v2",
            event_sink=EventLog([sink]),
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        for _ in range(300):
            controller.tick(_context(controller))
            if controller.state is ControllerState.FINISHED:
                break
        else:
            self.fail("controller did not finish")

        events = [
            event for event in sink.events if event.type == "manipulation_subphase"
        ]
        self.assertEqual(
            [event.details["subphase"] for event in events],
            ["approach", "bilateral_lock", "settled"],
        )
        self.assertTrue(all(event.step_id == TaskStage.GRASP.value for event in events))

    def test_engine_rejects_subphase_from_wrong_stage(self) -> None:
        malformed = StageResult.succeeded("bad").with_manipulation_subphase(
            "grasp", "approach"
        )
        controller = CompetitionController(
            {
                1: _ScriptedExecutor(1, [malformed]),
                2: _ScriptedExecutor(2),
                3: _ScriptedExecutor(3),
            },
            referee_driven=False,
            scheduler_mode="v2",
        )
        controller.configure(TASKS)
        controller.set_inputs_ready(True)

        for _ in range(20):
            controller.tick(_context(controller))
            if controller.state is ControllerState.SAFE_HOLD:
                break

        self.assertIs(controller.state, ControllerState.SAFE_HOLD)
        self.assertIn("outside the grasp stage", controller.snapshot().message)


if __name__ == "__main__":
    unittest.main()
