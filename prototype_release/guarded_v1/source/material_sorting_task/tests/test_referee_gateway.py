from __future__ import annotations

import unittest

from scheduler.referee import RefereeGateway


class RefereeGatewayTests(unittest.TestCase):
    def test_parses_existing_server_topics_and_chinese_finish(self) -> None:
        gateway = RefereeGateway()

        update = gateway.observe(
            "t=12.0s score=40 task=3/3 best=[20, 20, 0] attempt=1 step=place",
            "全部任务结束",
        )

        self.assertEqual(update.task_ordinal, 3)
        self.assertEqual(update.task_total, 3)
        self.assertEqual(update.attempts_completed, 1)
        self.assertEqual(update.score, 40)
        self.assertEqual(update.step, "place")
        self.assertTrue(update.all_tasks_done)
        self.assertTrue(update.changed)

    def test_repeated_semantics_are_idempotent_even_when_time_changes(self) -> None:
        gateway = RefereeGateway()
        first = gateway.observe(
            "t=1.0s score=0 task=1/3 attempt=0 step=nav",
            "任务1: 搬运粉色物体",
        )
        repeated = gateway.observe(
            "t=1.1s score=0 task=1/3 attempt=0 step=nav",
            "任务1: 搬运粉色物体",
        )

        self.assertTrue(first.changed)
        self.assertFalse(repeated.changed)
        self.assertEqual(repeated.revision, first.revision)
        self.assertEqual(repeated.observation_serial, first.observation_serial + 1)

    def test_structured_mapping_is_supported(self) -> None:
        update = RefereeGateway().observe(
            {
                "task_ordinal": "2",
                "task_total": 3,
                "attempts_completed": "1",
                "step": "lift",
                "score": 20,
            },
            "任务2: 从货架取回物体",
        )

        self.assertEqual(update.task_ordinal, 2)
        self.assertEqual(update.task_id, 2)
        self.assertFalse(update.desynchronised)

    def test_disagreement_and_regression_are_reported(self) -> None:
        gateway = RefereeGateway()
        gateway.observe("score=0 task=2/3 attempt=2 step=nav", "任务2: x")

        mismatch = gateway.observe("score=0 task=1/3 attempt=1 step=nav", "任务2: x")
        repeated = gateway.observe("score=0 task=1/3 attempt=1 step=nav", "任务2: x")

        self.assertTrue(mismatch.desynchronised)
        self.assertIn("task ordinal regressed", mismatch.desync_reasons)
        self.assertIn(
            "gameinfo and taskinfo disagree on current task",
            mismatch.desync_reasons,
        )
        self.assertTrue(repeated.desynchronised)
        self.assertFalse(repeated.changed)

    def test_attempt_may_reset_when_referee_advances_task(self) -> None:
        gateway = RefereeGateway()
        gateway.observe("task=1/3 attempt=2 step=place", "任务1: x")
        update = gateway.observe("task=2/3 attempt=0 step=nav", "任务2: y")

        self.assertFalse(update.desynchronised)
        self.assertTrue(update.changed)

    def test_bad_skipped_frame_does_not_poison_later_correction(self) -> None:
        gateway = RefereeGateway()
        gateway.observe("task=1/3 attempt=0 step=nav", "任务1: x")

        skipped = gateway.observe("task=3/3 attempt=0 step=nav", "任务3: x")
        corrected = gateway.observe("task=2/3 attempt=0 step=nav", "任务2: x")

        self.assertTrue(skipped.desynchronised)
        self.assertIn("task ordinal skipped a task", skipped.desync_reasons)
        self.assertFalse(corrected.desynchronised)


if __name__ == "__main__":
    unittest.main()
