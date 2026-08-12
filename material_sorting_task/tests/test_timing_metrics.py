from __future__ import annotations

import contextlib
import io
import unittest

from timing_metrics import TimingRecorder, percentile95


class TimingMetricsTests(unittest.TestCase):
    def test_nearest_rank_p95(self) -> None:
        self.assertEqual(percentile95(list(range(1, 21))), 19.0)
        self.assertIsNone(percentile95([]))

    def test_records_phase_mean_and_p95(self) -> None:
        recorder = TimingRecorder("test")
        with contextlib.redirect_stdout(io.StringIO()):
            recorder.begin("place", 1.0)
            recorder.finish(2.0)
            recorder.begin("place", 5.0)
            recorder.finish(8.0)

        summary = recorder.summary()["place"]
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["mean_s"], 2.0)
        self.assertEqual(summary["p95_s"], 3.0)


if __name__ == "__main__":
    unittest.main()
