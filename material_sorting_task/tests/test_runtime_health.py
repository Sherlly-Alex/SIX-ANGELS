from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runtime_health import (
    ControlLoopTelemetry,
    FreshnessState,
    InputDropFaultInjector,
    InputFreshnessWatchdog,
)


class InputFreshnessWatchdogTests(unittest.TestCase):
    def build_watchdog(self) -> InputFreshnessWatchdog:
        return InputFreshnessWatchdog(
            {"odometry": 0.20, "joint_states": 0.30},
            stale_grace_s=0.50,
        )

    def test_startup_reports_missing_without_terminal_stop(self) -> None:
        watchdog = self.build_watchdog()
        watchdog.observe("odometry", 1.0)

        report = watchdog.evaluate(1.1)

        self.assertIs(report.state, FreshnessState.STARTUP)
        self.assertEqual(report.missing_inputs, ("joint_states",))
        self.assertFalse(report.motion_allowed)
        self.assertFalse(report.terminal)

    def test_fresh_inputs_allow_motion_and_report_ages(self) -> None:
        watchdog = self.build_watchdog()
        watchdog.observe("odometry", 1.0)
        watchdog.observe("joint_states", 1.05)

        report = watchdog.evaluate(1.10)

        self.assertIs(report.state, FreshnessState.FRESH)
        self.assertTrue(report.motion_allowed)
        self.assertAlmostEqual(report.ages_s["odometry"], 0.10)
        self.assertAlmostEqual(report.ages_s["joint_states"], 0.05)

    def test_stale_input_uses_bounded_grace_then_exhausts(self) -> None:
        watchdog = self.build_watchdog()
        watchdog.observe("odometry", 1.0)
        watchdog.observe("joint_states", 1.0)
        self.assertTrue(watchdog.evaluate(1.1).motion_allowed)
        watchdog.arm()

        stale = watchdog.evaluate(1.31)
        boundary = watchdog.evaluate(1.81)
        exhausted = watchdog.evaluate(1.811)

        self.assertIs(stale.state, FreshnessState.STALE_GRACE)
        self.assertEqual(stale.stale_inputs, ("odometry", "joint_states"))
        self.assertFalse(stale.terminal)
        self.assertIs(boundary.state, FreshnessState.STALE_GRACE)
        self.assertIs(exhausted.state, FreshnessState.EXHAUSTED)
        self.assertTrue(exhausted.terminal)

    def test_recovery_inside_grace_rearms_cleanly(self) -> None:
        watchdog = self.build_watchdog()
        watchdog.observe("odometry", 1.0)
        watchdog.observe("joint_states", 1.0)
        watchdog.arm()
        self.assertIs(watchdog.evaluate(1.31).state, FreshnessState.STALE_GRACE)

        watchdog.observe("odometry", 1.4)
        watchdog.observe("joint_states", 1.4)
        recovered = watchdog.evaluate(1.41)

        self.assertIs(recovered.state, FreshnessState.FRESH)
        self.assertTrue(recovered.transitioned)
        self.assertEqual(recovered.stale_for_s, 0.0)

    def test_unarmed_watchdog_never_turns_startup_delay_into_terminal_stop(self) -> None:
        watchdog = self.build_watchdog()
        watchdog.observe("odometry", 1.0)
        watchdog.observe("joint_states", 1.0)

        first = watchdog.evaluate(2.0)
        much_later = watchdog.evaluate(20.0)

        self.assertIs(first.state, FreshnessState.STALE_GRACE)
        self.assertIs(much_later.state, FreshnessState.STALE_GRACE)
        self.assertFalse(much_later.terminal)

    def test_invalid_configuration_and_unknown_observation_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            InputFreshnessWatchdog({}, stale_grace_s=1.0)
        with self.assertRaises(ValueError):
            InputFreshnessWatchdog({"odom": 0.0}, stale_grace_s=1.0)
        watchdog = self.build_watchdog()
        with self.assertRaises(KeyError):
            watchdog.observe("camera", 1.0)


class ControlLoopTelemetryTests(unittest.TestCase):
    def test_statistics_and_deadline_misses_are_deterministic(self) -> None:
        telemetry = ControlLoopTelemetry(
            0.05,
            report_period_s=1.0,
            interval_miss_ratio=1.5,
            window_size=10,
        )
        telemetry.begin(0.0)
        self.assertIsNone(telemetry.finish(0.0, 0.010))
        telemetry.begin(0.05)
        self.assertIsNone(telemetry.finish(0.05, 0.070))
        telemetry.begin(0.14)
        self.assertIsNone(telemetry.finish(0.14, 0.201))

        health = telemetry.snapshot()

        self.assertEqual(health.sample_count, 3)
        self.assertAlmostEqual(health.interval_p50_ms, 50.0)
        self.assertAlmostEqual(health.interval_p95_ms, 90.0)
        self.assertAlmostEqual(health.execution_p95_ms, 61.0)
        self.assertEqual(health.interval_deadline_misses, 1)
        self.assertEqual(health.execution_deadline_misses, 1)

    def test_periodic_report_returns_snapshot(self) -> None:
        telemetry = ControlLoopTelemetry(0.05, report_period_s=0.10)
        telemetry.begin(1.0)
        self.assertIsNone(telemetry.finish(1.0, 1.01))
        telemetry.begin(1.10)

        report = telemetry.finish(1.10, 1.11)

        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report.sample_count, 2)


class InputDropFaultInjectorTests(unittest.TestCase):
    def test_blank_configuration_never_drops_live_inputs(self) -> None:
        injector = InputDropFaultInjector("")

        self.assertFalse(injector.enabled)
        self.assertFalse(injector.should_drop("odometry"))
        self.assertFalse(injector.should_drop("joint_states"))

    def test_markers_selectively_drop_and_restore_each_input(self) -> None:
        with TemporaryDirectory() as directory:
            injector = InputDropFaultInjector(directory)
            odom_marker = injector.marker_path("odometry")
            joints_marker = injector.marker_path("joint_states")

            odom_marker.touch()
            self.assertTrue(injector.should_drop("odometry"))
            self.assertFalse(injector.should_drop("joint_states"))
            odom_marker.unlink()
            joints_marker.touch()
            self.assertFalse(injector.should_drop("odometry"))
            self.assertTrue(injector.should_drop("joint_states"))
            joints_marker.unlink()
            self.assertFalse(injector.should_drop("joint_states"))

    def test_unknown_marker_name_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            injector = InputDropFaultInjector(directory)
            with self.assertRaises(KeyError):
                injector.marker_path("detections")


class RuntimeHealthWiringTests(unittest.TestCase):
    def test_formal_client_uses_one_common_guard_before_controller_tick(self) -> None:
        client = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "material_sorting"
            / "client_task.py"
        ).read_text(encoding="utf-8")

        guard_index = client.index("freshness = self._freshness_watchdog.evaluate")
        controller_index = client.index("snapshot = self.controller.tick", guard_index)
        self.assertLess(guard_index, controller_index)
        self.assertIn('self._freshness_watchdog.observe("odometry"', client)
        self.assertIn('self._freshness_watchdog.observe("joint_states"', client)
        self.assertIn("self._publish_held_arm_if_valid()", client[guard_index:controller_index])
        self.assertIn("input_ages_s=dict(freshness.ages_s)", client)


if __name__ == "__main__":
    unittest.main()
