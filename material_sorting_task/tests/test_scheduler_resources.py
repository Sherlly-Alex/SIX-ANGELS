from __future__ import annotations

import math
import unittest

from scheduler.models import BaseCommand, CommandFrame, FailureCode, Resource, WorldState
from scheduler.resources import (
    BaseCommandLease,
    CommandValidationError,
    CommandValidator,
    ResourceConflictError,
    ResourceManager,
)
from scheduler.safety import SafetySupervisor


class SchedulerResourceTests(unittest.TestCase):
    def test_multi_resource_acquire_is_atomic_on_conflict(self) -> None:
        manager = ResourceManager()
        manager.acquire({Resource.RIGHT_ARM}, owner="other")

        with self.assertRaises(ResourceConflictError):
            manager.acquire(
                {Resource.LEFT_ARM, Resource.RIGHT_ARM, Resource.GRIPPERS},
                owner="grasp",
            )

        self.assertIsNone(manager.owner_of(Resource.LEFT_ARM))
        self.assertIsNone(manager.owner_of(Resource.GRIPPERS))
        self.assertEqual(manager.owner_of(Resource.RIGHT_ARM), "other")

    def test_same_owner_reacquire_is_idempotent_and_release_is_scoped(self) -> None:
        manager = ResourceManager()
        manager.acquire({Resource.BASE, Resource.PERCEPTION}, owner="nav")
        manager.acquire({Resource.BASE}, owner="nav")

        released = manager.release({Resource.BASE}, owner="nav")

        self.assertEqual(released, frozenset({Resource.BASE}))
        self.assertEqual(manager.owner_of(Resource.PERCEPTION), "nav")

    def test_timed_resource_claim_expires(self) -> None:
        manager = ResourceManager()
        manager.acquire(
            {Resource.BASE},
            owner="nav",
            now_s=1.0,
            lease_duration_s=0.1,
        )

        self.assertEqual(manager.owner_of(Resource.BASE, now_s=1.1), "nav")
        self.assertIsNone(manager.owner_of(Resource.BASE, now_s=1.100001))

    def test_command_validator_enforces_claim_and_finite_values(self) -> None:
        manager = ResourceManager()
        validator = CommandValidator(manager)
        frame = CommandFrame("nav", base_command=BaseCommand(0.1, 0.2))

        with self.assertRaises(CommandValidationError) as missing:
            validator.validate(frame, now_s=1.0)
        self.assertIs(missing.exception.failure_code, FailureCode.RESOURCE_CONFLICT)

        manager.acquire({Resource.BASE}, owner="nav")
        self.assertIs(validator.validate(frame, now_s=1.0), frame)
        invalid = CommandFrame("nav", base_command=BaseCommand(math.nan, 0.0))
        with self.assertRaises(CommandValidationError) as nonfinite:
            validator.validate(invalid, now_s=1.0)
        self.assertIs(nonfinite.exception.failure_code, FailureCode.COMMAND_NON_FINITE)

    def test_base_command_lease_returns_zero_after_deadline(self) -> None:
        lease = BaseCommandLease(lease_duration_s=0.1)
        lease.renew("nav", BaseCommand(0.2, -0.1), 1.0)

        self.assertEqual(lease.resolve(1.1), BaseCommand(0.2, -0.1))
        self.assertEqual(lease.resolve(1.100001), BaseCommand.zero())

    def test_base_command_lease_rejects_competing_owner_until_expiry(self) -> None:
        lease = BaseCommandLease(0.1)
        lease.renew("first", BaseCommand(0.1, 0.0), 1.0)
        with self.assertRaises(ResourceConflictError):
            lease.renew("second", BaseCommand(0.0, 0.1), 1.05)
        lease.renew("second", BaseCommand(0.0, 0.1), 1.2)
        self.assertEqual(lease.snapshot(1.2).owner, "second")


class SchedulerSafetyTests(unittest.TestCase):
    def test_collision_has_highest_priority(self) -> None:
        supervisor = SafetySupervisor({"odom": 0.2})
        world = WorldState(
            now_s=1.0,
            unsafe_collision=True,
            input_ages_s={"odom": 5.0},
        )

        result = supervisor.check(world, BaseCommand(math.nan, 0.0))

        self.assertTrue(result.must_stop)
        self.assertIs(result.failure_code, FailureCode.UNSAFE_COLLISION)

    def test_optional_input_age_and_nonfinite_command_checks(self) -> None:
        supervisor = SafetySupervisor(max_input_age_s={"odom": 0.2})
        stale = supervisor.check(WorldState(now_s=1.0, input_ages_s={"odom": 0.21}))
        invalid = supervisor.check_command(BaseCommand(0.0, math.inf))
        safe = supervisor.check(WorldState(now_s=1.0, input_ages_s={"odom": 0.2}))

        self.assertIs(stale.failure_code, FailureCode.INPUT_STALE)
        self.assertIs(invalid.failure_code, FailureCode.COMMAND_NON_FINITE)
        self.assertTrue(safe.safe)
        self.assertFalse(safe)


if __name__ == "__main__":
    unittest.main()
