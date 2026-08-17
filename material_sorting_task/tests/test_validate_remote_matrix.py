from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "validate_remote_matrix", SCRIPTS / "validate_remote_matrix.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


CLIENT = """
controller=waiting_for_referee task=1 attempt=1 stage=- score=40
controller=waiting_for_referee task=2 attempt=1 stage=- score=100
controller=finished task=3 attempt=1 stage=- score=160
"""
SERVER = "all_tasks_done total 160 = task1 40 + task2 60 + task3 60\n"


def write_run(root: Path, seed: int, *, client: str = CLIENT) -> None:
    name = f"v2_multiseed_{seed}"
    run = root / name
    run.mkdir()
    (run / f"client_{name}.log").write_text(client, encoding="utf-8")
    (run / f"server_{name}.log").write_text(SERVER, encoding="utf-8")


class ValidateRemoteMatrixTests(unittest.TestCase):
    def test_accepts_when_every_seed_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root, 1)
            write_run(root, 2)

            report = MODULE.validate_matrix(root, [1, 2])

        self.assertTrue(report["passed"])
        self.assertEqual(report["passed_seed_count"], 2)

    def test_reports_missing_and_failed_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root, 1, client=CLIENT.replace("score=160", "score=140"))

            report = MODULE.validate_matrix(root, [1, 2])

        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_seeds"], ["1", "2"])
