#!/usr/bin/env python3
"""Create a deterministic, session-isolated RL-1 replay split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "material_sorting"
sys.path.insert(0, str(EXAMPLE))
from learning.replay_split import split_replay_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-sessions", type=int, default=3)
    parser.add_argument("--validation-sessions", type=int, default=1)
    parser.add_argument("--test-sessions", type=int, default=1)
    args = parser.parse_args(argv)
    result = split_replay_dataset(
        args.dataset, args.output_dir, seed=args.seed,
        train_sessions=args.train_sessions,
        validation_sessions=args.validation_sessions,
        test_sessions=args.test_sessions,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
