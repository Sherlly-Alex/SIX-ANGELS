#!/usr/bin/env python3
"""Bind model, blind benchmark and RL-shadow evidence for rl_guarded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SCRIPT_DIR.parent / "examples" / "material_sorting"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from learning.promotion import build_guarded_approval, file_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Approve one guarded RL scheduler model")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--shadow", required=True, type=Path)
    parser.add_argument("--minimum-blind-seeds", type=int, default=100)
    parser.add_argument("--minimum-shadow-suggestions", type=int, default=1000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build_guarded_approval(
        args.model,
        args.benchmark,
        args.shadow,
        minimum_blind_seeds=args.minimum_blind_seeds,
        minimum_shadow_suggestions=args.minimum_shadow_suggestions,
    )
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    print(f"approval_sha256={file_sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
