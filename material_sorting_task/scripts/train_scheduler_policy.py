#!/usr/bin/env python3
"""Stable project-root CLI for optional offline scheduler training."""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SCRIPT_DIR.parent / "examples" / "material_sorting"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from learning.train_maskable_ppo import main


if __name__ == "__main__":
    raise SystemExit(main())
