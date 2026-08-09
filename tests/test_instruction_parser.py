from __future__ import annotations

import json
import unittest

from instruction_parser import (
    InstructionParseError,
    parse_instruction_message,
    validate_instruction,
)


def _payload() -> str:
    return json.dumps(
        [
            {
                "task": 1,
                "instruction": "抓取桌面左侧的粉色方块，放到货架空层",
                "target_kind": "cuboid_box",
                "target_body": "box_pink",
                "target_color": "pink",
                "place_type": "shelf_point",
                "place_world": [-2.68, 0.778, 0.498],
                "place_radius": 0.24,
            },
            {
                "task": 2,
                "instruction": "抓取货架中的黄色方块，放到第一个方块原来的位置",
                "target_kind": "cuboid_box",
                "target_body": "box_yellow",
                "target_color": "yellow",
                "place_type": "table_point",
                "place_world": [-1.0, 2.2, 0.834],
                "place_radius": 0.28,
            },
            {
                "task": 3,
                "instruction": "抓取白色正方体顶部的褐色方块，放到白色长方体左边",
                "target_kind": "cuboid_box",
                "target_body": "box_brown",
                "target_color": "brown",
                "ref_prop_body": "prop_packaging_box",
                "direction": "left",
                "place_type": "shelf_prop_side",
                "place_world": [-2.68, 0.54, 1.156],
                "place_radius": 0.24,
            },
        ],
        ensure_ascii=False,
    )


class InstructionParserTests(unittest.TestCase):
    def test_parses_three_execution_ready_tasks(self) -> None:
        tasks = parse_instruction_message(_payload())

        self.assertEqual([task.task for task in tasks], [1, 2, 3])
        self.assertEqual(
            [task.target_color for task in tasks], ["pink", "yellow", "brown"]
        )
        for task in tasks:
            validate_instruction(task, require_execution_ready=True)

    def test_rejects_malformed_json(self) -> None:
        with self.assertRaises(InstructionParseError):
            parse_instruction_message("[{]")


if __name__ == "__main__":
    unittest.main()
