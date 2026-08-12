from __future__ import annotations

import json
import unittest

from competition_controller import CompetitionController, ControllerState
from executors import build_task_executors
from instruction_parser import (
    InstructionParseError,
    InstructionValidationError,
    parse_instruction_message,
    validate_instruction,
)
from task_orchestration import sorted_instructions


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


def _items() -> list[dict]:
    return json.loads(_payload())


def _one(**overrides) -> dict:
    item = dict(_items()[0])
    item.update(overrides)
    return item


def _context(controller: CompetitionController, tasks: list[dict]):
    from competition_controller import ExecutionContext

    index = min(controller.task_index, len(tasks) - 1)
    return ExecutionContext(
        now_s=0.0,
        instruction=tasks[index],
        task_index=index,
        attempt=controller.attempt,
        referee_gameinfo={"attempt": 0, "raw": ""},
        referee_taskinfo="",
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

    def test_execution_ready_requires_complete_official_contract(self) -> None:
        payload = json.loads(_payload())
        for field in ("task", "target_body", "place_radius"):
            item = dict(payload[0])
            item.pop(field)
            task = parse_instruction_message(json.dumps([item]))[0]
            with self.assertRaisesRegex(InstructionValidationError, field):
                validate_instruction(task, require_execution_ready=True)

    def test_json_missing_target_color_not_filled_from_chinese(self) -> None:
        item = _one()
        item.pop("target_color")
        task = parse_instruction_message(json.dumps([item], ensure_ascii=False))[0]
        self.assertNotIn("target_color", task.json_fields)
        self.assertIsNone(task.target_color)
        self.assertEqual(task.source, "structured")
        with self.assertRaisesRegex(InstructionValidationError, "JSON field target_color"):
            validate_instruction(task, require_execution_ready=True)

    def test_json_missing_place_type_not_filled_from_chinese(self) -> None:
        item = _one()
        item.pop("place_type")
        task = parse_instruction_message(json.dumps([item], ensure_ascii=False))[0]
        self.assertNotIn("place_type", task.json_fields)
        self.assertIsNone(task.place_type)
        self.assertEqual(task.source, "structured")
        with self.assertRaisesRegex(InstructionValidationError, "JSON field place_type"):
            validate_instruction(task, require_execution_ready=True)

    def test_json_missing_direction_not_filled_from_chinese(self) -> None:
        item = dict(_items()[2])
        item.pop("direction")
        task = parse_instruction_message(json.dumps([item], ensure_ascii=False))[0]
        self.assertNotIn("direction", task.json_fields)
        self.assertIsNone(task.direction)
        self.assertEqual(task.source, "structured")
        with self.assertRaisesRegex(InstructionValidationError, "JSON field direction"):
            validate_instruction(task, require_execution_ready=True)

    def test_json_missing_ref_not_filled_from_chinese(self) -> None:
        item = dict(_items()[2])
        item.pop("ref_prop_body", None)
        item.pop("ref_prop", None)
        task = parse_instruction_message(json.dumps([item], ensure_ascii=False))[0]
        self.assertNotIn("ref_prop", task.json_fields)
        self.assertNotIn("ref_prop_body", task.json_fields)
        self.assertIsNone(task.ref_prop)
        self.assertIsNone(task.ref_prop_body)
        self.assertEqual(task.source, "structured")
        with self.assertRaisesRegex(
            InstructionValidationError, "JSON field ref_prop"
        ):
            validate_instruction(task, require_execution_ready=True)

    # ── 载荷层 ──

    def test_rejects_empty_payload(self) -> None:
        with self.assertRaises(InstructionParseError):
            parse_instruction_message("")
        with self.assertRaises(InstructionParseError):
            parse_instruction_message("   ")

    def test_rejects_empty_json_array(self) -> None:
        with self.assertRaisesRegex(InstructionParseError, "empty"):
            parse_instruction_message("[]")

    def test_rejects_non_object_array_item(self) -> None:
        with self.assertRaisesRegex(InstructionParseError, "object"):
            parse_instruction_message(json.dumps([1, 2, 3]))

    def test_single_object_still_parses(self) -> None:
        item = _one()
        tasks = parse_instruction_message(json.dumps(item, ensure_ascii=False))
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task, 1)
        self.assertEqual(tasks[0].target_color, "pink")

    # ── 任务集合层（Client/Controller 约束） ──

    def test_task_set_missing_task_rejected_by_configure(self) -> None:
        items = _items()[:2]
        with self.assertRaises(ValueError):
            CompetitionController(
                build_task_executors("stub"), referee_driven=True
            ).configure(items)

    def test_task_set_duplicate_tasks_rejected_by_configure(self) -> None:
        items = _items()
        items[2] = dict(items[0])
        items[2]["task"] = 1
        with self.assertRaises(ValueError):
            CompetitionController(
                build_task_executors("stub"), referee_driven=True
            ).configure(items)

    def test_task_set_out_of_order_accepted_after_sort(self) -> None:
        items = list(reversed(_items()))
        sorted_ids = [t.get("task") for t in sorted_instructions(items)]
        self.assertEqual(sorted_ids, [1, 2, 3])
        controller = CompetitionController(
            build_task_executors("stub"), referee_driven=True
        )
        self.assertTrue(controller.configure(items))

    def test_rejects_non_positive_or_non_int_task_ids(self) -> None:
        for bad_task in (True, 0, -1, 1.5):
            item = _one(task=bad_task)
            task = parse_instruction_message(json.dumps([item]))[0]
            with self.assertRaises(InstructionValidationError):
                validate_instruction(task, require_execution_ready=True)

    # ── 字符串层 ──

    def test_rejects_empty_instruction_string(self) -> None:
        with self.assertRaises(InstructionParseError):
            parse_instruction_message(json.dumps([_one(instruction="")]))

    def test_rejects_blank_target_body_when_execution_ready(self) -> None:
        task = parse_instruction_message(json.dumps([_one(target_body="   ")]))[0]
        with self.assertRaisesRegex(InstructionValidationError, "target_body"):
            validate_instruction(task, require_execution_ready=True)

    def test_rejects_unknown_color_place_type_direction(self) -> None:
        cases = (
            {
                "target_color": "purple",
                "instruction": "抓取方块，放到货架空层",
            },
            {
                "place_type": "floor_point",
                "instruction": "抓取粉色方块，放到地面",
            },
            {
                "target_color": "brown",
                "target_body": "box_brown",
                "place_type": "shelf_prop_side",
                "direction": "up",
                "ref_prop_body": "prop_packaging_box",
                "instruction": "抓取褐色方块，放到白色长方体旁",
            },
        )
        for overrides in cases:
            item = _one(**overrides)
            task = parse_instruction_message(
                json.dumps([item], ensure_ascii=False)
            )[0]
            with self.assertRaises(InstructionValidationError):
                validate_instruction(task, require_execution_ready=True)

    # ── 几何层 ──

    def test_rejects_invalid_place_world(self) -> None:
        for bad_world in (
            [1.0, 2.0],
            ["a", "b", "c"],
            [1.0, 2.0, float("nan")],
            [1.0, 2.0, float("inf")],
        ):
            with self.assertRaises(InstructionParseError):
                parse_instruction_message(
                    json.dumps([_one(place_world=bad_world)])
                )

    def test_rejects_invalid_place_radius(self) -> None:
        for bad_radius in (0, -0.1, True, float("nan"), float("inf")):
            item = _one(place_radius=bad_radius)
            # bool/NaN/Inf may survive parse then fail validate
            try:
                task = parse_instruction_message(json.dumps([item]))[0]
            except (InstructionParseError, ValueError, OverflowError):
                continue
            with self.assertRaises(InstructionValidationError):
                validate_instruction(task, require_execution_ready=True)

    # ── 跨字段层 ──

    def test_shelf_point_and_table_point_do_not_require_reference(self) -> None:
        for item in (_items()[0], _items()[1]):
            task = parse_instruction_message(json.dumps([item], ensure_ascii=False))[0]
            self.assertIsNone(task.ref_prop)
            validate_instruction(task, require_execution_ready=True)

    def test_shelf_prop_side_requires_direction_and_reference(self) -> None:
        base = dict(_items()[2])
        missing_direction = dict(base)
        missing_direction.pop("direction")
        # Avoid Chinese direction aliases so text cannot fill the slot.
        missing_direction["instruction"] = "抓取褐色方块，放到白色长方体旁"
        task = parse_instruction_message(
            json.dumps([missing_direction], ensure_ascii=False)
        )[0]
        with self.assertRaisesRegex(InstructionValidationError, "direction"):
            validate_instruction(task, require_execution_ready=True)

        missing_ref = dict(base)
        missing_ref.pop("ref_prop_body", None)
        missing_ref.pop("ref_prop", None)
        # Keep Chinese from mentioning packaging box so text cannot fill ref.
        missing_ref["instruction"] = "抓取褐色方块，放到货架左侧"
        task = parse_instruction_message(
            json.dumps([missing_ref], ensure_ascii=False)
        )[0]
        with self.assertRaisesRegex(InstructionValidationError, "ref_prop"):
            validate_instruction(task, require_execution_ready=True)

    def test_rejects_structured_color_conflict_with_chinese_text(self) -> None:
        item = _one(
            target_color="pink",
            instruction="抓取桌面左侧的黄色方块，放到货架空层",
        )
        with self.assertRaisesRegex(InstructionParseError, "target_color conflict"):
            parse_instruction_message(json.dumps([item], ensure_ascii=False))

    def test_rejects_structured_direction_conflict_with_chinese_text(self) -> None:
        item = dict(_items()[2])
        item["direction"] = "right"
        item["instruction"] = "抓取白色正方体顶部的褐色方块，放到白色长方体左边"
        with self.assertRaisesRegex(InstructionParseError, "direction conflict"):
            parse_instruction_message(json.dumps([item], ensure_ascii=False))

    def test_structured_fields_unchanged_by_chinese_synonyms(self) -> None:
        item = dict(_items()[2])
        item["instruction"] = "抓取白色正方体顶部的棕色方块，放到白色长方体左侧"
        task = parse_instruction_message(json.dumps([item], ensure_ascii=False))[0]
        self.assertEqual(task.target_color, "brown")
        self.assertEqual(task.direction, "left")
        self.assertEqual(task.place_type, "shelf_prop_side")
        self.assertEqual(task.place_world, (-2.68, 0.54, 1.156))
        self.assertEqual(task.place_radius, 0.24)
        self.assertEqual(task.target_body, "box_brown")
        validate_instruction(task, require_execution_ready=True)

    # ── 生命周期层 ──

    def test_repeated_identical_instructions_are_idempotent(self) -> None:
        tasks = _items()
        controller = CompetitionController(
            build_task_executors("dry_run", dry_run_ticks_per_stage=1),
            referee_driven=False,
        )
        self.assertTrue(controller.configure(tasks))
        controller.set_inputs_ready(True)
        controller.tick(_context(controller, tasks))
        self.assertEqual(controller.state, ControllerState.STARTING_TASK)
        self.assertFalse(controller.configure(tasks))
        self.assertEqual(controller.state, ControllerState.STARTING_TASK)
        self.assertEqual(controller.task_id, 1)

    def test_changed_instructions_rejected_after_execution_starts(self) -> None:
        tasks = _items()
        changed = json.loads(json.dumps(tasks))
        changed[0]["target_color"] = "yellow"
        changed[0]["instruction"] = "抓取桌面左侧的黄色方块，放到货架空层"
        controller = CompetitionController(
            build_task_executors("dry_run", dry_run_ticks_per_stage=1),
            referee_driven=False,
        )
        self.assertTrue(controller.configure(tasks))
        controller.set_inputs_ready(True)
        controller.tick(_context(controller, tasks))
        with self.assertRaisesRegex(RuntimeError, "changed after"):
            controller.configure(changed)
        self.assertEqual(controller.state, ControllerState.STARTING_TASK)
        self.assertEqual(controller.task_id, 1)


if __name__ == "__main__":
    unittest.main()
