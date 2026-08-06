"""Task 2 executor placeholder.

Future implementation should perform shelf-layer alignment, shelf grasp,
transport, placement at task 1's original table-side point, and safe return.
"""

from executors.base import PlaceholderTaskExecutor


class Task2Executor(PlaceholderTaskExecutor):
    task_id = 2
    name = "task2_shelf_to_original_table_point"
