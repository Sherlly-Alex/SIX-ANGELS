from __future__ import annotations

import numpy as np

from competition_controller import ControllerSnapshot, ControllerState
from executors.base import TargetObservation, TaskStage
from navigation.dynamic_overlay import volumes_from_detections
from perception.shelf_empty_confirm import EMPTY, OCCUPIED, UNKNOWN, ShelfEmptyLayerVerifier
from shelf.state_tracker import ShelfStateTracker


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
CAMERA_MATRIX = np.array(
    (
        (300.0, 0.0, IMAGE_WIDTH / 2.0),
        (0.0, 300.0, IMAGE_HEIGHT / 2.0),
        (0.0, 0.0, 1.0),
    ),
    dtype=float,
)
CAMERA_WORLD_TMAT = np.array(
    (
        (0.0, 0.0, -1.0, -1.60),
        (1.0, 0.0, 0.0, 0.778),
        (0.0, -1.0, 0.0, 0.90),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=float,
)


def _render_depth(
    verifier: ShelfEmptyLayerVerifier,
    *,
    occupied_layers: tuple[int, ...] = (),
    occluded_layers: tuple[int, ...] = (),
    partially_occluded_layers: tuple[int, ...] = (),
    rear_plane_x: float = -2.84,
) -> np.ndarray:
    pixel_u, pixel_v = np.meshgrid(
        np.arange(IMAGE_WIDTH, dtype=np.float32),
        np.arange(IMAGE_HEIGHT, dtype=np.float32),
    )
    rays_camera = np.stack(
        (
            (pixel_u - CAMERA_MATRIX[0, 2]) / CAMERA_MATRIX[0, 0],
            (pixel_v - CAMERA_MATRIX[1, 2]) / CAMERA_MATRIX[1, 1],
            np.ones_like(pixel_u),
        ),
        axis=-1,
    )
    rays_world = rays_camera @ CAMERA_WORLD_TMAT[:3, :3].T
    camera_origin = CAMERA_WORLD_TMAT[:3, 3]

    def plane_depth(plane_x: float) -> tuple[np.ndarray, np.ndarray]:
        distance = (plane_x - camera_origin[0]) / rays_world[..., 0]
        points = camera_origin + distance[..., None] * rays_world
        return distance, points

    depth, _rear_points = plane_depth(rear_plane_x)
    shelf_y = verifier.geometry.shelf_xy[1]
    for layer in occupied_layers:
        object_depth, object_points = plane_depth(-2.52)
        board_z = verifier.geometry.board_z(layer)
        object_mask = (
            (np.abs(object_points[..., 1] - shelf_y) <= 0.125)
            & (object_points[..., 2] >= board_z + 0.02)
            & (object_points[..., 2] <= board_z + 0.225)
        )
        depth = np.where(object_mask, object_depth, depth)
    for layer in occluded_layers:
        occluder_depth, occluder_points = plane_depth(-2.20)
        board_z = verifier.geometry.board_z(layer)
        occluder_mask = (
            (np.abs(occluder_points[..., 1] - shelf_y) <= 0.30)
            & (occluder_points[..., 2] >= board_z - 0.20)
            & (occluder_points[..., 2] <= board_z + 0.50)
        )
        depth = np.where(occluder_mask, occluder_depth, depth)
    for layer in partially_occluded_layers:
        occluder_depth, occluder_points = plane_depth(-2.20)
        board_z = verifier.geometry.board_z(layer)
        occluder_mask = (
            (np.abs(occluder_points[..., 1] - shelf_y) <= 0.05)
            & (occluder_points[..., 2] >= board_z + 0.01)
            & (occluder_points[..., 2] <= board_z + 0.28)
        )
        depth = np.where(occluder_mask, occluder_depth, depth)
    return np.clip(np.rint(depth * 1000.0), 0, 65535).astype(np.uint16)


def _observation(label: str, xyz, stamp: float) -> TargetObservation:
    return TargetObservation(
        color=label,
        position_world=tuple(float(value) for value in xyz),
        received_at_s=float(stamp),
        score=0.9,
    )


def test_controller_requests_recognition_for_complete_task1_shelf_window() -> None:
    def snapshot(task_id: int, stage: TaskStage) -> ControllerSnapshot:
        return ControllerSnapshot(
            state=ControllerState.EXECUTING_STAGE,
            task_index=0,
            task_id=task_id,
            attempt=1,
            stage=stage,
            safe_stop=True,
            controls_base=False,
            base_linear_x=0.0,
            base_angular_z=0.0,
            controls_arm=False,
            arm_command=None,
            message="",
            transition_serial=0,
        )

    assert snapshot(1, TaskStage.TRANSPORT).requests_shelf_recognition
    assert snapshot(1, TaskStage.ALIGN_FOR_PLACE).requests_shelf_recognition
    assert not snapshot(1, TaskStage.PLACE).requests_shelf_recognition
    assert not snapshot(2, TaskStage.TRANSPORT).requests_shelf_recognition


def test_rgbd_verifier_is_idle_outside_shelf_recognition_session() -> None:
    verifier = ShelfEmptyLayerVerifier()
    depth = _render_depth(verifier, occupied_layers=(1, 2))

    assert verifier.update(depth, CAMERA_MATRIX, CAMERA_WORLD_TMAT) == {}
    assert verifier.confirmed_layer is None


def test_rgbd_verifier_confirms_only_unique_visible_empty_layer() -> None:
    verifier = ShelfEmptyLayerVerifier()
    verifier.start()
    depth = _render_depth(verifier, occupied_layers=(1, 2))

    for _ in range(4):
        evidence = verifier.update(depth, CAMERA_MATRIX, CAMERA_WORLD_TMAT)

    assert evidence[1].status == OCCUPIED
    assert evidence[2].status == OCCUPIED
    assert evidence[3].status == EMPTY
    assert verifier.confirmed_layer == 3
    assert verifier.confirmed_center_world() is not None


def test_rgbd_verifier_accepts_gs_rear_surface_without_hiding_objects() -> None:
    verifier = ShelfEmptyLayerVerifier()
    verifier.start()
    depth = _render_depth(
        verifier,
        occupied_layers=(1, 2),
        rear_plane_x=-2.71,
    )

    for _ in range(4):
        evidence = verifier.update(depth, CAMERA_MATRIX, CAMERA_WORLD_TMAT)

    assert evidence[1].status == OCCUPIED
    assert evidence[2].status == OCCUPIED
    assert evidence[3].status == EMPTY
    assert verifier.confirmed_layer == 3


def test_rgbd_verifier_confirms_each_layer_with_gs_rear_surface() -> None:
    for empty_layer in (1, 2, 3):
        verifier = ShelfEmptyLayerVerifier()
        verifier.start()
        occupied_layers = tuple(
            layer for layer in (1, 2, 3) if layer != empty_layer
        )
        depth = _render_depth(
            verifier,
            occupied_layers=occupied_layers,
            rear_plane_x=-2.71,
        )

        for _ in range(4):
            evidence = verifier.update(depth, CAMERA_MATRIX, CAMERA_WORLD_TMAT)

        assert evidence[empty_layer].status == EMPTY
        assert verifier.confirmed_layer == empty_layer
        for occupied_layer in occupied_layers:
            assert evidence[occupied_layer].status == OCCUPIED


def test_rgbd_verifier_never_treats_occlusion_as_empty() -> None:
    verifier = ShelfEmptyLayerVerifier()
    verifier.start()
    depth = _render_depth(verifier, occluded_layers=(3,))

    evidence = verifier.update(depth, CAMERA_MATRIX, CAMERA_WORLD_TMAT)

    assert evidence[3].status == UNKNOWN
    assert evidence[3].occluder_ratio > verifier.MAX_OCCLUDER_RATIO
    assert verifier.confirmed_layer is None


def test_rgbd_verifier_excludes_partial_carried_object_occlusion() -> None:
    verifier = ShelfEmptyLayerVerifier()
    verifier.start()
    depth = _render_depth(
        verifier,
        occupied_layers=(1, 2),
        partially_occluded_layers=(3,),
    )

    for _ in range(4):
        evidence = verifier.update(depth, CAMERA_MATRIX, CAMERA_WORLD_TMAT)

    assert evidence[3].status == EMPTY
    assert evidence[3].occluder_ratio > 0.0
    assert evidence[3].usable_ratio >= verifier.MIN_USABLE_RATIO
    assert verifier.confirmed_layer == 3


def test_carried_object_exclusion_does_not_hide_real_shelf_occupancy() -> None:
    verifier = ShelfEmptyLayerVerifier()
    verifier.start()
    depth = _render_depth(
        verifier,
        occupied_layers=(3,),
        partially_occluded_layers=(3,),
    )

    evidence = verifier.update(depth, CAMERA_MATRIX, CAMERA_WORLD_TMAT)

    assert evidence[3].status == OCCUPIED
    assert evidence[3].occluder_ratio > 0.0
    assert verifier.confirmed_layer is None


def test_rgbd_verifier_never_treats_missing_depth_as_empty() -> None:
    verifier = ShelfEmptyLayerVerifier()
    verifier.start()
    depth = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint16)

    evidence = verifier.update(depth, CAMERA_MATRIX, CAMERA_WORLD_TMAT)

    assert all(item.status == UNKNOWN for item in evidence.values())
    assert verifier.confirmed_layer is None


def test_shelf_state_requires_matching_empty_confirmation_when_enabled() -> None:
    tracker = ShelfStateTracker(
        required_votes=3,
        require_empty_confirmation=True,
    )
    result = None
    for stamp in (1.0, 2.0, 3.0):
        result = tracker.update(
            {
                "brown": _observation("brown", (-2.55, 0.81, 0.837), stamp),
                "packaging_box": _observation(
                    "packaging_box", (-2.54, 0.78, 0.530), stamp
                ),
            },
            now_s=stamp,
            carried_class_id="yellow",
        )
    assert result is None

    for stamp in (4.0, 5.0, 6.0):
        result = tracker.update(
            {
                "brown": _observation("brown", (-2.55, 0.81, 0.837), stamp),
                "packaging_box": _observation(
                    "packaging_box", (-2.54, 0.78, 0.530), stamp
                ),
                "shelf_empty": _observation(
                    "shelf_empty", (-2.63, 0.778, 1.166), stamp
                ),
            },
            now_s=stamp,
            carried_class_id="yellow",
        )

    assert result is not None
    assert result.empty_layer == 3


def test_shelf_state_rejects_empty_confirmation_on_wrong_layer() -> None:
    tracker = ShelfStateTracker(
        required_votes=3,
        require_empty_confirmation=True,
    )
    result = None
    for stamp in (1.0, 2.0, 3.0):
        result = tracker.update(
            {
                "brown": _observation("brown", (-2.55, 0.81, 0.837), stamp),
                "packaging_box": _observation(
                    "packaging_box", (-2.54, 0.78, 0.530), stamp
                ),
                "shelf_empty": _observation(
                    "shelf_empty", (-2.63, 0.778, 0.837), stamp
                ),
            },
            now_s=stamp,
            carried_class_id="yellow",
        )

    assert result is None


def test_shelf_state_does_not_revote_one_stale_empty_observation() -> None:
    tracker = ShelfStateTracker(
        required_votes=3,
        require_empty_confirmation=True,
    )
    stale_empty = _observation(
        "shelf_empty", (-2.63, 0.778, 1.166), 1.0
    )
    result = None
    for stamp in (1.0, 2.0, 3.0, 4.0):
        result = tracker.update(
            {
                "brown": _observation(
                    "brown", (-2.55, 0.81, 0.837), stamp
                ),
                "packaging_box": _observation(
                    "packaging_box", (-2.54, 0.78, 0.530), stamp
                ),
                "shelf_empty": stale_empty,
            },
            now_s=stamp,
            carried_class_id="yellow",
        )

    assert result is None
    assert "empty=L3(1/3)" in tracker.diagnostic_summary


def test_empty_confirmation_is_not_added_to_navigation_obstacles() -> None:
    volumes = volumes_from_detections(
        (("shelf_empty", (-2.63, 0.778, 1.166), 0.9),)
    )

    assert volumes == []
