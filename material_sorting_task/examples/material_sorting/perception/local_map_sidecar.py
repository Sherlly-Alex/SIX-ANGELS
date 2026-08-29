#!/usr/bin/env python3
"""Optional local-map sidecar for the formal client (fail-open, default off).

Environment
-----------
MATERIAL_LOCAL_MAP=0|1
    When off (default): no map object, no depth work, advice is always None.
    When on: ``box_detect`` fuses depth (same decode as YOLO/RGB-D) and publishes
    ``/material/local_map_advice``; ``client_task`` relays without a second depth
    subscription.
MATERIAL_LOCAL_MAP_APPLY=0|1
    When off (default): advice may be computed/logged but must not change motion.
    When on: ``_shelf_clearance_m`` may widen task-2 shelf/table standoffs and
    ``local_map_linear_scale`` may boost or slow transit segments.  Task-1
    placement keeps a fixed ``SHELF_SCAN_CENTER_CLEARANCE_M``; task 3 keeps a
    fixed scan-stand X because align-for-place reuses it.
MATERIAL_LOCAL_MAP_HZ
    Max depth integrate rate (default 0.5). Lower values reduce Client CPU load.
MATERIAL_LOCAL_MAP_SPEED_MODE=boost_only|full
    ``boost_only`` (default): fresh clear corridor → faster transit only; hits
    do not slow base motion (A* grid still handles static obstacles).
    ``full``: also slow near forward hits (legacy safety profile).
MATERIAL_LOCAL_MAP_CLEAR_BOOST
    Linear multiplier when ``clear=True`` (default 1.40).
MATERIAL_LOCAL_MAP_STRIDE
    Depth subsample stride for fusion (default 8).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

try:
    from perception.local_map import RollingLocalHeightMap, local_map_enabled, local_map_hz
except ImportError:
    from local_map import RollingLocalHeightMap, local_map_enabled, local_map_hz  # type: ignore


# Tier-2 speed apply (same APPLY+fresh gate as standoff).
LOCAL_MAP_CLEAR_LINEAR_BOOST = 1.40
LOCAL_MAP_NEAR_HIT_LINEAR_SCALE = 0.52
LOCAL_MAP_NEAR_HIT_MAX_DIST_M = 0.38
LOCAL_MAP_MID_HIT_LINEAR_SCALE = 0.85
LOCAL_MAP_SPEED_APPLY_MAX_DIST_M = 1.00


def local_map_apply_enabled(default: bool = False) -> bool:
    raw = os.environ.get("MATERIAL_LOCAL_MAP_APPLY")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return value if math.isfinite(value) else float(default)


def local_map_speed_mode(default: str = "boost_only") -> str:
    """Return ``boost_only`` or ``full`` for transit linear scaling."""

    raw = os.environ.get("MATERIAL_LOCAL_MAP_SPEED_MODE")
    if raw is None or not str(raw).strip():
        return default
    mode = str(raw).strip().lower()
    if mode in {"full", "boost_and_slow", "slow"}:
        return "full"
    return "boost_only"


def local_map_clear_boost(default: float = LOCAL_MAP_CLEAR_LINEAR_BOOST) -> float:
    value = _env_float("MATERIAL_LOCAL_MAP_CLEAR_BOOST", default)
    return max(1.0, min(2.0, float(value)))


def _pose_from_odometry(odometry: Any) -> Optional[tuple[float, float, float]]:
    if odometry is None:
        return None
    try:
        p = odometry.pose.pose.position
        q = odometry.pose.pose.orientation
        x, y = float(p.x), float(p.y)
        # yaw from quaternion (ROS: x,y,z,w)
        siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
        cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
        yaw = math.atan2(siny_cosp, cosy_cosp)
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            return None
        return (x, y, yaw)
    except (AttributeError, TypeError, ValueError):
        return None


@dataclass(frozen=True)
class LocalMapAdvice:
    """Read-only near-field hint attached to ``ExecutionContext``."""

    enabled: bool
    apply: bool
    fresh: bool
    clear: Optional[bool]
    distance_m: Optional[float]
    suggested_standoff_m: Optional[float]
    frames_accepted: int = 0
    frames_rejected: int = 0
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "apply": self.apply,
            "fresh": self.fresh,
            "clear": self.clear,
            "distance_m": self.distance_m,
            "suggested_standoff_m": self.suggested_standoff_m,
            "frames_accepted": self.frames_accepted,
            "frames_rejected": self.frames_rejected,
            "reason": self.reason,
        }


def applied_standoff_m(
    advice: LocalMapAdvice | Mapping[str, Any] | None,
    fallback_m: float,
) -> float:
    """Return map standoff only when enabled+apply+fresh; else fallback.

    Fail-open: any missing/invalid advice returns ``fallback_m`` unchanged.
    """
    try:
        fb = float(fallback_m)
    except (TypeError, ValueError):
        return float(fallback_m)
    if advice is None:
        return fb
    if isinstance(advice, LocalMapAdvice):
        data = advice.as_dict()
    elif isinstance(advice, Mapping):
        data = dict(advice)
    else:
        return fb
    if not data.get("enabled") or not data.get("apply") or not data.get("fresh"):
        return fb
    value = data.get("suggested_standoff_m")
    try:
        suggested = float(value)
    except (TypeError, ValueError):
        return fb
    if not math.isfinite(suggested) or suggested <= 0.0:
        return fb
    return suggested


def _advice_mapping(
    advice: LocalMapAdvice | Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if advice is None:
        return None
    if isinstance(advice, LocalMapAdvice):
        return advice.as_dict()
    if isinstance(advice, Mapping):
        return advice
    return None


def local_map_linear_scale(
    advice: LocalMapAdvice | Mapping[str, Any] | None,
) -> float:
    """Fail-open linear multiplier when APPLY sees a fresh corridor snapshot.

    * ``clear=True`` → transit boost (``MATERIAL_LOCAL_MAP_CLEAR_BOOST``).
    * ``clear=False`` → slowdown only when ``MATERIAL_LOCAL_MAP_SPEED_MODE=full``.
    * otherwise → 1.0 (unchanged).
    """

    data = _advice_mapping(advice)
    if data is None:
        return 1.0
    if not data.get("enabled") or not data.get("apply") or not data.get("fresh"):
        return 1.0
    if data.get("clear") is True:
        return local_map_clear_boost()
    if local_map_speed_mode() != "full":
        return 1.0
    if data.get("clear") is not False:
        return 1.0
    try:
        hit_dist = float(data.get("distance_m"))
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(hit_dist) or hit_dist >= LOCAL_MAP_SPEED_APPLY_MAX_DIST_M:
        return 1.0
    if hit_dist <= LOCAL_MAP_NEAR_HIT_MAX_DIST_M:
        return LOCAL_MAP_NEAR_HIT_LINEAR_SCALE
    return LOCAL_MAP_MID_HIT_LINEAR_SCALE


class LocalMapSidecar:
    """Client-owned optional local map. Never raises into the control loop."""

    def __init__(self, log=None) -> None:
        self._log = log
        self.enabled = local_map_enabled(False)
        self.apply = local_map_apply_enabled(False) if self.enabled else False
        self._map: RollingLocalHeightMap | None = None
        self._latest_depth = None
        self._latest_K = None
        self._latest_depth_stamp_s: float | None = None
        self._last_integrate_attempt_s = 0.0
        self._hz = local_map_hz(0.5) if self.enabled else 0.0
        self._min_interval_s = (1.0 / self._hz) if self._hz > 0.0 else 0.0
        self._last_advice = LocalMapAdvice(
            enabled=False,
            apply=False,
            fresh=False,
            clear=None,
            distance_m=None,
            suggested_standoff_m=None,
            reason="disabled",
        )
        self._last_error = ""
        if self.enabled:
            try:
                self._map = RollingLocalHeightMap.from_env()
            except Exception as exc:  # pragma: no cover - defensive
                self.enabled = False
                self.apply = False
                self._map = None
                self._last_error = f"init_failed:{type(exc).__name__}"
                self._info(f"local map disabled after init failure: {exc}")

    def describe(self) -> str:
        if not self.enabled:
            return "local map disabled (MATERIAL_LOCAL_MAP default off)"
        mode = "apply-on" if self.apply else "observe-only"
        return f"local map enabled ({mode}, {self._hz:g}Hz, fail-open)"

    def _info(self, message: str) -> None:
        if self._log is None:
            return
        try:
            self._log(message)
        except Exception:
            return

    def set_depth_image(self, depth_img, stamp_s: float | None = None) -> None:
        if not self.enabled:
            return
        try:
            self._latest_depth = np.asarray(depth_img)
            if stamp_s is not None and math.isfinite(float(stamp_s)):
                self._latest_depth_stamp_s = float(stamp_s)
        except Exception as exc:
            self._last_error = f"depth_set:{type(exc).__name__}"
            return

    def set_camera_info_K(self, k_matrix) -> None:
        if not self.enabled:
            return
        try:
            K = np.asarray(k_matrix, dtype=float).reshape(3, 3)
            if not np.all(np.isfinite(K)):
                return
            self._latest_K = K
        except Exception as exc:
            self._last_error = f"K_set:{type(exc).__name__}"
            return

    def _advice_from_map(
        self,
        *,
        now_s: float,
        pose: tuple[float, float, float],
        integrated: bool,
        gate_reason: str,
    ) -> LocalMapAdvice:
        assert self._map is not None
        clearance = self._map.forward_clearance(pose)
        standoff = self._map.suggested_standoff(pose)
        fresh = bool(integrated) or (
            self._map.last_integrate_s is not None
            and (float(now_s) - float(self._map.last_integrate_s)) <= 2.0
        )
        if fresh:
            reason = "ok"
        elif gate_reason:
            reason = gate_reason
        elif self._last_error:
            reason = self._last_error
        else:
            reason = "stale"
        if (
            not fresh
            and self._map.frames_accepted == 0
            and clearance.clear
        ):
            reason = f"{reason}|empty_map_defaults"
        return LocalMapAdvice(
            enabled=True,
            apply=self.apply,
            fresh=fresh,
            clear=bool(clearance.clear),
            distance_m=float(clearance.distance_m),
            suggested_standoff_m=float(standoff),
            frames_accepted=self._map.frames_accepted,
            frames_rejected=self._map.frames_rejected,
            reason=reason,
        )

    def on_tick(
        self,
        *,
        now_s: float,
        odometry: Any = None,
        t_cam_world: Any = None,
    ) -> LocalMapAdvice:
        """Update map if possible and return advice (never raises)."""
        if not self.enabled or self._map is None:
            self._last_advice = LocalMapAdvice(
                enabled=False,
                apply=False,
                fresh=False,
                clear=None,
                distance_m=None,
                suggested_standoff_m=None,
                reason="disabled",
            )
            return self._last_advice

        try:
            pose = _pose_from_odometry(odometry)
            if pose is None:
                self._last_advice = LocalMapAdvice(
                    enabled=True,
                    apply=self.apply,
                    fresh=False,
                    clear=None,
                    distance_m=None,
                    suggested_standoff_m=None,
                    frames_accepted=self._map.frames_accepted,
                    frames_rejected=self._map.frames_rejected,
                    reason="no_pose",
                )
                return self._last_advice

            self._map.seed_pose(pose)
            self._map._decay(float(now_s))

            throttled = (
                self._min_interval_s > 0.0
                and self._last_integrate_attempt_s > 0.0
                and (float(now_s) - self._last_integrate_attempt_s)
                < self._min_interval_s
            )
            if throttled:
                self._last_advice = self._advice_from_map(
                    now_s=float(now_s),
                    pose=pose,
                    integrated=False,
                    gate_reason="throttled",
                )
                return self._last_advice

            self._last_integrate_attempt_s = float(now_s)
            integrated = False
            gate_reason = ""
            if self._latest_depth is None:
                gate_reason = "no_depth"
            elif self._latest_K is None:
                gate_reason = "no_K"
            elif t_cam_world is None:
                gate_reason = "no_T_cam_world"
            else:
                T = np.asarray(t_cam_world, dtype=float)
                if T.shape != (4, 4) or not np.all(np.isfinite(T)):
                    gate_reason = "bad_T_cam_world"
                    self._last_error = gate_reason
                else:
                    status = self._map.integrate_depth(
                        self._latest_depth,
                        self._latest_K,
                        T,
                        pose,
                        float(now_s),
                    )
                    integrated = bool(status.get("accepted"))
                    if not integrated:
                        gate_reason = str(status.get("reason") or "integrate_rejected")

            if not integrated and gate_reason != "throttled":
                # Keep the window attached for queries on previously fused cells.
                pass

            self._last_advice = self._advice_from_map(
                now_s=float(now_s),
                pose=pose,
                integrated=integrated,
                gate_reason=gate_reason,
            )
            return self._last_advice
        except Exception as exc:
            self._last_error = f"tick:{type(exc).__name__}"
            self._last_advice = LocalMapAdvice(
                enabled=True,
                apply=False,  # fail-open: never apply after exception
                fresh=False,
                clear=None,
                distance_m=None,
                suggested_standoff_m=None,
                reason=self._last_error,
            )
            return self._last_advice

    @property
    def last_advice(self) -> LocalMapAdvice:
        return self._last_advice


__all__ = [
    "LocalMapAdvice",
    "LocalMapSidecar",
    "applied_standoff_m",
    "local_map_apply_enabled",
    "local_map_clear_boost",
    "local_map_speed_mode",
]
