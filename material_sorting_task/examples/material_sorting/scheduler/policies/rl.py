"""Optional MaskablePPO adapter for finite scheduler macro actions.

No Stable-Baselines dependency is imported on the formal Client path.  A model
is loaded only after ``load()``/``predict()`` is explicitly requested, and test
or alternative inference engines can be injected through the small ``predict``
protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np

from learning.action_mask import validate_action_mask
from learning.action_space import coerce_discrete_action


class RLPolicyError(RuntimeError):
    """Base error for optional learned-policy failures."""


class ModelUnavailableError(RLPolicyError):
    """The configured model or its optional runtime is unavailable."""


class InvalidPolicyOutput(RLPolicyError):
    """A model returned an unsafe or malformed discrete action."""


@dataclass(frozen=True)
class PolicyPrediction:
    action_index: int
    inference_ms: float
    model_sha256: str | None = None
    deterministic: bool = True


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RLPolicy:
    """Lazy, deployment-safe adapter around a MaskablePPO-like model."""

    def __init__(
        self,
        model: Any | None = None,
        *,
        model_path: str | Path | None = None,
        expected_sha256: str | None = None,
        expected_schema_hash: str | None = None,
        loader: Callable[[str], Any] | None = None,
    ) -> None:
        self._model = model
        self.model_path = None if model_path is None else Path(model_path)
        self.expected_sha256 = expected_sha256
        self.expected_schema_hash = expected_schema_hash
        self._loader = loader
        self._model_sha256: str | None = None
        self.last_error: str | None = None

    @property
    def configured(self) -> bool:
        return self._model is not None or (
            self.model_path is not None and self.model_path.is_file()
        )

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def model_sha256(self) -> str | None:
        return self._model_sha256

    def _check_metadata(self, actual_model_sha256: str) -> None:
        if self.model_path is None or self.expected_schema_hash is None:
            return
        metadata_path = Path(f"{self.model_path}.metadata.json")
        if not metadata_path.is_file():
            raise ModelUnavailableError(
                f"policy metadata missing: {metadata_path}; cannot verify feature schema"
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelUnavailableError(
                f"cannot read policy metadata {metadata_path}: {exc}"
            ) from exc
        actual = metadata.get("observation_schema_hash")
        if actual != self.expected_schema_hash:
            raise ModelUnavailableError(
                "policy feature schema mismatch: "
                f"expected {self.expected_schema_hash}, got {actual!r}"
            )
        if metadata.get("metadata_schema_version") != "scheduler-model-metadata-v1":
            raise ModelUnavailableError("policy metadata schema is not approved")
        if metadata.get("algorithm") != "MaskablePPO":
            raise ModelUnavailableError("policy metadata algorithm is not MaskablePPO")
        if metadata.get("model_sha256") != actual_model_sha256:
            raise ModelUnavailableError(
                "policy model hash disagrees with its metadata"
            )

    def load(self) -> bool:
        """Load an explicitly configured model; never downloads model weights."""

        if self._model is not None:
            return True
        if self.model_path is None or not self.model_path.is_file():
            self.last_error = f"policy model not found: {self.model_path}"
            return False
        try:
            actual_hash = file_sha256(self.model_path)
            if self.expected_sha256 and actual_hash.lower() != self.expected_sha256.lower():
                raise ModelUnavailableError(
                    "policy model hash mismatch: "
                    f"expected {self.expected_sha256}, got {actual_hash}"
                )
            self._check_metadata(actual_hash)
            if self._loader is not None:
                model = self._loader(str(self.model_path))
            else:
                try:
                    from sb3_contrib import MaskablePPO
                except ImportError as exc:
                    raise ModelUnavailableError(
                        "loading an RL scheduler model requires the optional "
                        "training/runtime package 'sb3-contrib'; the heuristic "
                        "scheduler remains available without it"
                    ) from exc
                model = MaskablePPO.load(str(self.model_path))
            if model is None or not callable(getattr(model, "predict", None)):
                raise ModelUnavailableError("loaded object has no callable predict method")
            self._model = model
            self._model_sha256 = actual_hash
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self._model = None
            return False

    def predict(
        self,
        observation: Sequence[float] | np.ndarray,
        *,
        action_masks: Sequence[bool] | np.ndarray,
        deterministic: bool = True,
    ) -> PolicyPrediction:
        """Predict and validate one mask-constrained discrete candidate slot."""

        if self._model is None and not self.load():
            raise ModelUnavailableError(self.last_error or "policy model unavailable")
        obs = np.asarray(observation, dtype=np.float32)
        if obs.size == 0 or not bool(np.all(np.isfinite(obs))):
            raise InvalidPolicyOutput("policy observation must be finite and non-empty")
        mask = validate_action_mask(
            action_masks, len(action_masks), require_any=True
        )
        started = time.perf_counter()
        try:
            raw = self._model.predict(
                obs,
                action_masks=mask,
                deterministic=bool(deterministic),
            )
        except Exception as exc:
            raise RLPolicyError(f"policy inference failed: {exc}") from exc
        inference_ms = (time.perf_counter() - started) * 1000.0
        try:
            action_index = coerce_discrete_action(raw)
        except ValueError as exc:
            raise InvalidPolicyOutput(str(exc)) from exc
        if not 0 <= action_index < mask.size:
            raise InvalidPolicyOutput(
                f"policy action {action_index} is outside [0, {mask.size})"
            )
        if not bool(mask[action_index]):
            raise InvalidPolicyOutput(
                f"policy selected masked action index {action_index}"
            )
        return PolicyPrediction(
            action_index=action_index,
            inference_ms=inference_ms,
            model_sha256=self._model_sha256,
            deterministic=bool(deterministic),
        )

    def warmup(
        self,
        *,
        observation_size: int,
        action_count: int,
        iterations: int = 2,
    ) -> tuple[float, ...]:
        """Load and warm the inference backend before the guarded deadline starts.

        Warm-up uses synthetic finite input and cannot dispatch an action.  Any
        package, schema, dependency or prediction failure remains fail-closed.
        """

        if observation_size <= 0 or action_count <= 0 or iterations <= 0:
            raise ValueError("warmup dimensions and iterations must be positive")
        observation = np.zeros(int(observation_size), dtype=np.float32)
        action_mask = np.ones(int(action_count), dtype=np.bool_)
        return tuple(
            self.predict(
                observation,
                action_masks=action_mask,
                deterministic=True,
            ).inference_ms
            for _ in range(int(iterations))
        )


__all__ = [
    "InvalidPolicyOutput",
    "ModelUnavailableError",
    "PolicyPrediction",
    "RLPolicy",
    "RLPolicyError",
    "file_sha256",
]
