"""Standard-library-only official Server matrix runner for RL-2.

The remote host can orchestrate Docker without installing NumPy or Torch;
learning dependencies stay inside the isolated training container.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence


def collect_official_matrix(
    seeds: Sequence[int],
    *,
    runner: Callable[[int], Mapping[str, Any]],
    fail_fast: bool = True,
) -> dict[str, Any]:
    seed_values = [int(seed) for seed in seeds]
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be non-empty and unique")
    runs: list[dict[str, Any]] = []
    stopped_at = None
    for seed in seed_values:
        report = dict(runner(seed))
        report["seed"] = seed
        runs.append(report)
        if not bool(report.get("passed", False)):
            stopped_at = seed
            if fail_fast:
                break
    return {
        "passed": stopped_at is None
        and len(runs) == len(seed_values)
        and all(bool(item.get("passed")) for item in runs),
        "stopped_at_seed": stopped_at,
        "completed_seeds": [item["seed"] for item in runs],
        "runs": runs,
        "fail_fast": bool(fail_fast),
        "promotion_allowed": False,
    }


def _wait_for_log(path: Path, pattern: str, *, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.is_file() and pattern in path.read_text(
            encoding="utf-8", errors="replace"
        ):
            return True
        time.sleep(1.0)
    return False


def official_docker_runner(
    *,
    project: Path,
    output_root: Path,
    mode: str,
    model: Path | None = None,
    model_sha256: str | None = None,
    approval: Path | None = None,
    approval_sha256: str | None = None,
    server_ready_timeout_s: float = 180.0,
    client_timeout_s: float = 1800.0,
    expected_score: int = 160,
) -> Callable[[int], dict[str, Any]]:
    ctl = project / "material_sorting_task" / "scripts" / "competitionctl.sh"

    def _run(seed: int) -> dict[str, Any]:
        run_name = f"v2_multiseed_{seed}"
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        current = output_root / "current"
        current.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PROJECT"] = str(project)
        env["MATERIAL_ARTIFACT_ROOT"] = str(output_root)
        if model is not None:
            env["MATERIAL_RL_MODEL_RELATIVE_PATH"] = str(model)
            if model_sha256:
                env["MATERIAL_RL_MODEL_SHA256"] = model_sha256
        if approval is not None:
            env["MATERIAL_RL_APPROVAL_RELATIVE_PATH"] = str(approval)
            if approval_sha256:
                env["MATERIAL_RL_APPROVAL_SHA256"] = approval_sha256

        def stop() -> None:
            subprocess.run(
                ["bash", str(ctl), "stop"],
                check=False,
                env=env,
                cwd=str(project),
            )

        def snapshot_server_log() -> None:
            """Persist the complete detached Server log before removal."""

            snapshot = server_log.with_suffix(server_log.suffix + ".snapshot")
            try:
                with snapshot.open("w", encoding="utf-8", newline="\n") as stream:
                    result = subprocess.run(
                        ["docker", "logs", "material_sorting_server"],
                        check=False,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                if result.returncode == 0:
                    snapshot.replace(server_log)
                elif snapshot.exists():
                    snapshot.unlink()
            except OSError:
                if snapshot.exists():
                    snapshot.unlink()

        stop()
        server_log = run_dir / f"server_{run_name}.log"
        client_log = run_dir / f"client_{run_name}.log"
        server = subprocess.run(
            ["bash", str(ctl), "server-detached", run_name, str(seed)],
            check=False,
            env=env,
            cwd=str(project),
        )
        if server.returncode != 0 or not _wait_for_log(
            server_log, "referee enabled", timeout_s=server_ready_timeout_s
        ):
            stop()
            return {
                "passed": False,
                "failures": ["server did not become ready"],
                "client_log": str(client_log),
                "server_log": str(server_log),
            }

        client_returncode: int | None = None
        timed_out = False
        client_process: subprocess.Popen[bytes] | None = None
        try:
            client_process = subprocess.Popen(
                ["bash", str(ctl), "client", run_name, mode],
                env=env,
                cwd=str(project),
            )
            deadline = time.time() + client_timeout_s
            while client_process.poll() is None:
                if client_log.is_file():
                    live_text = client_log.read_text(
                        encoding="utf-8", errors="replace"
                    ).casefold()
                    if (
                        (
                            "controller=finished" in live_text
                            and f"score={expected_score}" in live_text
                        )
                        or "controller=blocked" in live_text
                        or "controller=safe_hold" in live_text
                    ):
                        break
                if time.time() >= deadline:
                    timed_out = True
                    break
                time.sleep(1.0)
        finally:
            snapshot_server_log()
            stop()
            if client_process is not None:
                try:
                    client_returncode = client_process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    client_process.terminate()
                    try:
                        client_returncode = client_process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        client_process.kill()
                        client_returncode = client_process.wait(timeout=5.0)

        if client_log.is_file():
            shutil.copy2(client_log, current / "client.log")
        text = (
            client_log.read_text(encoding="utf-8", errors="replace")
            if client_log.is_file()
            else ""
        )
        folded = text.casefold()
        finished = "controller=finished" in text and f"score={expected_score}" in text
        failures: list[str] = []
        if timed_out:
            failures.append(f"client timeout after {client_timeout_s:.0f}s")
        if client_returncode not in {0, None} and not finished:
            failures.append(f"client exit={client_returncode}")
        if not finished:
            failures.append(f"missing controller=finished score={expected_score}")
        for marker in (
            "controller=blocked",
            "controller=safe_hold",
            "executor error",
            "unsafe collision",
        ):
            if marker in folded:
                failures.append(marker)
        return {
            "passed": not failures,
            "failures": failures,
            "client_log": str(client_log),
            "server_log": str(server_log),
            "score": expected_score if finished else None,
        }

    return _run


__all__ = ["collect_official_matrix", "official_docker_runner"]
