import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RunContext:
    run_id: str
    run_root: str
    user_config_snapshot_path: str
    index_path: str
    task_paths: dict = field(default_factory=dict)


def _sanitize_run_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return safe or "unknown"


def _build_run_id(*, problem: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = secrets.token_hex(2)
    return f"run_{problem}_{ts}_{suffix}"


def create_run_context(*, project_root: str, user_config_snapshot: dict) -> RunContext:
    problem = _sanitize_run_name(str(user_config_snapshot.get("problem", "")))
    run_id = ""
    run_root = ""
    max_attempts = 5
    for _ in range(max_attempts):
        run_id = _build_run_id(problem=problem)
        run_root = os.path.join(project_root, "result", run_id)
        try:
            os.makedirs(run_root, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError(
            f"Failed to create unique run directory after {max_attempts} attempts "
            f"for problem='{problem}'"
        )

    user_config_path = os.path.join(run_root, "user_config_snapshot.json")
    with open(user_config_path, "w") as f:
        json.dump(user_config_snapshot, f, indent=2)

    index_path = os.path.join(run_root, "index.json")
    index_payload = {
        "schema_version": "3.0",
        "run_id": run_id,
        "tasks": {},
    }
    with open(index_path, "w") as f:
        json.dump(index_payload, f, indent=2)

    return RunContext(
        run_id=run_id,
        run_root=run_root,
        user_config_snapshot_path=user_config_path,
        index_path=index_path,
    )


def _to_run_relative(context: RunContext, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.relpath(path, context.run_root)
    return path


def _to_abs(context: RunContext, path: str | None) -> str | None:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(context.run_root, path)


def update_run_index(context: RunContext, task: str, metadata_path: str) -> None:
    with open(context.index_path, "r") as f:
        payload = json.load(f)
    task_map = payload.get("tasks", {})
    if not isinstance(task_map, dict):
        task_map = {}
    task_map[task] = _to_run_relative(context, metadata_path)
    payload["tasks"] = task_map
    payload["schema_version"] = "3.0"
    with open(context.index_path, "w") as f:
        json.dump(payload, f, indent=2)


def get_task_metadata_path(context: RunContext, task: str) -> str | None:
    with open(context.index_path, "r") as f:
        payload = json.load(f)
    task_map = payload.get("tasks", {})
    if not isinstance(task_map, dict):
        return None
    return _to_abs(context, task_map.get(task))
