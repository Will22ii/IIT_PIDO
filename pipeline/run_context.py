import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RunContext:
    # Backend/user-facing identifier is run_root. RunContext is the internal
    # execution object that resolves paths and updates index.json.
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
    """Create a new run directory and initialize index.json.

    Use this when backend starts a new execution. Existing executions should be
    opened with load_run_context(run_root=...).
    """
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
    with open(user_config_path, "w", encoding="utf-8") as f:
        json.dump(user_config_snapshot, f, indent=2, ensure_ascii=False)

    index_path = os.path.join(run_root, "index.json")
    index_payload = {
        "schema_version": "3.0",
        "run_id": run_id,
        "tasks": {},
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, indent=2, ensure_ascii=False)

    return RunContext(
        run_id=run_id,
        run_root=run_root,
        user_config_snapshot_path=user_config_path,
        index_path=index_path,
    )


def load_run_context(*, run_root: str) -> RunContext:
    """Open an existing run directory passed by backend.

    This does not decide whether reuse is allowed. It only validates that the
    run_root has the minimum files needed to act as a run context.
    """
    run_root_abs = os.path.abspath(str(run_root))
    if not os.path.isdir(run_root_abs):
        raise FileNotFoundError(f"Run root not found: {run_root_abs}")

    index_path = os.path.join(run_root_abs, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Run index not found: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid run index payload: {index_path}")

    run_id = str(payload.get("run_id") or os.path.basename(run_root_abs))
    user_config_snapshot_path = os.path.join(run_root_abs, "user_config_snapshot.json")
    if not os.path.exists(user_config_snapshot_path):
        raise FileNotFoundError(
            f"Run user config snapshot not found: {user_config_snapshot_path}"
        )

    task_paths = payload.get("tasks", {})
    if not isinstance(task_paths, dict):
        task_paths = {}

    return RunContext(
        run_id=run_id,
        run_root=run_root_abs,
        user_config_snapshot_path=user_config_snapshot_path,
        index_path=index_path,
        task_paths=dict(task_paths),
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
    with open(context.index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    task_map = payload.get("tasks", {})
    if not isinstance(task_map, dict):
        task_map = {}
    task_map[task] = _to_run_relative(context, metadata_path)
    payload["tasks"] = task_map
    payload["schema_version"] = "3.0"
    with open(context.index_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_task_metadata_path(context: RunContext, task: str) -> str | None:
    with open(context.index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    task_map = payload.get("tasks", {})
    if not isinstance(task_map, dict):
        return None
    return _to_abs(context, task_map.get(task))
