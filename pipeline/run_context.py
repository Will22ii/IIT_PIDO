import json
import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RunContext:
    run_id: str
    run_root: str
    user_config_snapshot_path: str
    index_path: str
    stage_paths: dict = field(default_factory=dict)


def _sanitize_run_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return safe or "unknown"


def create_run_context(*, project_root: str, user_config_snapshot: dict) -> RunContext:
    problem = _sanitize_run_name(str(user_config_snapshot.get("problem", "")))
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"run_{problem}_{ts}"
    run_root = os.path.join(project_root, "result", run_id)
    os.makedirs(run_root, exist_ok=True)

    user_config_path = os.path.join(run_root, "user_config_snapshot.json")
    with open(user_config_path, "w") as f:
        json.dump(user_config_snapshot, f, indent=2)

    index_path = os.path.join(run_root, "index.json")
    index_payload = {
        "run_id": run_id,
        "stages": {},
    }
    with open(index_path, "w") as f:
        json.dump(index_payload, f, indent=2)

    return RunContext(
        run_id=run_id,
        run_root=run_root,
        user_config_snapshot_path=user_config_path,
        index_path=index_path,
    )


def update_run_index(context: RunContext, stage: str, metadata_path: str) -> None:
    with open(context.index_path, "r") as f:
        payload = json.load(f)
    payload["stages"][stage] = metadata_path
    with open(context.index_path, "w") as f:
        json.dump(payload, f, indent=2)


def get_stage_metadata_path(context: RunContext, stage: str) -> str | None:
    with open(context.index_path, "r") as f:
        payload = json.load(f)
    return payload.get("stages", {}).get(stage)
